#!/usr/bin/env python3
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers.utils.import_utils import is_flash_attn_2_available

# --------------- Dataset -----------------
class PDFPageDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.all_images = sorted([f for f in os.listdir(folder) if f.endswith(".png")])

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_name = self.all_images[idx]
        img_path = os.path.join(self.folder, img_name)
        try:
            image_data = Image.open(img_path).convert("RGB")
            return image_data, img_name
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise

def collate_fn(batch):
    images = [item[0] for item in batch]
    metadata = [item[1] for item in batch]
    return images, metadata

# --------------- Utils -----------------
def is_distributed():
    return dist.is_available() and dist.is_initialized()

def print_once(msg):
    if not is_distributed() or dist.get_rank() == 0:
        print(msg, flush=True)

def save_json_ranklocal(obj, path, rank):
    base, ext = os.path.splitext(path)
    rank_path = f"{base}_rank{rank}{ext}"
    with open(rank_path, "w") as f:
        json.dump(obj, f, indent=2)
    return rank_path

def setup_device(local_rank):
    """Setup device and distributed training for data parallelism"""
    if torch.cuda.is_available() and local_rank >= 0:
        # Initialize process group for data distribution
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        ddp = True
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ddp = False

    return device, ddp

def main(args):
    # 1) Setup distributed for data parallelism
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device, ddp = setup_device(local_rank)

    rank = dist.get_rank() if ddp else 0
    world_size = dist.get_world_size() if ddp else 1

    # 2) IO
    os.makedirs(args.out_dir, exist_ok=True)
    print_once(f"World size: {world_size} | Using data parallelism: {ddp}")
    print_once(f"Images folder: {args.images_folder}")
    print_once(f"Output dir: {args.out_dir}")

    # 3) Data - Use DistributedSampler to split data across GPUs
    try:
        dataset = PDFPageDataset(args.images_folder)
        total_pages = len(dataset)
        if total_pages == 0:
            print_once("No .png files found. Exiting.")
            return
        print_once(f"Total pages detected: {total_pages}")
    except Exception as e:
        print_once(f"Error loading dataset: {e}")
        return

    if ddp:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0
    )

    # 4) Model - Optimized for fast inference
    try:
        # Load model with optimized settings
        model = ColQwen2_5.from_pretrained(
            "vidore/colqwen2.5-v0.2",
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "sdpa",
        )

        for param in model.parameters():
            param.requires_grad = False

        model = model.to(device)
        model.eval()

        processor = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-v0.2",use_fast=False)

    except Exception as e:
        print_once(f"Error loading model: {e}")
        return

    # 5) Inference loop - Optimized for speed
    metadata_accum = []
    total_processed = 0
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    if torch.cuda.is_available():
        start_time.record()

    with torch.no_grad():
        if ddp:
            loader.sampler.set_epoch(0)

        for batch_idx, (batch_images, batch_names) in enumerate(loader):
            try:

                inputs = processor.process_images(images=batch_images).to(model.device)

                # Forward pass
                outputs = model(**inputs)
                image_embeddings = outputs.detach().cpu()

                # Save batch with efficient storage
                shard_path = os.path.join(args.out_dir, f"embeddings_rank{rank}_batch{batch_idx:06d}.pt")
                torch.save(image_embeddings, shard_path)

                metadata_accum.extend(batch_names)
                total_processed += len(batch_names)

                # Progress reporting
                if (batch_idx + 1) % args.log_every == 0:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    print_once(
                        f"[Rank {rank}] Processed {total_processed} pages | "
                        f"Batch {batch_idx + 1} | Saved: {os.path.basename(shard_path)}"
                    )

            except Exception as e:
                print_once(f"Error processing batch {batch_idx}: {e}")
                continue

    # 6) Timing and performance metrics
    if torch.cuda.is_available():
        end_time.record()
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # seconds
        pages_per_second = total_processed / elapsed_time if elapsed_time > 0 else 0
        print_once(f"[Rank {rank}] Processed {total_processed} pages in {elapsed_time:.2f}s ({pages_per_second:.2f} pages/s)")

    # 7) Save metadata
    try:
        meta_path = os.path.join(args.out_dir, "metadata.json")
        rank_meta_path = save_json_ranklocal(metadata_accum, meta_path, rank)
        print_once(f"[Rank {rank}] Saved {len(metadata_accum)} embeddings to {rank_meta_path}")
    except Exception as e:
        print_once(f"Error saving metadata: {e}")

    # 8) Cleanup
    if ddp:
        dist.barrier()
        dist.destroy_process_group()

    print_once("All ranks finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ColQwen2.5 fast image embedding with data parallelism")
    parser.add_argument("--images_folder", type=str, required=True, help="Folder with .png pages")
    parser.add_argument("--out_dir", type=str, default="./embeddings_out", help="Where to write shard files")
    parser.add_argument("--batch_size", type=int, default=64, help="Tune based on VRAM")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers (2-4 per GPU is optimal)")
    parser.add_argument("--log_every", type=int, default=20, help="Logger frequency in batches")


    args = parser.parse_args()

    main(args)
