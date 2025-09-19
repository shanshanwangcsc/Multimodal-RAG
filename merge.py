#!/usr/bin/env python3
import os
import json
import torch
import argparse
from glob import glob

def merge_embeddings(in_dir, out_dir, out_embeddings, out_metadata):
    # 1) Find all embedding shards
    shard_files = sorted(glob(os.path.join(in_dir, "embeddings_rank*_batch*.pt")))
    if not shard_files:
        raise ValueError(f"No embedding shards found in {in_dir}")

    print(f"Found {len(shard_files)} embedding shards.")

    # 2) Concatenate embeddings
    all_embeddings = []
    for f in shard_files:
        batch_emb = torch.load(f, map_location="cpu")
        for emb in batch_emb:  # each [seq_len, 128]
            all_embeddings.append(emb)



    # 3) Save merged embeddings
    os.makedirs(out_dir, exist_ok=True)
    torch.save(all_embeddings, os.path.join(out_dir, out_embeddings))
    print(f"Saved merged embeddings to {os.path.join(out_dir, out_embeddings)}")

    # 4) Merge metadata JSONs
    meta_files = sorted(glob(os.path.join(in_dir, "metadata_rank*.json")))
    all_metadata = []
    for mf in meta_files:
        with open(mf, "r") as f:
            all_metadata.extend(json.load(f))

    print(f"Final metadata count: {len(all_metadata)}")

    with open(os.path.join(out_dir,out_metadata), "w") as f:
        json.dump(all_metadata, f, indent=2)
    print(f"Saved merged metadata to {os.path.join(out_dir,out_metadata)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge DDP embedding shards")
    parser.add_argument("--in_dir", type=str, required=True, help="Folder containing rank shard files")
    parser.add_argument("--out_dir", type=str, required=True, help="Folder containing merged embeddings and metadata files")
    parser.add_argument("--out_embeddings", type=str, default="embeddings_merged.pt")
    parser.add_argument("--out_metadata", type=str, default="metadata_merged.json")
    args = parser.parse_args()

    merge_embeddings(args.in_dir,args.out_dir, args.out_embeddings, args.out_metadata)
