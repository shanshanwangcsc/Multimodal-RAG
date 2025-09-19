import os
import gc
import torch
import json
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor,BitsAndBytesConfig

from qwen_vl_utils import process_vision_info
import time
from transformers.utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor


IMAGE_FOLDER = "./images"
TOP_K = 3
MAX_NEW_TOKENS = 256

# Load embeddings + metadata
embeddings = torch.load("merged_files/embeddings_merged.pt")
print(len(embeddings))
with open("merged_files/metadata_merged.json") as f:
    metadata = json.load(f)
print(f"Loaded {len(embeddings)} embeddings with metadata.")

# -----------------------------
# Load Retrieval Model
# -----------------------------
retrieval_model = ColQwen2_5.from_pretrained(
        "vidore/colqwen2.5-v0.2",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    ).eval()
retrieval_processor = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-v0.2",use_fast=False)



embeddings = [emb.to(retrieval_model.device) for emb in embeddings]

# -----------------------------
# Load Qwen2.5-VL QA Model
# -----------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
qa_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(

    "Qwen/Qwen2.5-VL-7B-Instruct",
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
    device_map="auto",
)


min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",min_pixels=min_pixels, max_pixels=max_pixels)

# -----------------------------
# Retrieval
# -----------------------------
def retrieve(query, top_k=TOP_K):

    batch_queries = retrieval_processor.process_queries([query]).to(retrieval_model.device)

    with torch.no_grad():
        query_embeddings = retrieval_model(**batch_queries)
        start = time.perf_counter()
        scores = retrieval_processor.score_multi_vector(query_embeddings, embeddings)
        end = time.perf_counter()
        print(f"\n⏱️ retrieving takes: {end - start:.2f} sec")
    topk = torch.topk(scores[0], k=top_k)
    return [(idx, metadata[idx], score.item()) for idx, score in zip(topk.indices, topk.values)]

# -----------------------------
# Answer Generation
# -----------------------------
def generate_answer(query, top_k=TOP_K, max_new_tokens=MAX_NEW_TOKENS):
    # Step 1: Retrieve images
    top_results = retrieve(query, top_k=top_k)
    images = [Image.open(os.path.join(IMAGE_FOLDER, fname)).convert("RGB")  for _, fname, _ in top_results]


    # Step 2: Prepare input
    chat = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on retrieved document images."},
        {"role": "user", "content": [{"type": "text", "text": query}, *[{"type": "image", "image": img} for img in images]]},
    ]

    text = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(chat)

    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(qa_model.device)

    # Step 3: Generate
    start = time.perf_counter()

    generated_ids = qa_model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    answer = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    end = time.perf_counter()
    print(f"\n⏱️ generating time: {end - start:.2f} sec")
    return answer, top_results

# -----------------------------
# Simple CLI
# -----------------------------
if __name__ == "__main__":

    print("🤖 Qwen2.5-VL Retrieval QA ready. Type 'exit' to quit.\n")

    while True:
        user_query = input("❓ You: ").strip()
        if user_query.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        start = time.perf_counter()
        answer, retrieved = generate_answer(user_query)
        end = time.perf_counter()

        print("\n🔍 Retrieved:")
        for _, fname, score in retrieved:
            print(f"   {fname} (score={score:.4f})")

        print("\n💡 Assistant:")
        print(answer)
        print(f"\n⏱️ Total time: {end - start:.2f} sec")
        print("\n" + "-"*50 + "\n")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

# Questions and Answers:
# 1. Who are the authors of paper “A distributed soil moisture, temperature and infiltrometer dataset for permeable pavements and green spaces”?
# Axel Schaffitel, Tobias Schuetz, and Markus Weiler

# 2. Where can A distributed soil moisture, temperature and infiltrometer dataset be accessed?
# The dataset is freely available from the FreiDok plus data repository at https://freidok.uni-freiburg.de/data/151573
# and https://doi.org/10.6094/UNIFR/151573 (Schaffitel et al., 2019).

# 3. what are the Radiative effects of ozone?
# The answers are collecated from various papers which discussed such topics, such as acp-18-6637-2018, acp-24-3613-2024
