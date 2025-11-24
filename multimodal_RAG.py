import torch
import os
import tempfile
from PIL import Image
import gradio as gr
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pdf2image import convert_from_path
import time


# ---------- Visual RAG Class ----------
class VisualRAG:
    def __init__(self, retriever_model, retriever_processor, vl_model, vl_processor):
        self.retriever_model = retriever_model
        self.retriever_processor = retriever_processor
        self.vl_model = vl_model
        self.vl_processor = vl_processor
        self.indexed_embeddings = None
        self.indexed_images = []
        self.filenames = []
    def index_documents(self, path):
        """Index documents from a PDF file or a folder of images"""

        images = []
        filenames = []

        if os.path.isdir(path):  # Folder of images
            image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg'))]
            for f in image_files:
                images.append(Image.open(os.path.join(path, f)))
                filenames.append(f)

        elif path.lower().endswith(".pdf"):  # Convert PDF to images
            pages = convert_from_path(path, dpi=300)
            for i, page in enumerate(pages):
                img_name = f"page_{i+1}.png"
                images.append(page)
                filenames.append(img_name)

        else:
            raise ValueError("Please provide either a PDF file or an image folder")

        self.indexed_images = images
        self.filenames = filenames

        # Process in batches
        batch_size = 8
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            processed = self.retriever_processor.process_images(batch_images).to(self.retriever_model.device)
            with torch.no_grad():
                emb = self.retriever_model(**processed)
            all_embeddings.append(emb)

        self.indexed_embeddings = torch.cat(all_embeddings, dim=0)
        return len(images)

    def retrieve(self, query, k=5):
        """Retrieve top-k relevant documents"""
        if self.indexed_embeddings is None:
            return []

        processed_query = self.retriever_processor.process_queries([query]).to(self.retriever_model.device)
        with torch.no_grad():
            query_embeddings = self.retriever_model(**processed_query)

        scores = self.retriever_processor.score_multi_vector(query_embeddings, self.indexed_embeddings)
        if scores.dim() > 1:
            scores = scores.squeeze(0)

        top_k_scores, top_k_indices = torch.topk(scores, min(k, len(scores)))

        results = []
        for idx, score in zip(top_k_indices.tolist(), top_k_scores.tolist()):
            results.append({
                "filename": self.filenames[idx],
                "page_image": self.indexed_images[idx],
                "score": score
            })
        return results

    def answer_query(self, query, k=3):
        """Retrieve + answer with Qwen2.5-VL"""
        retrieved = self.retrieve(query, k=k)
        if not retrieved:
            return "⚠️ No documents indexed yet.", []

        messages = [{"role": "user", "content": [{"type": "text", "text": f"Question: {query}"}]}]

        # Attach top-k retrieved images
        for r in retrieved:
            messages[0]["content"].append({"type": "image", "image": r["page_image"]})

        messages[0]["content"].append({"type": "text", "text": "Answer based on the provided document images."})

        text = self.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.vl_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.vl_model.device)

        with torch.no_grad():
            generated_ids = self.vl_model.generate(**inputs, max_new_tokens=512)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        answer = self.vl_processor.batch_decode(trimmed, skip_special_tokens=True)[0]

        return answer, retrieved


# ---------- Load Models Once ----------
print("Loading ColQwen2.5 retriever...")
retriever_model = ColQwen2_5.from_pretrained(
    "vidore/colqwen2.5-v0.2",
    #torch_dtype=torch.bfloat16,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()
retriever_processor = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-v0.2")

print("Loading Qwen2.5-VL for QA...")
vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    #torch_dtype=torch.bfloat16,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
)
vl_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

visual_rag = VisualRAG(retriever_model, retriever_processor, vl_model, vl_processor)


# ---------- Gradio UI ----------
def upload_and_index(file):
    if file is None:
        return "⚠️ Please upload a PDF or folder of images."
    #path = file.name if hasattr(file, "name") else file
    count = visual_rag.index_documents(file)
    return f"✅ Indexed {count} document pages."


# def chat_fn(query, history):
#     if not query.strip():
#         return history, history
#     start = time.perf_counter()
#     answer, docs = visual_rag.answer_query(query, k=3)
#     end = time.perf_counter()

#     retrieved_previews = [f"{d['filename']} (score={d['score']:.3f})" for d in docs]
#     history = history + [(query, f"{answer}\n\nRetrieved: {retrieved_previews}\n⏱ {end-start:.2f}s")]
#     return history, history

# ---------- Gradio Chat Function ----------
from gradio import ChatMessage

def chat_fn(query, history):
    if not query.strip():
        return history, history

    start = time.perf_counter()
    answer, docs = visual_rag.answer_query(query, k=3)
    end = time.perf_counter()

    # Build retrieved preview list
    retrieved_previews = "\n".join([f"- {d['filename']} (score={d['score']:.3f})" for d in docs])
    bot_content = f"{answer}\n\nRetrieved documents:\n{retrieved_previews}\n⏱ {end-start:.2f}s"

    # Initialize history as a list of dicts with 'role' and 'content'
    if history is None:
        history = []

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": bot_content})

    return history, history


# with gr.Blocks() as demo:
#     gr.Markdown("# 📚 Multimodal RAG with ColQwen2.5 + Qwen2.5-VL")

#     with gr.Row():
#         file_input = gr.File(label="Upload PDF or image folder", type="filepath")
#         status = gr.Label()

#     chatbot_ui = gr.Chatbot([], elem_id="chatbot")
#     msg = gr.Textbox(placeholder="Ask a question about the document...")
#     clear = gr.Button("Clear Chat")
#     state = gr.State([])

#     file_input.upload(upload_and_index, file_input, status)
#     msg.submit(chat_fn, [msg, state], [chatbot_ui, state])
#     clear.click(lambda: ([], []), None, [chatbot_ui, state])

# demo.launch(share=True)

with gr.Blocks() as demo:
    gr.Markdown("# 📚 Multimodal RAG with ColQwen2.5 + Qwen2.5-VL")

    with gr.Row():
        file_input = gr.File(label="Upload PDF or image folder", type="filepath")
        status = gr.Label()

    chatbot_ui = gr.Chatbot([], elem_id="chatbot")  # new messages format
    msg = gr.Textbox(placeholder="Ask a question about the document...")
    clear = gr.Button("Clear Chat")
    state = gr.State([])  # history as list of dicts

    file_input.upload(upload_and_index, file_input, status)
    msg.submit(chat_fn, [msg, state], [chatbot_ui, state])
    clear.click(lambda: [], None, [chatbot_ui, state])

demo.launch(share=True)
