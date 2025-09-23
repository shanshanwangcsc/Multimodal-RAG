import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import time


# Simple RAG class for document indexing and retrieval
class VisualRAG:
    def __init__(self, retriever_model, retriever_processor, vl_model, vl_processor):
        self.retriever_model = retriever_model
        self.retriever_processor = retriever_processor
        self.vl_model = vl_model
        self.vl_processor = vl_processor
        self.indexed_embeddings = None
        self.indexed_images = []
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def index_documents(self, image_paths):
        """Index documents from a list of image paths"""

        image_files = [f for f in os.listdir(image_paths) if f.endswith('.png')]
        images = [Image.open(os.path.join(image_paths, f)) for f in image_files]

        self.indexed_images = images
        self.filenames=image_files

        # Process images in batches if needed
        batch_size = 4  # Adjust based on your GPU memory
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            processed_images = self.retriever_processor.process_images(batch_images).to(self.retriever_model.device)

            with torch.no_grad():
                embeddings = self.retriever_model(**processed_images)

            all_embeddings.append(embeddings)

        # Combine embeddings if needed
        if len(all_embeddings) > 1:
            # This is a simplified approach - you may need to adjust based on the actual embedding structure
            self.indexed_embeddings = torch.cat(all_embeddings, dim=0)
        else:
            self.indexed_embeddings = all_embeddings[0]

        return len(images)

    def retrieve(self, query, k=5):
        """Retrieve relevant documents based on a query"""
        if self.indexed_embeddings is None:
            raise ValueError("No documents have been indexed yet")

        # Process the query
        processed_query = self.retriever_processor.process_queries([query]).to(self.retriever_model.device)

        # Get query embeddings
        with torch.no_grad():
            query_embeddings = self.retriever_model(**processed_query)

        # Score documents
        scores = self.retriever_processor.score_multi_vector(query_embeddings, self.indexed_embeddings)

        # Get top-k results
        if scores.dim() > 1:
            scores = scores.squeeze(0)  # Remove batch dimension if present

        top_k_scores, top_k_indices = torch.topk(scores, min(k, len(scores)))


        # Return results
        results = []
        for i, (idx, score) in enumerate(zip(top_k_indices.tolist(), top_k_scores.tolist())):
            results.append({
                'filename': self.filenames[idx],
                "page_image": self.indexed_images[idx],
                "score": score
            })

        return results

    def answer_query(self, query, k=3):
        """Retrieve relevant documents and generate an answer using Qwen2.5-VL"""
        # Retrieve relevant documents
        retrieved_results = self.retrieve(query, k=k)


        # Prepare messages for Qwen2.5-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"I have a question about some documents: {query}"}
                ]
            }
        ]

        # Add retrieved pages to the message
        for result in retrieved_results:
            messages[0]["content"].append({
                "type": "image",
                "image": result["page_image"]
            })

        # Add final instruction
        messages[0]["content"].append({
            "type": "text",
            "text": "Based on the document images provided, please answer my question in detail."
        })

        # Process the messages
        text = self.vl_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.vl_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.vl_model.device)

        # Generate response
        generated_ids = self.vl_model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.vl_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return {
            "answer": output_text,
            "retrieved_documents": retrieved_results
        }

# Main code
def main():
    # Load the colQwen2.5 model for retrieval
    print("Loading colQwen2.5 model for retrieval...")
    retriever_model = ColQwen2_5.from_pretrained(
        "vidore/colqwen2.5-v0.2",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    ).eval()
    retriever_processor = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-v0.2")

    # Load Qwen2.5-VL-7B-Instruct for answer generation
    print("Loading Qwen2.5-VL-7B-Instruct for answer generation...")
    vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    )
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    vl_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",min_pixels=min_pixels, max_pixels=max_pixels)

    # Create a Visual RAG instance
    visual_rag = VisualRAG(
        retriever_model=retriever_model,
        retriever_processor=retriever_processor,
        vl_model=vl_model,
        vl_processor=vl_processor
    )

    # Document paths
    document_paths = "images/"
    num_indexed = visual_rag.index_documents(document_paths)
    print(f"Indexed {num_indexed} documents")

    print("🤖 Qwen2.5-VL Retrieval QA ready. Type 'exit' to quit.\n")

    while True:
        query = input("❓ You: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break
        start = time.perf_counter()
        result = visual_rag.answer_query(query, k=3)
        end = time.perf_counter()
        print(f"\n⏱️ Generation time: {end - start:.2f} sec")
        print(f"\nAnswer: {result['answer']}")

        print("\nRetrieved documents:")
        for i, doc in enumerate(result['retrieved_documents']):
            print(f"Document {i+1}: filename = {doc['filename']}, Score = {doc['score']:.4f}")



if __name__ == "__main__":
    main()
## Questions
# How to bring CSC closer to researchers?
# What Do Ambassadors Do?
# Whom should I get in touch if I am interested in the ambassador?
# Which organisation is Longying Xiao from?
# Which organisation is Henri Wallen from?
# Who is from VTT / Oulu / Research IT Services?
# When does the service desk open?
# When was this blog written?
# Where is CSC located?
