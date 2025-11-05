import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
import os
import shutil
import tempfile

# Embedding model (keep loaded globally for speed)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={"device": 0})

# Qwen2.5 Model (load once globally)
model_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

qwen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.2,
    device_map="auto"
)
llm = HuggingFacePipeline(pipeline=qwen_pipeline)

# Prompt
template = """You are a helpful assistant.
Answer the question based only on the provided context.
Give ONLY the final answer, no reasoning or explanation.

Context:
{context}

Question:
{question}

Final Answer:"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

# Global state
qa_chain = None  # will be rebuilt after PDF upload

def process_pdf(pdf_file):
    """Build vector store + retriever from uploaded PDF."""
    global qa_chain

    # Temp directory for chroma store
    persist_dir = tempfile.mkdtemp()

    # 1. Load PDF
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # 3. Store in Chroma
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    db.persist()
    retriever = db.as_retriever()

    # 4. Build RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}

    )

    return "✅ PDF processed successfully. You can now ask questions!"

def chatbot(user_input, history):
    global qa_chain
    if qa_chain is None:
        return history + [(user_input, "⚠️ Please upload a PDF first.")], history

    #raw_answer = qa_chain.run(user_input)
    result = qa_chain({"query": user_input})
    raw_answer = result["result"]
    sources = result.get("source_documents", [])
    if "Final Answer:" in raw_answer:
        final_answer = raw_answer.split("Final Answer:")[-1].strip()
    else:
        final_answer = raw_answer.strip()

    history = history + [(user_input, final_answer)]
    return history, history

with gr.Blocks() as demo:
    gr.Markdown("# 📄 text-only RAG with Qwen2.5 LLM")

    with gr.Row():
        pdf_upload = gr.File(label="Upload a PDF", file_types=[".pdf"])
        status = gr.Label()

    chatbot_ui = gr.Chatbot([], elem_id="chatbot")
    msg = gr.Textbox(placeholder="Ask a question about the PDF...")
    clear = gr.Button("Clear Chat")

    state = gr.State([])

    # Events
    pdf_upload.upload(process_pdf, pdf_upload, status)
    msg.submit(chatbot, [msg, state], [chatbot_ui, state])
    clear.click(lambda: ([], []), None, [chatbot_ui, state])

demo.launch(share=True)

## Usage:
# module use /appl/local/csc/modulefiles/
# module load pytorch/2.7
# export SING_IMAGE='/scratch/project_462000824/shanshan/csc-popper.sif'
# export HF_HOME="/scratch/project_462000824/${USER}/hf-cache"
# mkdir -p $HF_HOME
# srun --account=project_462000824 --partition=dev-g --ntasks=1 --cpus-per-task=14 --gpus-per-node=1 --mem=160G --time=02:30:00 --nodes=1 --pty bash
# python Normal_RAG.py
