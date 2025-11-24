# --- External Libraries ---
import tempfile
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import gradio as gr
import torch

# --- LangChain Core Components ---
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA

# --- Device setup ---
device = 0 if torch.cuda.is_available() else -1

# --- Embeddings ---
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": f"cuda:{device}" if device != -1 else "cpu"}
)

# --- LLM setup ---
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

# --- Prompt template ---
template = """You are a helpful assistant.
Answer the question based only on the provided context.
Give ONLY the final answer, no reasoning or explanation.

Context:
{context}

Question:
{question}

Final Answer:"""

prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# --- Global state ---
qa_chain = None

# --- PDF processing ---
def process_pdf(pdf_file):
    """Build vector store + retriever from uploaded PDF."""
    global qa_chain

    persist_dir = tempfile.mkdtemp()

    # 1. Load PDF
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # 3. Store in Chroma
    db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)
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

# # --- Chatbot ---
# def chatbot(user_input, history):
#     """
#     Processes user input and updates history.
#     History must be a list of tuples (user_message_dict, assistant_message_dict)
#     """
#     from gradio import ChatMessage  # safer option

#     # Ensure history is always a list
#     history = list(history)

#     if qa_chain is None:
#         warning_msg = ChatMessage(role="assistant", content="⚠️ Please upload a PDF first.")
#         new_turn = (
#             ChatMessage(role="user", content=user_input),
#             warning_msg
#         )
#         return history + [new_turn], history + [new_turn]

#     # Run the RAG chain
#     result = qa_chain({"query": user_input})
#     raw_answer = result.get("result", "").strip()

#     # Extract the final answer
#     if "Final Answer:" in raw_answer:
#         final_answer = raw_answer.split("Final Answer:")[-1].strip()
#     else:
#         final_answer = raw_answer

#     new_turn = (
#         ChatMessage(role="user", content=user_input),
#         ChatMessage(role="assistant", content=final_answer)
#     )

#     return history + [new_turn], history + [new_turn]

def chatbot(user_input, history):
    from gradio import ChatMessage

    history = list(history)

    if qa_chain is None:
        history.append(ChatMessage(role="user", content=user_input))
        history.append(ChatMessage(role="assistant", content="⚠️ Please upload a PDF first."))
        return history, history

    result = qa_chain({"query": user_input})
    raw_answer = result.get("result", "").strip()

    final_answer = raw_answer.split("Final Answer:")[-1].strip() if "Final Answer:" in raw_answer else raw_answer

    # Add user + assistant messages sequentially
    history.append(ChatMessage(role="user", content=user_input))
    history.append(ChatMessage(role="assistant", content=final_answer))

    return history, history


# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 📄 Text-only RAG with Qwen2.5 LLM")

    with gr.Row():
        pdf_upload = gr.File(label="Upload a PDF", file_types=[".pdf"])
        status = gr.Label()

    chatbot_ui = gr.Chatbot([], elem_id="chatbot")
    msg = gr.Textbox(placeholder="Ask a question about the PDF...")
    clear = gr.Button("Clear Chat")
    state = gr.State([])  # stores list of tuples of ChatMessage objects

    # Events
    pdf_upload.upload(process_pdf, pdf_upload, status)
    msg.submit(chatbot, [msg, state], [chatbot_ui, state], concurrency_limit=1)
    clear.click(lambda: ([], []), None, [chatbot_ui, state])

demo.launch(share=True)
