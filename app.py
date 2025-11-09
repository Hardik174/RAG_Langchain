import os
import shutil
import gradio as gr
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch
from src.embedding import EmbeddingPipeline

# ------------------------------
# Initialize pipeline components
# ------------------------------
DATA_DIR = "data"
FAISS_DIR = "faiss_store"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)

rag_search = RAGSearch()
store = FaissVectorStore(FAISS_DIR)

# ------------------------------
# Core functions
# ------------------------------
def process_query(query):
    """Perform RAG search and summarization."""
    if not query.strip():
        return "⚠️ Please enter a valid query."
    try:
        summary = rag_search.search_and_summarize(query, top_k=3)
        return f"🧠 **Summary:**\n\n{summary}"
    except Exception as e:
        return f"❌ Error: {e}"

def upload_file(file):
    """Handle file upload and place it in the correct data folder."""
    if file is None:
        return "⚠️ Please upload a file."

    filename = os.path.basename(file.name)
    ext = filename.split('.')[-1].lower()

    # Route files based on extension
    target_dir = os.path.join(DATA_DIR, {
        "pdf": "pdfs",
        "txt": "texts",
        "csv": "csvs",
        "docx": "docs"
    }.get(ext, "others"))

    os.makedirs(target_dir, exist_ok=True)
    dest_path = os.path.join(target_dir, filename)
    shutil.copy(file.name, dest_path)

    # Update embeddings
    docs = load_all_documents(DATA_DIR)
    chunks = EmbeddingPipeline().chunk_documents(docs)
    chunk_vectors = EmbeddingPipeline().embed_chunks(chunks)
    store.build_from_documents(docs)

    return f"✅ File '{filename}' uploaded successfully and data store updated!"

# ------------------------------
# Gradio UI setup
# ------------------------------
with gr.Blocks(theme=gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="orange",
    neutral_hue="gray"
)) as demo:

    gr.Markdown(
        """
        <h1 style='text-align:center; color:#1abc9c;'>🧠 RAG-Powered Knowledge Chatbot</h1>
        <p style='text-align:center; color:#555;'>
        Ask anything about your uploaded documents — from technical concepts to API log analysis!
        </p>
        """
    )

    with gr.Row():
        query_box = gr.Textbox(
            label="💬 Ask a Question",
            placeholder="e.g. Explain Transformer Architecture...",
            lines=2
        )
        submit_btn = gr.Button("🔍 Search", variant="primary")

    chatbot_output = gr.Markdown("### 🤖 Awaiting your query...")

    with gr.Accordion("📂 Upload Files to Expand Knowledge Base", open=False):
        file_input = gr.File(label="Upload new files (PDF, TXT, DOCX, CSV)")
        upload_status = gr.Markdown()
        file_input.upload(upload_file, inputs=file_input, outputs=upload_status)

    gr.Markdown("### 🔖 Popular Questions on API Log Analysis")

    with gr.Row():
        pyq1 = gr.Button("What is an API Log?")
        pyq2 = gr.Button("How to analyze API logs?")
        pyq3 = gr.Button("Common issues in API logs?")
        pyq4 = gr.Button("How to detect anomalies in API logs?")

    # Event bindings
    submit_btn.click(process_query, inputs=query_box, outputs=chatbot_output)

    # PYQ shortcuts
    pyq1.click(process_query, inputs=gr.Textbox(value="What is an API Log?", visible=False), outputs=chatbot_output)
    pyq2.click(process_query, inputs=gr.Textbox(value="How to analyze API logs?", visible=False), outputs=chatbot_output)
    pyq3.click(process_query, inputs=gr.Textbox(value="Common issues in API logs?", visible=False), outputs=chatbot_output)
    pyq4.click(process_query, inputs=gr.Textbox(value="How to detect anomalies in API logs?", visible=False), outputs=chatbot_output)

# ------------------------------
# Launch the app
# ------------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
