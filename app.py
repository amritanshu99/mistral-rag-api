# ===========================================
# Minimal RAG API for Render Free Tier (512 MB) - Mistral Version
# ===========================================

# ---- Core Imports ----
from flask import Flask, request, jsonify
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

import tempfile
import shutil
import os

# ===========================================
# Flask App Setup
# ===========================================
app = Flask(__name__)

# ===========================================
# Custom QA Prompt (RAG Instruction)
# ===========================================
template = """
Use the provided context to answer the question concisely.
If the context is empty or irrelevant, say "I don't know based on the provided information."

Context:
{context}

Question:
{question}

Answer:
"""

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# ===========================================
# Route: Ask a Question from Blog Content
# ===========================================
@app.route("/ask", methods=["POST"])
def ask():
    """
    Expects JSON:
    {
        "blog_content": "...",   # The content of the blog (string)
        "question": "..."        # User's question (string)
    }
    """
    data = request.json
    blog_content = data.get("blog_content", "")
    question = data.get("question", "")

    if not blog_content.strip() or not question.strip():
        return jsonify({"error": "Both blog_content and question are required"}), 400

    # ---- TEMPORARY CHROMA STORAGE (memory optimization) ----
    temp_dir = tempfile.mkdtemp()

    try:
        # ---- Create Document ----
        docs = [Document(page_content=blog_content)]

        # ---- Initialize Embeddings ----
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")

        # ---- Create Chroma Vector Store ----
        vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=temp_dir)

        # ---- Create Retriever ----
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        # ---- Initialize LLM (Mistral from Ollama) ----
        llm = Ollama(model="mistral", temperature=0)

        # ---- Create QA Chain ----
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )

        # ---- Run Query ----
        answer = qa_chain.run(question)

        return jsonify({"answer": answer})

    finally:
        # ---- Clean up to save RAM ----
        shutil.rmtree(temp_dir, ignore_errors=True)

# ===========================================
# Entry Point
# ===========================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
