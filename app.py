from flask import Flask, request, jsonify
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline
import os

import hashlib
from transformers import pipeline

app = Flask(__name__)

# Global stores
vectorstore_cache = {}  # session_id -> vectorstore
memory_store = {}       # session_id -> ConversationBufferMemory
llm = None
embeddings = None

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Use the provided context to answer the question concisely.
If the context is empty or irrelevant, say "I don't know based on the provided information."

Context:
{context}

Question:
{question}

Answer:
"""
)

def get_hf_llm():
    global llm
    if llm is None:
        pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            max_length=256,
            do_sample=False
        )
        llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def get_embeddings():
    global embeddings
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

@app.route("/load_blog", methods=["POST"])
def load_blog():
    data = request.json
    session_id = data.get("session_id")
    blog_content = data.get("blog_content", "").strip()

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    if not blog_content:
        return jsonify({"error": "blog_content is required"}), 400

    try:
        # Create document and embeddings once per session/blog load
        docs = [Document(page_content=blog_content)]
        emb = get_embeddings()

        # In-memory vectorstore for this session
        vectorstore_cache[session_id] = Chroma.from_documents(docs, emb)
        memory_store[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=1000
        )

        return jsonify({"message": f"Blog content loaded and embeddings created for session '{session_id}'."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    session_id = data.get("session_id")
    question = data.get("question", "").strip()

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    if not question:
        return jsonify({"error": "question is required"}), 400

    if session_id not in vectorstore_cache or session_id not in memory_store:
        return jsonify({"error": "Session not initialized. Please load blog content first via /load_blog."}), 400

    try:
        vectorstore = vectorstore_cache[session_id]
        memory = memory_store[session_id]

        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        hf_llm = get_hf_llm()

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=hf_llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT}
        )

        result = qa_chain({"question": question})

        return jsonify({
            "answer": result["answer"],
            "chat_history": [m.dict() for m in memory.chat_memory.messages]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
