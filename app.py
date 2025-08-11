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
import uuid

app = Flask(__name__)

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
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=128,  # shorter output
        do_sample=False
    )
    return HuggingFacePipeline(pipeline=pipe)

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

@app.route("/load_blog", methods=["POST"])
def load_blog():
    data = request.json
    session_id = data.get("session_id") or str(uuid.uuid4())
    blog_content = data.get("blog_content", "").strip()

    if not blog_content:
        return jsonify({"error": "blog_content is required"}), 400

    try:
        docs = [Document(page_content=blog_content)]
        emb = get_embeddings()

        # Store vectorstore on disk (temporary)
        persist_dir = f"/tmp/{session_id}"
        os.makedirs(persist_dir, exist_ok=True)
        Chroma.from_documents(docs, emb, persist_directory=persist_dir)

        return jsonify({"message": "Blog content loaded.", "session_id": session_id})

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

    persist_dir = f"/tmp/{session_id}"
    if not os.path.exists(persist_dir):
        return jsonify({"error": "Session not found. Please load blog first."}), 400

    try:
        emb = get_embeddings()
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=emb)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, max_token_limit=300)
        llm = get_hf_llm()

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT}
        )

        result = qa_chain({"question": question})

        return jsonify({"answer": result["answer"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
