# rag/query_bot.py

from dotenv import load_dotenv
load_dotenv()

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama  # ✅ LOCAL LLM

# -----------------------
# Config & Paths
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX = os.path.join(BASE_DIR, "faiss_index")
print("FAISS_INDEX path:", FAISS_INDEX)

# -----------------------
# Use local HuggingFace embeddings + Ollama LLM
# -----------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Use local Ollama with Mistral
llm = Ollama(model="mistral")

# -----------------------
# Load FAISS index
# -----------------------
db = FAISS.load_local(FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)

# -----------------------
# Build RAG chain
# -----------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(),
    return_source_documents=True
)

# -----------------------
# Interactive chat loop
# -----------------------
print("\n🧠 Ask your consciousness questions! (Type 'exit' to quit)\n")

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break

    result = qa_chain.invoke(query)  # ✅ modern usage!
    answer = result["result"]
    sources = result["source_documents"]

    print("\n🤖 Answer:\n", answer)
    print("\n📚 Sources:")
    for doc in sources:
        print(f" - {doc.metadata.get('title', 'Unknown Title')}")
    print("\n" + "-" * 50)
