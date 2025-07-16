from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.schema import Document

# Path setup
FAISS_INDEX = "rag/faiss_index"

# Load transcript
loader = TextLoader("data/external/david_chalmers_ted.txt")
raw_docs = loader.load()

# Add metadata with title
docs = [Document(page_content=doc.page_content, metadata={"title": "David Chalmers TED Talk on Consciousness"}) for doc in raw_docs]



# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(docs)

# Load existing FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # ✅ 384-dim

db = FAISS.load_local(FAISS_INDEX, embedding_model, allow_dangerous_deserialization=True)

# Add Chalmers transcript to FAISS
db.add_documents(docs)
db.save_local(FAISS_INDEX)





# Add metadata with title
docs = [Document(page_content=doc.page_content, metadata={"title": "David Chalmers TED Talk on Consciousness"}) for doc in raw_docs]


print("✅ Chalmers TED Talk embedded into FAISS index.")
