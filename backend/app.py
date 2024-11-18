"""
RAG System Backend
- Handles multimodal inputs (text, images, audio, hashed files).
- Integrates multiple AI models for retrieval, transcription, and summarization.
- Includes LangChain for conversational query workflows.
"""

import os
import uuid
import logging
import hashlib
import base64
import torch
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer
import whisper
from transformers import LlamaTokenizer, LlamaForCausalLM, CLIPModel, CLIPProcessor
import faiss
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# Load environment variables
load_dotenv()

# Flask app setup
app = Flask(__name__)

# Secure file storage configuration
UPLOAD_FOLDER = "/secure_storage/data/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"pdf", "docx", "xlsx", "csv", "jpg", "png", "mp3", "wav", "mp4", "txt", "bin", "hash"}

# Logging setup
logging.basicConfig(filename="error.log", level=logging.ERROR, format="%(asctime)s [%(levelname)s] %(message)s")

# Apple Silicon GPU setup (Metal Performance Shaders backend)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Initialize AI models
sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)  # Semantic search embeddings
whisper_model = whisper.load_model("backend/models/whisper").to(device)  # Audio transcription
lama_tokenizer = LlamaTokenizer.from_pretrained("backend/models/llama/meta-llama_Llama-3.1-8B")
llama_model = LlamaForCausalLM.from_pretrained("backend/models/llama/meta-llama_Llama-3.1-8B").to(device)
clip_model = CLIPModel.from_pretrained("backend/models/clip/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("backend/models/clip/clip-vit-base-patch32")

# FAISS indexes
faiss_index = faiss.IndexFlatL2(384)  # SentenceTransformer embeddings
image_faiss_index = faiss.IndexFlatL2(512)  # CLIP image embeddings
data_store = {}

# LangChain setup
conversation_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Google Gemini Tool
def google_gemini_tool(query):
    """
    Fetch results from the Google Gemini API.
    Args:
        query (str): User query.
    Returns:
        str: Web-based response.
    """
    api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"prompt": query, "model": "gemini-1.5-flash-latest"}
    response = requests.post("https://api.google.com/gemini/v1/chat:complete", headers=headers, json=payload)
    return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")

tools = [
    Tool(
        name="FAISS Search",
        func=lambda q: " ".join([data_store[i]["text"] for i in faiss_index.search(sentence_model.encode(q).reshape(1, -1), k=5)[1][0]]),
        description="Search documents using semantic retrieval."
    ),
    Tool(
        name="Google Search",
        func=google_gemini_tool,
        description="Fetches results from Google Gemini API."
    )
]

agent = initialize_agent(tools, llm=llama_model, memory=conversation_memory, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION)

def allowed_file(filename):
    """
    Check if the file extension is allowed.
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def add_to_faiss(doc_id, content):
    """
    Add text content to FAISS for semantic search.
    """
    embedding = sentence_model.encode(content).reshape(1, -1)
    faiss_index.add(embedding)
    data_store[doc_id] = {"text": content, "embedding": embedding}

def process_file(file_path, file_type, doc_id):
    """
    Process file based on type and add to FAISS or image index.
    """
    content = ""
    if file_type in {"jpg", "png"}:
        image = Image.open(file_path).convert("RGB")
        image_embedding = clip_model.get_image_features(**clip_processor(images=image, return_tensors="pt").to(device))
        image_faiss_index.add(image_embedding.detach().cpu().numpy())
        data_store[doc_id] = {"image_path": file_path, "image_embedding": image_embedding}
        content = pytesseract.image_to_string(image)
    elif file_type in {"mp3", "wav"}:
        content = whisper_model.transcribe(file_path)["text"]
    elif file_type in {"pdf", "docx", "txt"}:
        with open(file_path, "r") as f:
            content = f.read()
    add_to_faiss(doc_id, content)

@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Upload and process files.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type."}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    doc_id = uuid.uuid4().hex

    process_file(file_path, file.filename.rsplit(".", 1)[1].lower(), doc_id)
    return jsonify({"message": "File processed successfully.", "doc_id": doc_id}), 200

@app.route("/query", methods=["POST"])
def query():
    """
    Query the system using LangChain.
    """
    query_text = request.json.get("query", "").strip()
    if not query_text:
        return jsonify({"error": "Query cannot be empty."}), 400

    response = agent.run(query_text)
    return jsonify({"response": response}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)
