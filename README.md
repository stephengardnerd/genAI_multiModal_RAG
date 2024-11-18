# # **RAG System: Comprehensive Overview**

The **Retrieval-Augmented Generation (RAG)** system is an AI-driven application that integrates multiple AI models to handle multimodal inputs (text, images, audio, video, and hashed files). It employs a secure and scalable architecture using Docker, CI/CD pipelines, and OWASP-compliant best practices.

---

## **Table of Contents**
1. [Backend Overview (`app.py`)](#backend-overview-apppy)
2. [Frontend Overview](#frontend-overview)
3. [Docker Compose](#docker-compose)
4. [CI/CD Pipelines](#cicd-pipelines)
5. [Multimodal Capabilities](#multimodal-capabilities)
6. [OWASP Compliance](#owasp-compliance)
7. [Setup Instructions](#setup-instructions)
8. [Future Enhancements](#future-enhancements)

---

## **Backend Overview (`app.py`)**

The backend is built with **Flask** and serves as the central API layer for the RAG system.

### **Core Features**
1. **File Uploads**:
   - **Route**: `/upload`
   - Handles uploads of various file types (`.pdf`, `.jpg`, `.mp3`, `.mp4`, etc.).
   - Securely stores files in `secure_storage/data` and processes them for indexing.

2. **Query Processing**:
   - **Route**: `/query`
   - Accepts user queries, retrieves relevant information using **FAISS** hybrid search (semantic and keyword-based), and synthesizes answers using **LLAMA**.

3. **Integrated Models**:
   - **Text**: Uses **SentenceTransformers** for semantic embeddings.
   - **Images**: Processes images with **CLIP** for embeddings and **Tesseract OCR** for text extraction.
   - **Audio**: Transcribes audio using **Whisper**.
   - **Video**: Extracts audio from videos using **MoviePy** and processes it with **Whisper**.
   - **Hashed Files**: Validates and securely stores hashed files.

4. **Google Gemini Integration**:
   - Queries not answerable by the system are sent to the **Google Gemini API** for web search.

### **Backend Folder Structure**
```plaintext
backend/
├── app.py                # Main Flask application
├── requirements.txt      # Backend dependencies
├── secure_storage/       # Secure directory for uploaded files
├── models/               # Pretrained models and FAISS indexes
├── Dockerfile            # Docker configuration for the backend

## **Frontend Overview**

The frontend of the RAG system is built using **React**. It provides an intuitive user interface for interacting with the system, including features like file uploads, query input, and result display. It integrates with the backend API to send user inputs and display the retrieved and processed data.

---

### **Folder Structure**

The folder structure of the frontend ensures modularity and maintainability.

```plaintext
frontend/
├── public/                   # Public static files (e.g., index.html)
├── src/                      # Source files
│   ├── App.js                # Main React component
│   ├── App.css               # Global styles
│   ├── index.js              # React entry point
│   ├── components/           # Reusable components
│   │   ├── FileUpload.js     # Handles file uploads
│   │   ├── QueryInput.js     # Allows user query input
│   │   ├── ResultsDisplay.js # Displays results to the user
├── Dockerfile                # Docker configuration for the frontend
├── package.json              # Dependency definitions for the frontend
├── package-lock.json         # Locked versions of dependencies

## **Docker Compose Configuration**

The RAG system uses a `docker-compose.yml` file to orchestrate its services, including the backend (Flask), frontend (React), and FAISS indexing service. This setup simplifies deployment and ensures seamless communication between the services.

---

### **docker-compose.yml**

```yaml
version: "3.8"

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: rag-backend
    ports:
      - "5000:5000" # Exposes the Flask API on port 5000
    environment:
      FLASK_ENV: development
      GOOGLE_GEMINI_API_KEY: ${GOOGLE_GEMINI_API_KEY} # Secured via .env file
    volumes:
      - ./backend/secure_storage:/secure_storage # Mount secure storage directory
      - ./backend/models:/models # Mount models directory for persistence
    depends_on:
      - faiss_index # Ensures FAISS service starts before the backend

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: rag-frontend
    ports:
      - "3000:80" # Exposes the React app on port 3000
    depends_on:
      - backend # Ensures backend starts before the frontend

  faiss_index:
    image: faiss-cpu:latest
    container_name: faiss-container
    volumes:
      - ./backend/models:/models # Mounts shared model directory for FAISS

## **OWASP Compliance in the RAG System**

The **RAG System** adheres to the **OWASP (Open Web Application Security Project)** standards to ensure a secure, robust, and compliant architecture. Below are the key functionalities implemented to address the OWASP Top 10 vulnerabilities:

---

### **1. Injection Prevention**

#### **Description**
Injection vulnerabilities occur when untrusted data is sent to a system and executed as part of a command or query.

#### **Implementation in RAG**:
- **Sanitizing File Uploads**:
  - Files are checked against a predefined set of allowed extensions using the `ALLOWED_EXTENSIONS` configuration.
  - Filenames are sanitized using `secure_filename` from Flask.

- **Example Code**:
  ```python
  def allowed_file(filename):
      return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

  if not file or not allowed_file(file.filename):
      return jsonify({"error": "Unsupported file type."}), 400

# **Retrieval-Augmented Generation (RAG) System**

The **RAG System** is an AI-powered, multimodal application designed to handle various types of data such as text, images, audio, video, and hashed files. It enables users to:
- Upload files for analysis and storage.
- Query the system using advanced AI models for insights.
- Retrieve external data when necessary using the Google Gemini API for web-based searches.

This guide details the setup and usage of the system on macOS, specifically optimized for Apple Silicon, and includes instructions for Docker deployment, CI/CD integration, and secure configurations.

---

## **Features**

1. **Multimodal Data Support**:
   - Text, PDF, images, audio, video, and hashed file inputs.
   - Processing with AI models: LLAMA, CLIP, Whisper, and FAISS.

2. **RAG System Functionality**:
   - Combines indexed retrieval with generative AI for detailed responses.
   - Queries supported through both React UI and API endpoints.

3. **Security and Compliance**:
   - Adheres to OWASP standards.
   - Secure `.env` and Docker configurations.

4. **Scalable Deployment**:
   - Supports Docker-based containerization and CI/CD pipelines.

---

## **System Requirements**

### **Hardware**
- macOS with Apple Silicon (M1/M2/M3 recommended).
- Minimum 16GB Unified Memory for efficient model processing.

### **Software**
- **Backend**: Python 3.8+
- **Frontend**: Node.js 16+
- **Dependencies**: Homebrew (to install system-level tools like Tesseract and FFmpeg).

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/stephengardnerd/genAI_multiModal_RAG.git
cd rag-system

