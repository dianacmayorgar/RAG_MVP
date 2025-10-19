import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Paths
    PDF_PATH = os.getenv('PDF_PATH', 'data/')
    CHROMA_DIR = os.getenv('CHROMA_DIR', './chroma_db')
    
    # Model settings
    MODEL_NAME = os.getenv('MODEL_NAME', 'llama3.2:3b')
    TEMPERATURE = float(os.getenv('TEMPERATURE', '0.3'))
    
    # RAG settings
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200'))
    TOP_K = int(os.getenv('TOP_K', '3'))
    
    # Flask
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', '5000'))