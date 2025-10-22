from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

from config import Config
from pdf_processor import PDFProcessor
from rag_engine import RAGEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

vectorstore = None
rag_engine = None
processor = None

def initialize_system():
    """Initialize RAG system"""
    global vectorstore, rag_engine, processor
    
    logger.info("🚀 Initializing BrainTrainr RAG API...")
    
    try:
        processor = PDFProcessor()
        vectorstore = processor.process()
        
        rag_engine = RAGEngine(vectorstore)
        
        logger.info("✅ System initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization error: {str(e)}")
        return False

initialize_system()

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        "name": "BrainTrainr RAG API",
        "version": "0.1.0",
        "status": "online" if rag_engine else "initializing",
        "endpoints": {
            "POST /chat": "Ask a question",
            "GET /health": "Health check",
            "GET /stats": "System statistics"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        "status": "healthy" if rag_engine else "unhealthy",
        "rag_ready": rag_engine is not None,
        "vectorstore_ready": vectorstore is not None,
        "model": Config.MODEL_NAME
    })

@app.route('/stats', methods=['GET'])
def stats():
    """System statistics"""
    if not processor:
        return jsonify({"error": "System not initialized"}), 503
    
    stats_data = processor.get_stats()
    stats_data.update({
        "model": Config.MODEL_NAME,
        "temperature": Config.TEMPERATURE,
        "top_k": Config.TOP_K
    })
    
    return jsonify(stats_data)

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    if not rag_engine:
        return jsonify({"error": "RAG system not initialized"}), 503
    
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({"error": "'question' field required"}), 400
    
    question = data['question'].strip()
    
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400
    
    include_sources = data.get('include_sources', False)
    
    try:
        response = rag_engine.query(question, include_sources)
        
        return jsonify({
            "success": True,
            **response
        })
        
    except Exception as e:
        logger.error(f"Error in /chat: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found", "status": 404}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error", "status": 500}), 500

if __name__ == '__main__':
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )