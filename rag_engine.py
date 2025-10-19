from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import logging
import time

from config import Config

logger = logging.getLogger(__name__)

class RAGEngine:
    """Motor de RAG con Llama local"""
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = self._setup_llm()
        self.qa_chain = self._setup_chain()
        
    def _setup_llm(self):
        """Configurar Llama"""
        logger.info(f"Configurando LLM: {Config.MODEL_NAME}")
        
        return Ollama(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
        )
    
    def _setup_chain(self):
        """Crear cadena RAG con soporte bilingüe automático"""
        
        # Template bilingüe - detecta idioma automáticamente
        template = """You are an expert assistant from BrainTrainr that helps answer questions based on official documentation.

Important rules:
- Use ONLY the information from the provided context
- If you don't have the information, say: "I don't have that information in my database" (or in Spanish: "No tengo esa información en mi base de datos")
- Be concise and direct
- **CRITICAL: Respond in the SAME LANGUAGE as the user's question**
  - If the question is in English → Answer in English
  - If the question is in Spanish → Answer in Spanish
- When mentioning specific information (dates, requirements, contacts), cite the source

Context:
{context}

Question: {question}

Helpful answer (in the same language as the question):"""
        
        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": Config.TOP_K}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def query(self, question: str, include_sources=False):
        """
        Procesar una pregunta
        
        Args:
            question: La pregunta del usuario (en cualquier idioma)
            include_sources: Si se incluyen los chunks fuente
            
        Returns:
            dict con answer, sources (opcional), y metadata
        """
        logger.info(f"Query: {question}")
        
        start_time = time.time()
        
        try:
            result = self.qa_chain({"query": question})
            
            elapsed_time = time.time() - start_time
            
            response = {
                "answer": result["result"],
                "metadata": {
                    "response_time_seconds": round(elapsed_time, 2),
                    "chunks_used": len(result["source_documents"]),
                    "model": Config.MODEL_NAME
                }
            }
            
            if include_sources:
                response["sources"] = [
                    {
                        "content": doc.page_content[:500],
                        "page": doc.metadata.get("page", "N/A"),
                        "source_file": doc.metadata.get("source_file", "Unknown")
                    }
                    for doc in result["source_documents"]
                ]
            
            logger.info(f"✅ Respuesta generada en {elapsed_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error en query: {str(e)}")
            raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from pdf_processor import PDFProcessor
    
    processor = PDFProcessor()
    vectorstore = processor.process()
    
    engine = RAGEngine(vectorstore)
    
    # Test bilingüe - mezcla inglés y español
    test_questions = [
        "What are the requirements to immigrate to Canada?",  # Inglés
        "¿Qué es Express Entry?",  # Español
        "How long does the permanent residence process take?",  # Inglés
        "¿Cuáles son los tipos de visa disponibles?",  # Español
    ]
    
    print("\n" + "="*60)
    print("🌐 TEST BILINGÜE DEL RAG ENGINE")
    print("="*60)
    
    for q in test_questions:
        print(f"\n❓ Pregunta: {q}")
        response = engine.query(q, include_sources=False)
        print(f"💬 Respuesta: {response['answer']}")
        print(f"⏱️  Tiempo: {response['metadata']['response_time_seconds']}s")
        print(f"📊 Chunks: {response['metadata']['chunks_used']}")