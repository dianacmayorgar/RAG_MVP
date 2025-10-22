from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import logging
import time

from config import Config

logger = logging.getLogger(__name__)

class RAGEngine:  # ← Clase debe empezar aquí
    """Motor de RAG con Llama local"""
    
    def __init__(self, vectorstore):  # ← 4 espacios de indentación
        self.vectorstore = vectorstore
        self.llm = self._setup_llm()
        self.qa_chain = self._setup_chain()
        
    def _setup_llm(self):  # ← 4 espacios de indentación
        """Configurar Llama"""
        logger.info(f"Configurando LLM: {Config.MODEL_NAME}")
        
        return Ollama(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
        )
    
    def _setup_chain(self):  # ← 4 espacios de indentación (CRÍTICO)
        """Crear cadena RAG con control estricto de idioma"""
        
        # Template con máxima fuerza para control de idioma
        template = """LANGUAGE INSTRUCTION (MOST IMPORTANT - READ FIRST):
- You MUST answer in ENGLISH only

You are an expert assistant from BrainTrainr. Answer questions based only on the provided context.

RULES:
1. Use ONLY information from the context below
2. If you don't have the information, say: "I don't have that information in my database" (English) or "No tengo esa información en mi base de datos" (Spanish)
3. Be concise and direct
4. When mentioning specific details (dates, requirements, contacts), cite the source

CONTEXT (documents in English):
{context}

QUESTION (answer in the SAME language as this question):
{question}

ANSWER (in the exact same language as the question above):"""
        
        PROMPT = PromptTemplate(  # ← 8 espacios de indentación
            template=template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(  # ← 8 espacios de indentación
            llm=self.llm,  # ← AQUÍ está el "self" - debe tener 12 espacios
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(  # ← AQUÍ también
                search_type="similarity",
                search_kwargs={"k": Config.TOP_K}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def query(self, question: str, include_sources=False):  # ← 4 espacios
        """Procesar una pregunta"""
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


if __name__ == "__main__":  # ← Sin indentación (nivel raíz)
    logging.basicConfig(level=logging.INFO)
    
    from pdf_processor import PDFProcessor
    
    processor = PDFProcessor()
    vectorstore = processor.process()
    
    engine = RAGEngine(vectorstore)
    
    test_questions = [
        "What are the requirements to immigrate to Canada?",
        "¿Qué es Express Entry?",
        "How long does the permanent residence process take?",
        "¿Cuáles son los tipos de visa disponibles?",
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