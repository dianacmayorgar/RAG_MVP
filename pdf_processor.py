from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import glob
import logging

from config import Config

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Procesa PDFs y crea vector store - soporta archivo individual o carpeta completa"""
    
    def __init__(self, pdf_path=None):
        self.pdf_path = pdf_path or Config.PDF_PATH
        self.embeddings = self._load_embeddings()
        self.vectorstore = None
        
    def _load_embeddings(self):
        """Cargar modelo de embeddings"""
        logger.info("Cargando modelo de embeddings...")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}
        )
    
    def _get_pdf_files(self):
        """Obtener lista de PDFs a procesar"""
        # Si es una carpeta, obtener todos los PDFs
        if os.path.isdir(self.pdf_path):
            pdf_files = glob.glob(os.path.join(self.pdf_path, "*.pdf"))
            if not pdf_files:
                raise FileNotFoundError(f"No se encontraron PDFs en {self.pdf_path}")
            logger.info(f"📁 Carpeta detectada: {len(pdf_files)} PDFs encontrados")
            return sorted(pdf_files)
        
        # Si es un archivo individual
        elif os.path.isfile(self.pdf_path):
            logger.info(f"📄 Archivo individual: {self.pdf_path}")
            return [self.pdf_path]
        
        else:
            raise FileNotFoundError(f"No existe: {self.pdf_path}")
    
    def load_and_split(self):
        """Cargar PDF(s) y dividir en chunks"""
        pdf_files = self._get_pdf_files()
        
        all_chunks = []
        successful_files = 0
        failed_files = 0
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"📖 Procesando: {os.path.basename(pdf_file)}")
                
                loader = PyPDFLoader(pdf_file)
                documents = loader.load()
                
                # Agregar metadata del archivo fuente
                for doc in documents:
                    doc.metadata['source_file'] = os.path.basename(pdf_file)
                    doc.metadata['source_path'] = pdf_file
                
                logger.info(f"   ✓ {len(documents)} páginas cargadas")
                
                chunks = text_splitter.split_documents(documents)
                all_chunks.extend(chunks)
                
                logger.info(f"   ✓ {len(chunks)} chunks creados")
                successful_files += 1
                
            except Exception as e:
                logger.error(f"   ✗ Error procesando {os.path.basename(pdf_file)}: {e}")
                failed_files += 1
                continue
        
        # Resumen
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 RESUMEN:")
        logger.info(f"   ✓ Archivos exitosos: {successful_files}")
        logger.info(f"   ✗ Archivos fallidos: {failed_files}")
        logger.info(f"   📝 Total chunks: {len(all_chunks)}")
        logger.info(f"{'='*60}\n")
        
        if len(all_chunks) == 0:
            raise ValueError("No se pudo procesar ningún PDF")
        
        return all_chunks
    
    def create_vectorstore(self, chunks, persist_dir=None):
        """Crear y persistir vector store"""
        persist_dir = persist_dir or Config.CHROMA_DIR
        
        logger.info(f"💾 Creando vector store con {len(chunks)} chunks...")
        logger.info("   (esto puede tardar unos minutos)")
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_dir
        )
        
        logger.info(f"✅ Vector store guardado en {persist_dir}")
        return self.vectorstore
    
    def load_vectorstore(self, persist_dir=None):
        """Cargar vector store existente"""
        persist_dir = persist_dir or Config.CHROMA_DIR
        
        if not os.path.exists(persist_dir):
            return None
            
        logger.info(f"📂 Cargando vector store existente de {persist_dir}...")
        self.vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )
        
        return self.vectorstore
    
    def process(self, force_recreate=False):
        """Pipeline completo: cargar o crear vector store"""
        persist_dir = Config.CHROMA_DIR
        
        if not force_recreate:
            vectorstore = self.load_vectorstore(persist_dir)
            if vectorstore:
                logger.info("✅ Vector store cargado exitosamente")
                stats = self.get_stats()
                logger.info(f"   📊 {stats['total_chunks']} chunks en base de datos")
                return vectorstore
        
        logger.info("🔄 Creando nuevo vector store...")
        chunks = self.load_and_split()
        vectorstore = self.create_vectorstore(chunks, persist_dir)
        logger.info("✅ Vector store creado exitosamente")
        
        return vectorstore
    
    def get_stats(self):
        """Obtener estadísticas del vector store"""
        if not self.vectorstore:
            return None
            
        collection = self.vectorstore._collection
        return {
            "total_chunks": collection.count(),
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "chunk_size": Config.CHUNK_SIZE,
            "chunk_overlap": Config.CHUNK_OVERLAP
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    
    processor = PDFProcessor()
    vectorstore = processor.process()
    
    results = vectorstore.similarity_search("procesos para inmigrar a Canadá", k=3)
    
    print("\n" + "="*60)
    print("🔍 TEST DE BÚSQUEDA")
    print("="*60)
    for i, doc in enumerate(results, 1):
        print(f"\n--- Resultado {i} ---")
        print(f"Archivo: {doc.metadata.get('source_file', 'N/A')}")
        print(f"Página: {doc.metadata.get('page', 'N/A')}")
        print(f"Contenido: {doc.page_content[:200]}...")
    
    print("\n" + "="*60)
    print("📊 ESTADÍSTICAS")
    print("="*60)
    stats = processor.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")