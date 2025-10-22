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
    """Processes PDFs and creates vector store - supports individual file or entire folder"""
    
    def __init__(self, pdf_path=None):
        self.pdf_path = pdf_path or Config.PDF_PATH
        self.embeddings = self._load_embeddings()
        self.vectorstore = None
        
    def _load_embeddings(self):
        """Load embeddings model"""
        logger.info("Loading embeddings model...")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}
        )
    
    def _get_pdf_files(self):
        """Get list of PDFs to process"""
        # If it's a folder, get all PDFs
        if os.path.isdir(self.pdf_path):
            pdf_files = glob.glob(os.path.join(self.pdf_path, "*.pdf"))
            if not pdf_files:
                raise FileNotFoundError(f"No PDFs found in {self.pdf_path}")
            logger.info(f"📁 Folder detected: {len(pdf_files)} PDFs found")
            return sorted(pdf_files)
        
        # If it's an individual file
        elif os.path.isfile(self.pdf_path):
            logger.info(f"📄 Individual file: {self.pdf_path}")
            return [self.pdf_path]
        
        else:
            raise FileNotFoundError(f"Does not exist: {self.pdf_path}")
    
    def load_and_split(self):
        """Load PDF(s) and split into chunks"""
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
                logger.info(f"📖 Processing: {os.path.basename(pdf_file)}")
                
                loader = PyPDFLoader(pdf_file)
                documents = loader.load()
                
                # Add source file metadata
                for doc in documents:
                    doc.metadata['source_file'] = os.path.basename(pdf_file)
                    doc.metadata['source_path'] = pdf_file
                
                logger.info(f"   ✓ {len(documents)} pages loaded")
                
                chunks = text_splitter.split_documents(documents)
                all_chunks.extend(chunks)
                
                logger.info(f"   ✓ {len(chunks)} chunks created")
                successful_files += 1
                
            except Exception as e:
                logger.error(f"   ✗ Error processing {os.path.basename(pdf_file)}: {e}")
                failed_files += 1
                continue
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 SUMMARY:")
        logger.info(f"   ✓ Successful files: {successful_files}")
        logger.info(f"   ✗ Failed files: {failed_files}")
        logger.info(f"   📝 Total chunks: {len(all_chunks)}")
        logger.info(f"{'='*60}\n")
        
        if len(all_chunks) == 0:
            raise ValueError("Could not process any PDF")
        
        return all_chunks
    
    def create_vectorstore(self, chunks, persist_dir=None):
        """Create and persist vector store"""
        persist_dir = persist_dir or Config.CHROMA_DIR
        
        logger.info(f"💾 Creating vector store with {len(chunks)} chunks...")
        logger.info("   (this may take a few minutes)")
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_dir
        )
        
        logger.info(f"✅ Vector store saved in {persist_dir}")
        return self.vectorstore
    
    def load_vectorstore(self, persist_dir=None):
        """Load existing vector store"""
        persist_dir = persist_dir or Config.CHROMA_DIR
        
        if not os.path.exists(persist_dir):
            return None
            
        logger.info(f"📂 Loading existing vector store from {persist_dir}...")
        self.vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )
        
        return self.vectorstore
    
    def process(self, force_recreate=False):
        """Complete pipeline: load or create vector store"""
        persist_dir = Config.CHROMA_DIR
        
        if not force_recreate:
            vectorstore = self.load_vectorstore(persist_dir)
            if vectorstore:
                logger.info("✅ Vector store loaded successfully")
                stats = self.get_stats()
                logger.info(f"   📊 {stats['total_chunks']} chunks in database")
                return vectorstore
        
        logger.info("🔄 Creating new vector store...")
        chunks = self.load_and_split()
        vectorstore = self.create_vectorstore(chunks, persist_dir)
        logger.info("✅ Vector store created successfully")
        
        return vectorstore
    
    def get_stats(self):
        """Get vector store statistics"""
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
    
    results = vectorstore.similarity_search("immigration processes to Canada", k=3)
    
    print("\n" + "="*60)
    print("🔍 SEARCH TEST")
    print("="*60)
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"File: {doc.metadata.get('source_file', 'N/A')}")
        print(f"Page: {doc.metadata.get('page', 'N/A')}")
        print(f"Content: {doc.page_content[:200]}...")
    
    print("\n" + "="*60)
    print("📊 STATISTICS")
    print("="*60)
    stats = processor.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")