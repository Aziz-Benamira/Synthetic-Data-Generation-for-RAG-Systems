import sys
from pathlib import Path
import logging
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import json
from tqdm import tqdm

# Add src folder to path
root = Path("../../").resolve()
src_path = root / "src"
sys.path.insert(0, str(src_path))


from chunking.semantic_chunker import SemanticChunk, SemanticChunker
from llm.manager import LLMManager
from llm.base import LLMConfig
from config import Config



# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



# Enhanced RAG Agent with critic integration
class RAGAgent:
    """
    Complete RAG pipeline with answer evaluation.
    """
    
    def __init__(self, 
                 data_path : Path , #Should contain a pdf/ and a vectorDb/ folder
                 manager : LLMManager = None,
                 chunker = None,
                 embedding_model : OllamaEmbeddings = None
                 ):
        if not manager : 
            self.llm_manager = LLMManager.from_ollama(
                model="qwen3.5:0.8b",
                base_url="http://localhost:11434/v1"
            )qwe
        else:
            self.llm_manager = manager
        
        if not chunker:
            chunker = SemanticChunker

        if not embedding_model:
            self.embedding_model = OllamaEmbeddings(model="nomic-embed-text")
        else:
            self.embedding_model = embedding_model
        self.dataF = data_path

        ## embeddings

        # 1-Chunk documents
        docs = [] # list of lang documents
        for f in os.listdir(self.dataF / 'pdf'):
            print(f"Chunking {f}...")
            chunks  = (SemanticChunker(self.dataF / 'pdf' / f))
            docs += [c.to_langchain_document() for c in chunks.chunk_document()] #concatenate to get flat list of docs

        # 2-Get embeddings from file, or compute them if not available
        try : #very lazy try because i'm too lazy to check the path doc for existence of a file
            self.vector_db = Chroma(persist_directory=(self.dataF / 'vectorDb').as_posix(),
                                embedding_function=self.embedding_model
                                )
        except FileNotFoundError : 
            self.vector_db = Chroma.from_documents(
                documents=docs,
                embedding=self.embedding_model,
                persist_directory= (self.dataF / 'vectorDb').as_posix()
            )
        print(f"Base créée avec {len(docs)} chunks.")

    def retrieve(self, query : str, k=3):
        #return the list of the k best langchain documents to answer the question
        return self.vector_db.similarity_search(query, k=k)

    def answer(self, query: str, context: str = ""):
        """
        Answer a question given a context. To test the model itself, leave the context empty.
        """
        system_prompt = f"""
Use the following context to answer the question. When unsure, always answer that you don't know.
---
CONTEXTE :
{context}
"""

        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        response = self.llm_manager.generate_from_messages(messages)
        return response.content
    
    def rag_answer(self, 
                query : str,
                k = 3):
        #  vector retrieval
        docs = self.retrieve(query, k)

        context = "\n\n".join([doc.page_content for doc in docs])

        answer = self.answer(query, context)

        return docs, answer


    def answer_file(self, inputPath: Path, outputPath: Path = None, overwrite: bool = False):
        """
        Answer all questions present in a jsonl file. Each line should contain a "question" field.
        If outputPath is given, writes the answer in the file. (With or without overwriting).

        returns the answers as a list of dicts containing the input
        """
        results = []

        if outputPath and outputPath.exists() and not overwrite:
            print("Output file already exists. Aborting.")
            return []

        with open(inputPath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                
                data = json.loads(line)
                query = data.get("question")
                
                if query:
                    docs, answer = self.rag_answer(query)
                    # Update data with RAG results
                    data["generated_answer"] = answer
                    data["context_used"] = [doc.page_content for doc in docs]
                    results.append(data)

        if outputPath:
            outputPath.parent.mkdir(parents=True, exist_ok=True)
            mode = 'w' if overwrite else 'a'
            with open(outputPath, mode, encoding='utf-8') as f:
                for entry in results:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            print(f"Results written to {outputPath}")

        return results
    
