"""
Configuration Module
====================
Central configuration management for the RAG benchmark system.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ModelConfig(BaseModel):
    """LLM model configuration"""
    generator_model: str = Field(default="gpt-4-turbo-preview")
    answerer_model: str = Field(default="gpt-4-turbo-preview")
    critic_model: str = Field(default="gpt-4-turbo-preview")
    embedding_model: str = Field(default="text-embedding-3-small")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2000)


class VectorStoreConfig(BaseModel):
    """Vector store configuration"""
    persist_directory: str = Field(default="./chroma_db")
    collection_name: str = Field(default="textbook_chunks")
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=50)


class PDFConfig(BaseModel):
    """PDF processing configuration"""
    pdf_path: str = Field(default="data/textbook.pdf")
    extract_images: bool = Field(default=False)
    extract_tables: bool = Field(default=False)


class GenerationConfig(BaseModel):
    """Generation parameters"""
    target_questions: int = Field(default=100)
    max_reflexion_loops: int = Field(default=3)
    diversity_threshold: float = Field(default=0.85)
    min_words_per_question: int = Field(default=10)
    max_words_per_question: int = Field(default=50)


class EvaluationConfig(BaseModel):
    """RAGAS evaluation thresholds"""
    min_faithfulness: float = Field(default=0.85)
    min_answer_relevance: float = Field(default=0.80)
    min_context_precision: float = Field(default=0.70)
    min_context_recall: float = Field(default=0.75)


class OutputConfig(BaseModel):
    """Output configuration"""
    output_dir: str = Field(default="./output")
    dataset_file: str = Field(default="golden_dataset.jsonl")
    log_level: str = Field(default="INFO")
    save_intermediate: bool = Field(default=True)


class Config(BaseModel):
    """Main configuration object"""
    models: ModelConfig = Field(default_factory=ModelConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    pdf: PDFConfig = Field(default_factory=PDFConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        return cls(
            models=ModelConfig(
                generator_model=os.getenv("GENERATOR_MODEL", "gpt-4-turbo-preview"),
                answerer_model=os.getenv("ANSWERER_MODEL", "gpt-4-turbo-preview"),
                critic_model=os.getenv("CRITIC_MODEL", "gpt-4-turbo-preview"),
                embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            ),
            vector_store=VectorStoreConfig(
                persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
                collection_name=os.getenv("COLLECTION_NAME", "textbook_chunks"),
                chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50"))
            ),
            pdf=PDFConfig(
                pdf_path=os.getenv("PDF_PATH", "data/textbook.pdf")
            ),
            generation=GenerationConfig(
                target_questions=int(os.getenv("TARGET_QUESTIONS", "100")),
                max_reflexion_loops=int(os.getenv("MAX_REFLEXION_LOOPS", "3")),
                diversity_threshold=float(os.getenv("DIVERSITY_THRESHOLD", "0.85"))
            ),
            evaluation=EvaluationConfig(
                min_faithfulness=float(os.getenv("MIN_FAITHFULNESS", "0.85")),
                min_answer_relevance=float(os.getenv("MIN_ANSWER_RELEVANCE", "0.80")),
                min_context_precision=float(os.getenv("MIN_CONTEXT_PRECISION", "0.70"))
            ),
            output=OutputConfig(
                output_dir=os.getenv("OUTPUT_DIR", "./output"),
                dataset_file=os.getenv("DATASET_FILE", "golden_dataset.jsonl"),
                log_level=os.getenv("LOG_LEVEL", "INFO")
            ),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    
    def setup_directories(self):
        """Create necessary directories"""
        Path(self.vector_store.persist_directory).mkdir(parents=True, exist_ok=True)
        Path(self.output.output_dir).mkdir(parents=True, exist_ok=True)
        Path("data").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
    
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY not set")
        
        if not Path(self.pdf.pdf_path).exists():
            errors.append(f"PDF file not found: {self.pdf.pdf_path}")
        
        if self.generation.diversity_threshold < 0 or self.generation.diversity_threshold > 1:
            errors.append("diversity_threshold must be between 0 and 1")
        
        if errors:
            print("❌ Configuration errors:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        return True


# Global config instance
config = Config.from_env()


# Example usage
if __name__ == "__main__":
    config = Config.from_env()
    config.setup_directories()
    
    if config.validate():
        print("✅ Configuration valid")
        print(f"\n📊 Settings:")
        print(f"   Generator Model: {config.models.generator_model}")
        print(f"   Target Questions: {config.generation.target_questions}")
        print(f"   PDF Path: {config.pdf.pdf_path}")
        print(f"   Output: {config.output.output_dir}/{config.output.dataset_file}")
    else:
        print("❌ Configuration invalid")
