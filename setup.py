from setuptools import setup, find_packages

setup(
    name="synthetic-rag-evaluation",
    version="0.1.0",
    description="Synthetic data generation and evaluation for RAG systems",
    author="Research Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "openai>=1.0.0",
        "langchain>=0.1.0",
        "langgraph>=0.2.35",
        "chromadb>=0.5.20",
        "ragas>=0.2.7",
        "pydantic>=2.5.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
    },
)
