"""
Evaluates a RAG model on a Q/A/C dataset. Dataset should be a jsonl containing fields 'question',
 'answer' and a chunk indexing coherent with a vectorDB.

"""
import os
import sys
from pathlib import Path
import json
import glob
import requests
from langchain_community.embeddings import OllamaEmbeddings
from time import time
import logging

# Disable verbose HTTP logging
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)

from RAGAgent import RAGAgent


## Paths, to change if needed ######################################################################################
root = Path("").resolve()
parent = root / "experiments" / "metrics_comp"
####################################################################################################################


sys.path.append(root.as_posix())
from src.llm.manager import LLMManager
from src.llm.base import LLMConfig
from Projet_Pipeline import run_chunking

chunk_path = parent / "vectorDB"
gold_dataset_path = parent / "golden_QA.jsonl"
output_dir = parent / "output"

models = [f"qwen3.5:{i}b" for i in [0.8, 9]]
for model in models:
    manager = LLMManager.from_ollama(model = model, base_url= "http://ensta-h10001.r2.enst.fr:8080/v1")
    config = LLMConfig(reasoning = "none", max_tokens=500)

    embedding_model = OllamaEmbeddings(model="nomic-embed-text", base_url="http://ensta-h10001.r2.enst.fr:8080")


    #1 - with context
    t0=time()
    SmartAgent = RAGAgent(data_path=parent / 'data', model_name=model, manager= manager, embedding_model=embedding_model, config=config)

    SmartAgent.answer_file(gold_dataset_path,
                    output_dir / (model + "-context.jsonl"),
                    )

    #2 - without context
    BlindAgent = RAGAgent(data_path= None, model_name=model,
                        manager=manager,
                        embedding_model=embedding_model,
                        config=config)
    BlindAgent.answer_file(gold_dataset_path,
                        output_dir / (model + "-no-context.jsonl"))
    print(f"time elapse : {time()- t0}s")