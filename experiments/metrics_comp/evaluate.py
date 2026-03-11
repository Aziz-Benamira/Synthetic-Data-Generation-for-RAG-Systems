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



from RAGAgent import RAGAgent


## Paths, to change if needed ######################################################################################
root = "D:/Users/malopif/Documents/TRAVAIL/ENSTA/3A/Synthetic-Data-Generation-for-RAG-Systems/"
parent = (Path(root) / "experiments" / "metrics_comp").resolve()
####################################################################################################################


sys.path.append(root)
from src.llm.manager import LLMManager
from Projet_Pipeline import run_chunking

chunk_path = parent / "vectorDB"
gold_dataset_path = parent / "golden_QA.jsonl"
output_dir = parent / "output"


#1 - with context
SmartAgent = RAGAgent(data_path=parent / 'data')

SmartAgent.answer_file(gold_dataset_path,
                  output_dir / (SmartAgent.answererName + "-context"),
                  )
#2 - without context
BlindAgent = RAGAgent(data_path= None)
BlindAgent.answer_file(gold_dataset_path,
                       output_dir / (SmartAgent.answererName + "-no-context"))
