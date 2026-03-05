"""
RAG Evaluation pipeline for the academic domain (MI201, ENSTA Paris).

Steps:
  1. Load eval_dataset.json (questions + verbatim French refs)
  2. Load 5 chapter documents from data/academic/en/doc/
  3. Chunk each document (size=500, overlap=100)
  4. Build per-chapter + global FAISS index with paraphrase-multilingual-MiniLM-L12-v2
  5. For each question: retrieve top-k chunks, compute Context Recall and Hit Rate
  6. Load Ministral-8B-Instruct-2410, generate answers
  7. Compute Answer F1 and Exact Match
  8. Print and save results

Usage (from qar_generation/):
    python evaluate_rag.py [--dataset PATH] [--model PATH] [--chunk-size N] [--overlap N] [--top-k N]
"""

import os
import sys
import json
import re
import logging
import argparse
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True,
                    format='%(asctime)s [%(levelname)s] %(message)s')

QAR_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATASET = os.path.join(QAR_DIR, 'output', 'academic', 'en', 'eval_dataset.json')
DEFAULT_DOC_DIR = os.path.join(QAR_DIR, 'data', 'academic', 'en', 'doc')
DEFAULT_RESULTS = os.path.join(QAR_DIR, 'output', 'academic', 'en', 'eval_results.json')
DEFAULT_MODEL = '/home/ensta/data/Ministral-8B-Instruct-2410'
EMBED_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'

CHAPTERS = [
    ('Ch1_Intro_ML', 'Chapter 1: Introduction to Machine Learning'),
    ('Ch2_Decision_Trees', 'Chapter 2: Decision Trees and Ensemble Methods'),
    ('Ch3_SVM', 'Chapter 3: Regularization and Support Vector Machines'),
    ('Ch4_Neural_Networks', 'Chapter 4: Introduction to Neural Networks'),
    ('Ch5_Unsupervised', 'Chapter 5: Unsupervised Learning'),
]

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question based ONLY on the provided context. "
    "If the context does not contain the answer, say 'Unable to answer. The provided context "
    "does not contain information about this topic.' Do not use prior knowledge."
)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, size: int = 500, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Semantic index
# ---------------------------------------------------------------------------

class SemanticIndex:
    def __init__(self, embed_model: SentenceTransformer):
        self.model = embed_model
        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def build(self, chunks: List[str], label: str = ''):
        logging.info(f"Building index for: {label}")
        logging.info(f"Loading embedding model: {EMBED_MODEL}")
        logging.info(f"Load pretrained SentenceTransformer: {EMBED_MODEL}")
        logging.info(f"Encoding {len(chunks)} chunks...")
        self.chunks = chunks
        self.embeddings = self.embed_model_encode(chunks)
        logging.info("Index ready.")

    def embed_model_encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        q_emb = self.embed_model_encode([query])
        sims = self.embeddings @ q_emb.T
        sims = sims.flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]

    def ref_matches_context(self, ref: str, context_chunks: List[str], threshold: float = 0.45) -> bool:
        texts = [ref] + context_chunks
        embs = self.embed_model_encode(texts)
        sims = embs[1:] @ embs[0]
        return bool(np.any(sims >= threshold))


# ---------------------------------------------------------------------------
# Local LLM client
# ---------------------------------------------------------------------------

class LocalModelClient:
    def __init__(self, model_path: str):
        print("[LocalModelClient] Checking CUDA...")
        if torch.cuda.is_available():
            free = torch.cuda.mem_get_info()[0] / 1024**3
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[LocalModelClient] GPU: {gpu_name} -- Free: {free:.1f} GB")
        else:
            print("[LocalModelClient] No GPU available, using CPU.")

        print(f"[LocalModelClient] Loading tokenizer from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        print("[LocalModelClient] Tokenizer loaded.")

        print("[LocalModelClient] Loading model (float16, CPU first)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        print("[LocalModelClient] Model loaded to CPU, moving to GPU...")
        self.model = self.model.cuda()
        used = torch.cuda.memory_allocated() / 1024**3
        print(f"[LocalModelClient] Model on GPU -- Memory used: {used:.1f} GB")

    def generate(self, tasks: List[Dict], max_new_tokens: int = 512) -> List[str]:
        responses = []
        for i, task in enumerate(tasks):
            print(f"[LocalModelClient] Generating {i+1}/{len(tasks)}...")
            messages = [
                {"role": "system", "content": task["system_prompt"]},
                {"role": "user", "content": task["user_prompt"]},
            ]
            result = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )
            # apply_chat_template may return BatchEncoding or Tensor depending on version
            if hasattr(result, 'input_ids'):
                inputs = result.input_ids.to(self.model.device)
            else:
                inputs = result.to(self.model.device)

            with torch.no_grad():
                out = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = out[0][inputs.shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            responses.append(text)
            print(f"[LocalModelClient] Generation {i+1}/{len(tasks)} done.")
        return responses


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def f1_score(pred: str, gold: str) -> float:
    pred_tokens = set(pred.lower().split())
    gold_tokens = set(gold.lower().split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> bool:
    return pred.strip().lower() == gold.strip().lower()


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--chunk-size', type=int, default=500)
    parser.add_argument('--overlap', type=int, default=100)
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()

    # Load dataset
    with open(args.dataset, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    logging.info(f"Loaded {len(questions)} questions from {args.dataset}")

    # Load documents
    doc_dir = DEFAULT_DOC_DIR
    docs = {}
    for ch_id, ch_name in CHAPTERS:
        doc_path = os.path.join(doc_dir, ch_id, '0', '0.txt')
        if os.path.exists(doc_path):
            with open(doc_path, 'r', encoding='utf-8') as f:
                docs[ch_name] = f.read()
    logging.info(f"Loaded {len(docs)} chapter documents")

    # Chunk documents
    logging.info(f"Chunking (size={args.chunk_size}, overlap={args.overlap})...")
    all_chunks = []
    chapter_chunks: Dict[str, List[str]] = {}
    for ch_name, text in docs.items():
        chunks = chunk_text(text, size=args.chunk_size, overlap=args.overlap)
        chapter_chunks[ch_name] = chunks
        all_chunks.extend(chunks)
        logging.info(f"  {ch_name}: {len(chunks)} chunks")
    logging.info(f"Total chunks: {len(all_chunks)}")

    # Build indexes
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Embedding device: {device}")
    embed_model = SentenceTransformer(EMBED_MODEL)

    per_chapter_index: Dict[str, SemanticIndex] = {}
    for ch_name, chunks in chapter_chunks.items():
        idx = SemanticIndex(embed_model)
        idx.build(chunks, label=ch_name)
        per_chapter_index[ch_name] = idx

    global_index = SemanticIndex(embed_model)
    global_index.build(all_chunks, label='global')

    # Load generation model
    logging.info(f"Loading generation model: {os.path.basename(args.model)}")
    gen_client = LocalModelClient(args.model)
    logging.info("Generation model ready.")

    # Chapter name lookup
    ch_id_to_name = {ch_id: ch_name for ch_id, ch_name in CHAPTERS}

    # Evaluate
    results = []
    recall_sum = 0.0
    recall_count = 0
    hit_count = 0
    questions_with_refs = 0
    f1_sum = 0.0
    em_count = 0

    type_stats: Dict[str, Dict] = {}

    for i, q in enumerate(questions):
        question = q['question']
        gold_answer = q.get('answer', '')
        refs = q.get('ref', [])
        qtype = q.get('question_type', 'unknown')
        chapter = q.get('chapter', '')

        # Select index: per-chapter if possible
        ch_name = ch_id_to_name.get(chapter, '')
        if ch_name and ch_name in per_chapter_index:
            idx = per_chapter_index[ch_name]
        else:
            idx = global_index

        # Retrieve
        retrieved = idx.retrieve(question, top_k=args.top_k)

        # Context Recall
        ref_recall = None
        hit = None
        if refs:
            questions_with_refs += 1
            matched = [ref for ref in refs if idx.ref_matches_context(ref, retrieved)]
            recall = len(matched) / len(refs)
            ref_recall = recall
            recall_sum += recall
            recall_count += 1
            hit = len(matched) > 0
            if hit:
                hit_count += 1

        # Generate answer
        context = '\n\n---\n\n'.join(retrieved)
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        responses = gen_client.generate([{'system_prompt': SYSTEM_PROMPT, 'user_prompt': user_prompt}])
        pred_answer = responses[0]

        # Generation metrics
        f1 = f1_score(pred_answer, gold_answer)
        em = exact_match(pred_answer, gold_answer)
        f1_sum += f1
        if em:
            em_count += 1

        # Per-type stats
        if qtype not in type_stats:
            type_stats[qtype] = {'recall_sum': 0.0, 'recall_count': 0, 'hit': 0, 'total_with_refs': 0}
        if ref_recall is not None:
            type_stats[qtype]['recall_sum'] += ref_recall
            type_stats[qtype]['recall_count'] += 1
            type_stats[qtype]['total_with_refs'] += 1
            if hit:
                type_stats[qtype]['hit'] += 1

        results.append({
            'question': question,
            'question_type': qtype,
            'chapter': chapter,
            'gold_answer': gold_answer,
            'pred_answer': pred_answer,
            'refs': refs,
            'retrieved_chunks': retrieved,
            'context_recall': ref_recall,
            'hit': hit,
            'f1': f1,
            'exact_match': em,
        })

        if (i + 1) % 10 == 0:
            logging.info(f"  Processed {i+1}/{len(questions)} questions...")

    # Print summary
    n = len(questions)
    context_recall = recall_sum / recall_count if recall_count > 0 else 0.0
    hit_rate = hit_count / questions_with_refs if questions_with_refs > 0 else 0.0
    avg_f1 = f1_sum / n
    em_rate = em_count / n

    print()
    print('=' * 60)
    print('RAG EVALUATION RESULTS')
    print('=' * 60)
    print(f"Dataset          : {args.dataset}")
    print(f"Retriever        : multilingual-MiniLM (semantic cosine sim, cross-lingual ref matching)")
    print(f"Total questions  : {n}")
    print(f"Questions w/ refs: {questions_with_refs}")
    print(f"Chunk size       : {args.chunk_size} chars  |  Overlap: {args.overlap}")
    print(f"Top-k retrieved  : {args.top_k}")
    print()
    print('-- RETRIEVAL METRICS --')
    print(f"Context Recall@{args.top_k} : {context_recall:.3f}")
    print(f"Hit Rate@{args.top_k}       : {hit_rate:.3f}  ({hit_count}/{questions_with_refs})")
    print()
    print('-- GENERATION METRICS --')
    print(f"Answer F1        : {avg_f1:.3f}")
    print(f"Exact Match      : {em_rate:.3f}  ({em_count}/{n})")
    print('=' * 60)
    print()
    print('-- CONTEXT RECALL BY QUESTION TYPE --')
    for qtype, stats in sorted(type_stats.items()):
        rc = stats['recall_count']
        if rc > 0:
            recall = stats['recall_sum'] / rc
            hit = stats['hit']
            total = stats['total_with_refs']
            print(f"  {qtype:<24} : recall={recall:.3f}  hit={hit}/{total}")

    # Save detailed results
    output_data = {
        'summary': {
            'dataset': args.dataset,
            'total_questions': n,
            'questions_with_refs': questions_with_refs,
            'chunk_size': args.chunk_size,
            'overlap': args.overlap,
            'top_k': args.top_k,
            'context_recall': context_recall,
            'hit_rate': hit_rate,
            'answer_f1': avg_f1,
            'exact_match': em_rate,
            'by_type': {
                qt: {
                    'context_recall': s['recall_sum'] / s['recall_count'] if s['recall_count'] > 0 else 0.0,
                    'hit_rate': s['hit'] / s['total_with_refs'] if s['total_with_refs'] > 0 else 0.0,
                }
                for qt, s in type_stats.items()
            }
        },
        'results': results,
    }

    os.makedirs(os.path.dirname(DEFAULT_RESULTS), exist_ok=True)
    with open(DEFAULT_RESULTS, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logging.info(f"\nDetailed results saved: {DEFAULT_RESULTS}")


if __name__ == '__main__':
    main()
