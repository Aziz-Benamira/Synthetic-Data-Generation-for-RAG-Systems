"""
RAG Evaluation v2 — Academic MI201 (ENSTA Paris)
=================================================

Extends evaluate_rag.py with:
  — Maloe's metrics  : MRR, ROUGE-L, BERTScore (optional)
  — Aziz/Seif's Critic Agent (Constitutional AI, 5 criteria)
    adapted to LocalModelClient (Ministral-8B)

Pipeline per question:
  1. Retrieve top-k chunks (MiniLM semantic, same as v1)
  2. Context Recall + Hit Rate (same as v1)
  3. MRR   — rank of first matching chunk          [NEW - Maloe]
  4. Generate answer (Ministral-8B, same as v1)
  5. ROUGE-L F1 vs gold answer                     [NEW - Maloe]
  6. BERTScore F1 vs gold answer (optional)        [NEW - Maloe]
  7. Critic Agent evaluation (5 criteria)          [NEW - Aziz/Seif]

Usage:
  python evaluate_rag_v2.py
  python evaluate_rag_v2.py --no-critic          # skip critic (faster)
  python evaluate_rag_v2.py --no-bertscore       # skip BERTScore
  python evaluate_rag_v2.py --limit 10           # test on 10 questions
"""

import os
import sys
import json
import re
import logging
import argparse
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

QAR_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATASET  = os.path.join(QAR_DIR, 'output', 'academic', 'en', 'eval_dataset.json')
DEFAULT_DOC_DIR  = os.path.join(QAR_DIR, 'data', 'academic', 'en', 'doc')
DEFAULT_RESULTS  = os.path.join(QAR_DIR, 'output', 'academic', 'en', 'eval_results_v2.json')
DEFAULT_MODEL    = '/home/ensta/data/Ministral-8B-Instruct-2410'
EMBED_MODEL      = 'paraphrase-multilingual-MiniLM-L12-v2'

CHAPTERS = [
    ('Ch1_Intro_ML',        'Chapter 1: Introduction to Machine Learning'),
    ('Ch2_Decision_Trees',  'Chapter 2: Decision Trees and Ensemble Methods'),
    ('Ch3_SVM',             'Chapter 3: Regularization and Support Vector Machines'),
    ('Ch4_Neural_Networks', 'Chapter 4: Introduction to Neural Networks'),
    ('Ch5_Unsupervised',    'Chapter 5: Unsupervised Learning'),
]

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question based ONLY on the provided context. "
    "If the context does not contain the answer, say 'Unable to answer. The provided context "
    "does not contain information about this topic.' Do not use prior knowledge."
)

# ─────────────────────────────────────────────────────────────────────────────
# Critic Agent (from Aziz/Seif branch — Constitutional AI)
# ─────────────────────────────────────────────────────────────────────────────

CRITIC_SYSTEM_PROMPT = """You are an expert quality evaluator for Question-Answer datasets.

Evaluate the QA pair against these 5 criteria. Be strict but fair.

1. ANCHORING — Is the answer ENTIRELY derivable from the provided chunk? No external info.
2. LOCAL_ANSWERABILITY — Can the question be answered with ONLY this chunk?
3. FACTUAL_ACCURACY — Is the answer factually correct relative to the chunk? No hallucinations.
4. COMPLETENESS — Does the answer address ALL aspects of the question?
5. CLARITY — Are the question and answer clear and unambiguous?

Score each criterion 0.0 to 1.0. Final decision: PASS (all scores >= 0.6) or REJECT.

Respond ONLY with valid JSON, no extra text:
{
  "criteria": {
    "anchoring":            {"score": 0.0-1.0, "explanation": "..."},
    "local_answerability":  {"score": 0.0-1.0, "explanation": "..."},
    "factual_accuracy":     {"score": 0.0-1.0, "explanation": "..."},
    "completeness":         {"score": 0.0-1.0, "explanation": "..."},
    "clarity":              {"score": 0.0-1.0, "explanation": "..."}
  },
  "overall_score": 0.0-1.0,
  "decision": "pass" or "reject",
  "rejection_reasons": ["..."]
}"""

CRITIC_USER_TEMPLATE = """=== SOURCE CHUNK ===
Chapter: {chapter}
Content:
---
{chunk_content}
---

=== QA PAIR TO EVALUATE ===
Question: {question}
Answer: {answer}
Supporting references (verbatim from source): {refs}

Evaluate this QA pair against the 5 criteria. Output JSON only."""


def run_critic(local_client: Any, question: str, answer: str,
               refs: List[str], chunk_content: str, chapter: str) -> Dict:
    """
    Run Aziz/Seif's critic agent on a QA pair using LocalModelClient.
    Returns parsed evaluation dict or None on failure.
    """
    refs_str = " | ".join(refs) if refs else "None"
    user_prompt = CRITIC_USER_TEMPLATE.format(
        chapter=chapter,
        chunk_content=chunk_content[:2000],  # truncate very long chunks
        question=question,
        answer=answer,
        refs=refs_str,
    )
    responses = local_client.generate(
        [{'system_prompt': CRITIC_SYSTEM_PROMPT, 'user_prompt': user_prompt}],
        max_new_tokens=1024,
    )
    raw = responses[0]

    # Extract JSON from response
    json_match = re.search(r'\{[\s\S]*\}', raw)
    if not json_match:
        logger.warning("Critic: no JSON found in response")
        return None
    try:
        parsed = json.loads(json_match.group())
        return parsed
    except json.JSONDecodeError as e:
        logger.warning(f"Critic: JSON parse error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Maloe's metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_mrr(refs: List[str], ranked_chunks: List[Tuple[str, float]],
                threshold: float = 0.45, embed_fn=None) -> float:
    """
    MRR — 1 / rank of first chunk that matches any ref (via cosine sim).
    From Maloe's trad_metrics.py logic.
    """
    if not refs or embed_fn is None:
        return 0.0
    for rank, (chunk_text, _sim) in enumerate(ranked_chunks, start=1):
        for ref in refs:
            texts = [ref, chunk_text]
            embs = embed_fn(texts)
            sim = float(embs[0] @ embs[1])
            if sim >= threshold:
                return 1.0 / rank
    return 0.0


def compute_rouge_l(generated: str, reference: str) -> Dict[str, float]:
    """
    ROUGE-L via rouge_score package (Maloe's generation metrics).
    Falls back to a simple word-overlap F1 if package not installed.
    """
    try:
        from rouge_score import rouge_scorer as rs
        scorer = rs.RougeScorer(["rougeL"], use_stemmer=False)
        scores = scorer.score(reference, generated)
        return {
            "precision": round(scores["rougeL"].precision, 4),
            "recall":    round(scores["rougeL"].recall, 4),
            "f1":        round(scores["rougeL"].fmeasure, 4),
        }
    except ImportError:
        # Fallback: token F1 (same as v1 answer_f1)
        pred = set(generated.lower().split())
        gold = set(reference.lower().split())
        if not pred or not gold:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        common = pred & gold
        p = len(common) / len(pred)
        r = len(common) / len(gold)
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def compute_bert_score(generated: str, reference: str,
                       device: str = 'cuda') -> Optional[Dict[str, float]]:
    """
    BERTScore (Maloe's GenerationEvaluator).
    Returns None if bert_score is not installed.
    """
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(
            [generated], [reference],
            model_type="xlm-roberta-large",
            device=device,
            verbose=False,
        )
        return {
            "precision": round(float(P[0]), 4),
            "recall":    round(float(R[0]), 4),
            "f1":        round(float(F1[0]), 4),
        }
    except ImportError:
        return None
    except Exception as e:
        logger.warning(f"BERTScore failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Chunking (same as v1)
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = 500, overlap: int = 100) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += size - overlap
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Semantic index with ranked retrieval (extends v1)
# ─────────────────────────────────────────────────────────────────────────────

class SemanticIndex:
    def __init__(self, embed_model: SentenceTransformer):
        self.model = embed_model
        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def build(self, chunks: List[str], label: str = ''):
        logger.info(f"Building index: {label} ({len(chunks)} chunks)")
        self.chunks = chunks
        self.embeddings = self._encode(chunks)

    def _encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def retrieve_ranked(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Returns list of (chunk_text, cosine_similarity) sorted by similarity desc."""
        q_emb = self._encode([query])
        sims = (self.embeddings @ q_emb.T).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self.chunks[i], float(sims[i])) for i in top_idx]

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        return [c for c, _ in self.retrieve_ranked(query, top_k)]

    def ref_matches_context(self, ref: str, context_chunks: List[str],
                            threshold: float = 0.45) -> bool:
        texts = [ref] + context_chunks
        embs = self._encode(texts)
        sims = embs[1:] @ embs[0]
        return bool(np.any(sims >= threshold))

    def embed_fn(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts)


# ─────────────────────────────────────────────────────────────────────────────
# Local LLM client (same as v1)
# ─────────────────────────────────────────────────────────────────────────────

class LocalModelClient:
    def __init__(self, model_path: str):
        logger.info(f"Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        logger.info("Loading model (float16, CPU → GPU)")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, local_files_only=True)
        self.model = self.model.cuda()
        used = torch.cuda.memory_allocated() / 1024 ** 3
        logger.info(f"Model on GPU — {used:.1f} GB used")
        self.model.eval()

    def generate(self, tasks: List[Dict], max_new_tokens: int = 512) -> List[str]:
        responses = []
        for i, task in enumerate(tasks):
            messages = [
                {"role": "system", "content": task["system_prompt"]},
                {"role": "user",   "content": task["user_prompt"]},
            ]
            result = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True)
            inputs = result.input_ids.to(self.model.device) if hasattr(result, 'input_ids') \
                else result.to(self.model.device)
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
        return responses


# ─────────────────────────────────────────────────────────────────────────────
# v1 generation metrics (kept for backward compat)
# ─────────────────────────────────────────────────────────────────────────────

def word_f1(pred: str, gold: str) -> float:
    p_tok = set(pred.lower().split())
    g_tok = set(gold.lower().split())
    if not p_tok or not g_tok:
        return 0.0
    common = p_tok & g_tok
    if not common:
        return 0.0
    pr = len(common) / len(p_tok)
    rc = len(common) / len(g_tok)
    return 2 * pr * rc / (pr + rc)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',      default=DEFAULT_DATASET)
    parser.add_argument('--model',        default=DEFAULT_MODEL)
    parser.add_argument('--chunk-size',   type=int, default=500)
    parser.add_argument('--overlap',      type=int, default=100)
    parser.add_argument('--top-k',        type=int, default=5)
    parser.add_argument('--limit',        type=int, default=0,
                        help='Evaluate only first N questions (0 = all)')
    parser.add_argument('--no-critic',    action='store_true',
                        help='Skip critic agent evaluation')
    parser.add_argument('--no-bertscore', action='store_true',
                        help='Skip BERTScore (faster)')
    args = parser.parse_args()

    # ── Load dataset ──────────────────────────────────────────────────────────
    with open(args.dataset, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    if args.limit > 0:
        questions = questions[:args.limit]
    logger.info(f"Loaded {len(questions)} questions from {args.dataset}")

    # ── Load documents ────────────────────────────────────────────────────────
    docs = {}
    for ch_id, ch_name in CHAPTERS:
        p = os.path.join(DEFAULT_DOC_DIR, ch_id, '0', '0.txt')
        if os.path.exists(p):
            docs[ch_name] = open(p, encoding='utf-8').read()
    logger.info(f"Loaded {len(docs)} chapter documents")

    # ── Chunk + build indexes ─────────────────────────────────────────────────
    logger.info(f"Chunking (size={args.chunk_size}, overlap={args.overlap})")
    all_chunks, chapter_chunks = [], {}
    for ch_name, text in docs.items():
        chunks = chunk_text(text, args.chunk_size, args.overlap)
        chapter_chunks[ch_name] = chunks
        all_chunks.extend(chunks)
    logger.info(f"Total chunks: {len(all_chunks)}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Embedding device: {device}")
    embed_model = SentenceTransformer(EMBED_MODEL)

    per_chapter_index: Dict[str, SemanticIndex] = {}
    for ch_name, chunks in chapter_chunks.items():
        idx = SemanticIndex(embed_model)
        idx.build(chunks, ch_name)
        per_chapter_index[ch_name] = idx

    global_index = SemanticIndex(embed_model)
    global_index.build(all_chunks, 'global')

    # ── Load generation model ─────────────────────────────────────────────────
    logger.info(f"Loading generation model: {os.path.basename(args.model)}")
    gen_client = LocalModelClient(args.model)
    logger.info("Generation model ready.")

    ch_id_to_name = {cid: cn for cid, cn in CHAPTERS}

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = []
    # Accumulators for aggregate metrics
    recall_sum, recall_count = 0.0, 0
    hit_count, questions_with_refs = 0, 0
    mrr_sum, mrr_count = 0.0, 0
    f1_sum, em_count = 0.0, 0
    rouge_f1_sum, rouge_count = 0.0, 0
    bert_f1_sum, bert_count = 0.0, 0
    critic_pass, critic_total = 0, 0
    critic_crit_sums = {c: 0.0 for c in
        ["anchoring", "local_answerability", "factual_accuracy", "completeness", "clarity"]}

    for i, q in enumerate(questions):
        question   = q['question']
        gold_ans   = q.get('answer', '')
        refs       = q.get('ref', [])
        qtype      = q.get('question_type', 'unknown')
        chapter_id = q.get('chapter', '')

        ch_name = ch_id_to_name.get(chapter_id, '')
        idx = per_chapter_index.get(ch_name, global_index)

        # ── Retrieve with ranks ───────────────────────────────────────────────
        ranked = idx.retrieve_ranked(question, top_k=args.top_k)
        retrieved = [c for c, _ in ranked]

        # ── Context Recall + Hit Rate (v1 metrics) ────────────────────────────
        ref_recall, hit, mrr = None, None, 0.0
        if refs:
            questions_with_refs += 1
            matched = [r for r in refs if idx.ref_matches_context(r, retrieved)]
            ref_recall = len(matched) / len(refs)
            recall_sum += ref_recall
            recall_count += 1
            hit = len(matched) > 0
            if hit:
                hit_count += 1

            # ── MRR (Maloe) ───────────────────────────────────────────────────
            mrr = compute_mrr(refs, ranked, threshold=0.45,
                              embed_fn=idx.embed_fn)
            mrr_sum += mrr
            mrr_count += 1

        # ── Generate answer ───────────────────────────────────────────────────
        context = '\n\n---\n\n'.join(retrieved)
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        pred_answer = gen_client.generate(
            [{'system_prompt': SYSTEM_PROMPT, 'user_prompt': user_prompt}])[0]

        # ── Word F1 / EM (v1) ─────────────────────────────────────────────────
        f1 = word_f1(pred_answer, gold_ans)
        em = pred_answer.strip().lower() == gold_ans.strip().lower()
        f1_sum += f1
        if em:
            em_count += 1

        # ── ROUGE-L (Maloe) ───────────────────────────────────────────────────
        rouge = compute_rouge_l(pred_answer, gold_ans)
        rouge_f1_sum += rouge['f1']
        rouge_count += 1

        # ── BERTScore (Maloe, optional) ───────────────────────────────────────
        bert = None
        if not args.no_bertscore:
            bert = compute_bert_score(pred_answer, gold_ans, device=device)
            if bert is not None:
                bert_f1_sum += bert['f1']
                bert_count += 1

        # ── Critic Agent (Aziz/Seif) ──────────────────────────────────────────
        critic_result = None
        if not args.no_critic:
            # Use the top retrieved chunk as the source chunk for the critic
            best_chunk = retrieved[0] if retrieved else ""
            critic_result = run_critic(
                local_client=gen_client,
                question=question,
                answer=pred_answer,   # evaluate the GENERATED answer
                refs=refs,
                chunk_content=best_chunk,
                chapter=ch_name,
            )
            if critic_result:
                critic_total += 1
                if critic_result.get('decision') == 'pass':
                    critic_pass += 1
                crit = critic_result.get('criteria', {})
                for key in critic_crit_sums:
                    if key in crit and isinstance(crit[key], dict):
                        critic_crit_sums[key] += crit[key].get('score', 0.0)

        # ── Store result ──────────────────────────────────────────────────────
        results.append({
            'question':        question,
            'question_type':   qtype,
            'chapter':         chapter_id,
            'gold_answer':     gold_ans,
            'pred_answer':     pred_answer,
            'refs':            refs,
            'retrieved_chunks': retrieved,
            # v1 metrics
            'context_recall':  ref_recall,
            'hit':             hit,
            'answer_f1_v1':    round(f1, 4),
            'exact_match':     em,
            # new metrics
            'mrr':             round(mrr, 4),
            'rouge_l':         rouge,
            'bert_score':      bert,
            'critic':          critic_result,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(questions):
            logger.info(f"  Processed {i+1}/{len(questions)} questions")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n = len(questions)
    context_recall = recall_sum / recall_count if recall_count > 0 else 0.0
    hit_rate       = hit_count / questions_with_refs if questions_with_refs > 0 else 0.0
    avg_mrr        = mrr_sum / mrr_count if mrr_count > 0 else 0.0
    avg_f1_v1      = f1_sum / n
    em_rate        = em_count / n
    avg_rouge_f1   = rouge_f1_sum / rouge_count if rouge_count > 0 else 0.0
    avg_bert_f1    = bert_f1_sum / bert_count if bert_count > 0 else None
    critic_pass_rate = critic_pass / critic_total if critic_total > 0 else None
    critic_crit_avg = {k: round(v / critic_total, 4) for k, v in critic_crit_sums.items()} \
        if critic_total > 0 else None

    # ── Print summary ─────────────────────────────────────────────────────────
    print()
    print('=' * 65)
    print('RAG EVALUATION v2 — RESULTS')
    print('=' * 65)
    print(f"Dataset           : {args.dataset}")
    print(f"Total questions   : {n}  (with refs: {questions_with_refs})")
    print(f"Retriever         : multilingual-MiniLM-L12-v2 (cosine, cross-lingual)")
    print(f"Generator         : {os.path.basename(args.model)}")
    print()
    print('─── RETRIEVAL METRICS ─────────────────────────────────────────')
    print(f"  Context Recall@{args.top_k} : {context_recall:.3f}")
    print(f"  Hit Rate@{args.top_k}       : {hit_rate:.3f}  ({hit_count}/{questions_with_refs})")
    print(f"  MRR              : {avg_mrr:.3f}  [Maloe]")
    print()
    print('─── GENERATION METRICS ─────────────────────────────────────────')
    print(f"  Word F1 (v1)     : {avg_f1_v1:.3f}")
    print(f"  ROUGE-L F1       : {avg_rouge_f1:.3f}  [Maloe]")
    if avg_bert_f1 is not None:
        print(f"  BERTScore F1     : {avg_bert_f1:.3f}  [Maloe]")
    else:
        print(f"  BERTScore F1     : N/A (--no-bertscore or pkg missing)")
    print(f"  Exact Match      : {em_rate:.3f}")
    if critic_pass_rate is not None:
        print()
        print('─── CRITIC AGENT (Aziz/Seif — Constitutional AI) ───────────────')
        print(f"  Pass Rate        : {critic_pass_rate:.3f}  ({critic_pass}/{critic_total})")
        if critic_crit_avg:
            for crit, avg in critic_crit_avg.items():
                print(f"  {crit:<22} : {avg:.3f}")
    print('=' * 65)

    # ── Save ──────────────────────────────────────────────────────────────────
    summary = {
        'dataset':          args.dataset,
        'total_questions':  n,
        'questions_with_refs': questions_with_refs,
        'chunk_size':       args.chunk_size,
        'overlap':          args.overlap,
        'top_k':            args.top_k,
        # Retrieval
        'context_recall':   round(context_recall, 4),
        'hit_rate':         round(hit_rate, 4),
        'mrr':              round(avg_mrr, 4),
        # Generation
        'answer_f1_v1':     round(avg_f1_v1, 4),
        'rouge_l_f1':       round(avg_rouge_f1, 4),
        'bert_score_f1':    round(avg_bert_f1, 4) if avg_bert_f1 is not None else None,
        'exact_match':      round(em_rate, 4),
        # Critic
        'critic_pass_rate': round(critic_pass_rate, 4) if critic_pass_rate is not None else None,
        'critic_criterion_averages': critic_crit_avg,
    }

    output = {'summary': summary, 'results': results}
    os.makedirs(os.path.dirname(DEFAULT_RESULTS), exist_ok=True)
    with open(DEFAULT_RESULTS, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved: {DEFAULT_RESULTS}")


if __name__ == '__main__':
    main()
