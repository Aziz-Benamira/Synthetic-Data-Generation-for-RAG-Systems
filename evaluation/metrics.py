"""
Evaluation Metrics for RAG Systems
====================================

Métriques d'évaluation adaptées à notre projet :

RETRIEVAL (le retriever trouve-t-il les bons chunks ?) :
  - Hit Rate (Recall@k)  : Le chunk source (gold) est-il dans le top-k ?
  - MRR (Mean Reciprocal Rank) : À quel rang apparaît le chunk gold ?
  - Contextual Precision  : Proportion de chunks pertinents dans le top-k

GENERATION (la réponse générée est-elle bonne ?) :
  - ROUGE-L  : Overlap de sous-séquences entre réponse générée et gold
  - BERTScore : Similarité sémantique (embeddings) entre réponse et gold
  - LLM-as-Judge : Score 0-5 par un LLM juge (fidélité + complétude)
  - Faithfulness : La réponse est-elle fidèle au contexte fourni ?

COMBINED :
  - Score global pondéré
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  RETRIEVAL METRICS
# ══════════════════════════════════════════════════════════════════════════════

def hit_rate_at_k(gold_chunk_id: str, retrieved_chunks: List[Dict], k: int = 5) -> float:
    """
    Le chunk gold est-il présent dans le top-k résultats ?
    
    Returns: 1.0 si trouvé, 0.0 sinon
    """
    retrieved_ids = [c.get("chunk_id", "") for c in retrieved_chunks[:k]]
    return 1.0 if gold_chunk_id in retrieved_ids else 0.0


def reciprocal_rank(gold_chunk_id: str, retrieved_chunks: List[Dict]) -> float:
    """
    Mean Reciprocal Rank (MRR) : 1/rang du premier chunk gold trouvé.
    
    Returns: 1/rang si trouvé, 0.0 sinon
    """
    for i, chunk in enumerate(retrieved_chunks):
        if chunk.get("chunk_id", "") == gold_chunk_id:
            return 1.0 / (i + 1)
    return 0.0


def contextual_precision_at_k(
    gold_chunk_id: str,
    retrieved_chunks: List[Dict],
    gold_content: str,
    k: int = 5,
    overlap_threshold: float = 0.3
) -> float:
    """
    Proportion de chunks dans le top-k qui ont un overlap significatif 
    avec le contenu gold (pas seulement le chunk_id exact).
    
    Cela capture les cas où plusieurs chunks contiennent de l'info pertinente.
    """
    relevant_count = 0
    gold_words = set(_tokenize(gold_content))
    
    for chunk in retrieved_chunks[:k]:
        chunk_words = set(_tokenize(chunk.get("content", "")))
        if not gold_words:
            continue
        overlap = len(gold_words & chunk_words) / len(gold_words)
        if overlap >= overlap_threshold or chunk.get("chunk_id") == gold_chunk_id:
            relevant_count += 1
    
    return relevant_count / k if k > 0 else 0.0


def retrieval_similarity_score(retrieved_chunks: List[Dict], k: int = 5) -> float:
    """Score moyen de similarité cosine des top-k chunks."""
    similarities = [c.get("similarity", 0) for c in retrieved_chunks[:k]]
    return float(np.mean(similarities)) if similarities else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  GENERATION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_rouge_l(generated: str, reference: str) -> Dict[str, float]:
    """
    ROUGE-L : Plus longue sous-séquence commune (LCS).
    Mesure la qualité de la couverture du contenu.
    
    Returns: {"precision": float, "recall": float, "f1": float}
    """
    from rouge_score import rouge_scorer
    
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = scorer.score(reference, generated)
    
    return {
        "precision": round(scores["rougeL"].precision, 4),
        "recall": round(scores["rougeL"].recall, 4),
        "f1": round(scores["rougeL"].fmeasure, 4)
    }


def compute_bert_score(
    generated: str,
    reference: str,
    model_type: str = "bert-base-multilingual-cased",
    device: str = "cpu"
) -> Dict[str, float]:
    """
    BERTScore : Similarité sémantique via embeddings BERT.
    Utilise bert-base-multilingual-cased pour le français.
    
    Returns: {"precision": float, "recall": float, "f1": float}
    """
    from bert_score import score as bert_score_fn
    
    P, R, F1 = bert_score_fn(
        [generated], [reference],
        model_type=model_type,
        device=device,
        verbose=False
    )
    
    return {
        "precision": round(P.item(), 4),
        "recall": round(R.item(), 4),
        "f1": round(F1.item(), 4)
    }


def word_overlap_score(generated: str, reference: str) -> float:
    """
    Overlap de mots simple (Jaccard-like) entre la réponse générée et la référence.
    Rapide et interprétable.
    """
    gen_words = set(_tokenize(generated))
    ref_words = set(_tokenize(reference))
    
    if not ref_words:
        return 0.0
    
    intersection = gen_words & ref_words
    union = gen_words | ref_words
    
    return len(intersection) / len(union) if union else 0.0


def faithfulness_score(generated: str, context: str) -> float:
    """
    Faithfulness : La réponse est-elle fidèle au contexte fourni ?
    
    Mesure la proportion de n-grams de la réponse qui apparaissent dans le contexte.
    Un score élevé = la réponse ne "hallucine" pas.
    """
    gen_words = _tokenize(generated)
    context_words_set = set(_tokenize(context))
    
    if not gen_words:
        return 0.0
    
    # Vérifier les bigrams de la réponse présents dans le contexte
    gen_bigrams = set(zip(gen_words, gen_words[1:]))
    ctx_bigrams = set()
    ctx_words_list = _tokenize(context)
    ctx_bigrams = set(zip(ctx_words_list, ctx_words_list[1:]))
    
    if not gen_bigrams:
        # Fallback sur les unigrams
        grounded = sum(1 for w in gen_words if w in context_words_set)
        return grounded / len(gen_words)
    
    grounded = sum(1 for bg in gen_bigrams if bg in ctx_bigrams)
    return grounded / len(gen_bigrams)


# ══════════════════════════════════════════════════════════════════════════════
#  LLM-AS-JUDGE
# ══════════════════════════════════════════════════════════════════════════════

LLM_JUDGE_PROMPT = """Tu es un évaluateur expert. Compare la réponse générée par un système RAG avec la réponse de référence (gold standard).

**Question :** {question}

**Réponse de référence (Gold) :** {gold_answer}

**Réponse du système RAG :** {generated_answer}

**Contexte fourni au RAG :** {context}

Évalue la réponse du système RAG selon ces 4 critères (chacun sur 5) :

1. **Exactitude factuelle** (0-5) : Les faits sont-ils corrects par rapport à la référence ?
2. **Complétude** (0-5) : La réponse couvre-t-elle tous les points importants de la référence ?
3. **Fidélité au contexte** (0-5) : La réponse est-elle basée sur le contexte fourni (pas d'hallucination) ?
4. **Clarté pédagogique** (0-5) : La réponse est-elle claire et bien structurée ?

Réponds EXACTEMENT dans ce format JSON :
{{"exactitude": <0-5>, "completude": <0-5>, "fidelite": <0-5>, "clarte": <0-5>, "commentaire": "<bref commentaire>"}}"""


def llm_judge_score(
    question: str,
    gold_answer: str,
    generated_answer: str,
    context: str,
    llm: Any,
    temperature: float = 0.1,
    max_tokens: int = 300
) -> Dict[str, Any]:
    """
    LLM-as-Judge : Un LLM note la qualité de la réponse RAG.
    
    Utilise notre modèle llama-cpp pour évaluer sur 4 critères.
    
    Args:
        question: La question posée
        gold_answer: Notre réponse gold standard
        generated_answer: Réponse générée par le RAG
        context: Contexte fourni au RAG
        llm: Instance Llama (llama-cpp-python)
        
    Returns:
        {
            "exactitude": int,
            "completude": int,
            "fidelite": int,
            "clarte": int,
            "score_moyen": float,
            "commentaire": str
        }
    """
    prompt = LLM_JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
        context=context[:3000]  # Limiter pour éviter overflow
    )
    
    messages = [
        {"role": "system", "content": "Tu es un évaluateur expert. Réponds uniquement en JSON valide."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        raw = response["choices"][0]["message"]["content"]
        
        # Extraire le JSON (gérer <think>...</think> pour DeepSeek)
        if "</think>" in raw:
            raw = raw.split("</think>")[-1].strip()
        
        # Parser le JSON
        scores = _parse_judge_json(raw)
        
        return scores
        
    except Exception as e:
        logger.warning(f"LLM judge failed: {e}")
        return {
            "exactitude": -1,
            "completude": -1,
            "fidelite": -1,
            "clarte": -1,
            "score_moyen": -1.0,
            "commentaire": f"Error: {str(e)}"
        }


def _parse_judge_json(raw: str) -> Dict[str, Any]:
    """Parse le JSON du juge LLM, avec gestion des erreurs."""
    import json
    
    # Chercher le JSON dans la réponse
    json_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
    if not json_match:
        # Essayer de trouver les scores avec regex
        return _parse_judge_regex(raw)
    
    try:
        data = json.loads(json_match.group())
        
        scores = {
            "exactitude": int(data.get("exactitude", 0)),
            "completude": int(data.get("completude", 0)),
            "fidelite": int(data.get("fidelite", data.get("fidélité", 0))),
            "clarte": int(data.get("clarte", data.get("clarté", 0))),
            "commentaire": str(data.get("commentaire", ""))
        }
        scores["score_moyen"] = round(
            np.mean([scores["exactitude"], scores["completude"],
                     scores["fidelite"], scores["clarte"]]), 2
        )
        return scores
        
    except (json.JSONDecodeError, ValueError):
        return _parse_judge_regex(raw)


def _parse_judge_regex(raw: str) -> Dict[str, Any]:
    """Fallback: extraction par regex si le JSON est malformé."""
    scores = {}
    for key in ["exactitude", "completude", "fidelite", "fidélité", "clarte", "clarté"]:
        match = re.search(rf'"{key}"\s*:\s*(\d)', raw)
        if match:
            clean_key = key.replace("é", "e")
            scores[clean_key] = int(match.group(1))
    
    result = {
        "exactitude": scores.get("exactitude", -1),
        "completude": scores.get("completude", -1),
        "fidelite": scores.get("fidelite", -1),
        "clarte": scores.get("clarte", -1),
        "commentaire": "parsed via regex fallback"
    }
    valid = [v for v in [result["exactitude"], result["completude"],
                          result["fidelite"], result["clarte"]] if v >= 0]
    result["score_moyen"] = round(np.mean(valid), 2) if valid else -1.0
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINED EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_single_qa(
    question: str,
    gold_answer: str,
    gold_chunk_id: str,
    gold_chunk_content: str,
    generated_answer: str,
    retrieved_chunks: List[Dict],
    context_used: str,
    llm_judge: Any = None,
    top_k: int = 5,
    bert_device: str = "cpu"
) -> Dict[str, Any]:
    """
    Évaluation complète d'une paire QA : retrieval + generation.
    
    Args:
        question: La question
        gold_answer: Réponse gold standard
        gold_chunk_id: ID du chunk source gold
        gold_chunk_content: Contenu du chunk source gold
        generated_answer: Réponse générée par le RAG
        retrieved_chunks: Chunks retournés par le retriever
        context_used: Contexte formaté envoyé au LLM
        llm_judge: Instance Llama pour LLM-as-Judge (optionnel)
        top_k: k pour les métriques retrieval
        bert_device: Device pour BERTScore
    
    Returns:
        Dict avec toutes les métriques
    """
    results = {}
    
    # ── Retrieval metrics ──
    results["retrieval"] = {
        "hit_rate_at_5": hit_rate_at_k(gold_chunk_id, retrieved_chunks, k=5),
        "hit_rate_at_3": hit_rate_at_k(gold_chunk_id, retrieved_chunks, k=3),
        "hit_rate_at_1": hit_rate_at_k(gold_chunk_id, retrieved_chunks, k=1),
        "mrr": reciprocal_rank(gold_chunk_id, retrieved_chunks),
        "contextual_precision": contextual_precision_at_k(
            gold_chunk_id, retrieved_chunks, gold_chunk_content, k=top_k
        ),
        "avg_similarity": retrieval_similarity_score(retrieved_chunks, k=top_k),
        "retrieved_ids": [c.get("chunk_id", "") for c in retrieved_chunks[:top_k]]
    }
    
    # ── Generation metrics ──
    rouge = compute_rouge_l(generated_answer, gold_answer)
    
    results["generation"] = {
        "rouge_l": rouge,
        "word_overlap": word_overlap_score(generated_answer, gold_answer),
        "faithfulness": faithfulness_score(generated_answer, context_used)
    }
    
    # BERTScore (peut être lent, on le fait quand même)
    try:
        bert = compute_bert_score(generated_answer, gold_answer, device=bert_device)
        results["generation"]["bert_score"] = bert
    except Exception as e:
        logger.warning(f"BERTScore failed: {e}")
        results["generation"]["bert_score"] = {"precision": -1, "recall": -1, "f1": -1}
    
    # LLM-as-Judge (optionnel)
    if llm_judge is not None:
        results["llm_judge"] = llm_judge_score(
            question, gold_answer, generated_answer,
            context_used, llm_judge
        )
    
    return results


def compute_aggregate_metrics(all_results: List[Dict]) -> Dict[str, Any]:
    """
    Agrège les métriques de toutes les paires QA en un résumé.
    
    Returns:
        Dict avec moyennes, médianes, distributions
    """
    n = len(all_results)
    if n == 0:
        return {}
    
    # Retrieval
    hit5 = [r["retrieval"]["hit_rate_at_5"] for r in all_results]
    hit3 = [r["retrieval"]["hit_rate_at_3"] for r in all_results]
    hit1 = [r["retrieval"]["hit_rate_at_1"] for r in all_results]
    mrrs = [r["retrieval"]["mrr"] for r in all_results]
    avg_sims = [r["retrieval"]["avg_similarity"] for r in all_results]
    
    # Generation
    rouge_f1s = [r["generation"]["rouge_l"]["f1"] for r in all_results]
    word_overlaps = [r["generation"]["word_overlap"] for r in all_results]
    faiths = [r["generation"]["faithfulness"] for r in all_results]
    
    bert_f1s = [r["generation"]["bert_score"]["f1"] for r in all_results
                if r["generation"]["bert_score"]["f1"] >= 0]
    
    agg = {
        "total_questions": n,
        "retrieval": {
            "hit_rate@5": round(np.mean(hit5), 4),
            "hit_rate@3": round(np.mean(hit3), 4),
            "hit_rate@1": round(np.mean(hit1), 4),
            "mrr": round(np.mean(mrrs), 4),
            "avg_similarity": round(np.mean(avg_sims), 4),
        },
        "generation": {
            "rouge_l_f1_mean": round(np.mean(rouge_f1s), 4),
            "rouge_l_f1_median": round(np.median(rouge_f1s), 4),
            "word_overlap_mean": round(np.mean(word_overlaps), 4),
            "faithfulness_mean": round(np.mean(faiths), 4),
        }
    }
    
    if bert_f1s:
        agg["generation"]["bert_score_f1_mean"] = round(np.mean(bert_f1s), 4)
        agg["generation"]["bert_score_f1_median"] = round(np.median(bert_f1s), 4)
    
    # LLM Judge aggregates
    judge_scores = []
    for r in all_results:
        if "llm_judge" in r and r["llm_judge"].get("score_moyen", -1) >= 0:
            judge_scores.append(r["llm_judge"]["score_moyen"])
    
    if judge_scores:
        agg["llm_judge"] = {
            "score_moyen_mean": round(np.mean(judge_scores), 4),
            "score_moyen_median": round(np.median(judge_scores), 4),
            "count": len(judge_scores)
        }
    
    return agg


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> List[str]:
    """Tokenisation simple pour les métriques de mots."""
    text = text.lower()
    text = re.sub(r'[^\w\sàâäéèêëïîôùûüÿçœæ]', ' ', text)
    words = text.split()
    # Filtrer les stop words très courts
    return [w for w in words if len(w) > 2]
