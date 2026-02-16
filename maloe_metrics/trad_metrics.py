
import math
from typing import List, Dict, Tuple, Union
from collections import Counter


# =====================================================================
# 1. RETRIEVER METRICS - NON-RANK BASED
# =====================================================================

def unrankedMetrics(predictions, ground_truth):
    """
    Compute retrieval metrics: accuracy, precision, recall, and F1 score.
    
    Args:
        predictions: List of predicted relevant item indices
        ground_truth: List of ground truth relevant item indices
    
    Returns:
        dict: Dictionary containing accuracy, precision, recall, and f1 score
    """
    pred_set = set(predictions)
    gt_set = set(ground_truth)
    
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = tp / len(gt_set) if len(gt_set) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# =====================================================================
# 2. RETRIEVER METRICS - RANK BASED
# =====================================================================

def mean_reciprocal_rank(predictions_list: List[List[int]], 
                         ground_truth_list: List[List[int]]) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    The position of the first relevant item in the ranked results.
    MRR = (1/|Q|) * sum(1/rank_i) where rank_i is position of first relevant item for query i.
    
    Args:
        predictions_list: List of ranked predictions for each query
        ground_truth_list: List of ground truth relevant items for each query
    
    Returns:
        float: Mean Reciprocal Rank score (0-1)
    """
    mrr_scores = []
    
    for predictions, ground_truth in zip(predictions_list, ground_truth_list):
        gt_set = set(ground_truth)
        # Find position (1-indexed) of first relevant item
        rank = None
        for i, pred in enumerate(predictions, start=1):
            if pred in gt_set:
                rank = i
                break
        
        if rank is not None:
            mrr_scores.append(1.0 / rank)
        else:
            mrr_scores.append(0.0)
    
    return sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0


def dcg_at_k(predictions: List[int], ground_truth: List[int], k: int = None) -> float:
    """
    Compute Discounted Cumulative Gain (DCG@k).
    
    DCG = sum(relevance_i / log2(i+1)) for i from 1 to k
    where relevance_i is 1 if prediction at rank i is in ground truth, 0 otherwise.
    
    Args:
        predictions: Ranked list of predictions
        ground_truth: List of relevant items
        k: Number of top results to consider (None for all)
    
    Returns:
        float: DCG@k score
    """
    gt_set = set(ground_truth)
    dcg = 0.0
    
    for i, pred in enumerate(predictions[:k], start=1):
        if pred in gt_set:
            dcg += 1.0 / math.log2(i + 1)
    
    return dcg


def idcg_at_k(ground_truth: List[int], k: int = None) -> float:
    """
    Compute Ideal Discounted Cumulative Gain (IDCG@k).
    
    The maximum possible DCG if all relevant items are ranked first.
    
    Args:
        ground_truth: List of relevant items
        k: Number of top results to consider (None for all)
    
    Returns:
        float: IDCG@k score
    """
    num_relevant = len(ground_truth)
    if k is not None:
        num_relevant = min(num_relevant, k)
    
    idcg = 0.0
    for i in range(1, num_relevant + 1):
        idcg += 1.0 / math.log2(i + 1)
    
    return idcg


def ndcg_at_k(predictions_list: List[List[int]], 
              ground_truth_list: List[List[int]], 
              k: int = None) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@k).
    
    NDCG@k = DCG@k / IDCG@k
    This measures ranking quality, giving more weight to relevant items at top.
    
    Args:
        predictions_list: List of ranked predictions for each query
        ground_truth_list: List of ground truth relevant items for each query
        k: Number of top results to consider (None for all)
    
    Returns:
        float: NDCG@k score (0-1)
    """
    ndcg_scores = []
    
    for predictions, ground_truth in zip(predictions_list, ground_truth_list):
        dcg = dcg_at_k(predictions, ground_truth, k)
        idcg = idcg_at_k(ground_truth, k)
        
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)
    
    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0


def average_precision_at_k(predictions: List[int], 
                           ground_truth: List[int], 
                           k: int = None) -> float:
    """
    Compute Average Precision (AP@k) for a single query.
    
    AP = sum(precision@i / num_relevant) for i where prediction_i is relevant
    
    Args:
        predictions: Ranked list of predictions
        ground_truth: List of relevant items
        k: Number of top results to consider (None for all)
    
    Returns:
        float: Average Precision score
    """
    gt_set = set(ground_truth)
    num_relevant = len(gt_set)
    
    if num_relevant == 0:
        return 0.0
    
    predictions = predictions[:k] if k is not None else predictions
    
    num_hits = 0
    ap = 0.0
    
    for i, pred in enumerate(predictions, start=1):
        if pred in gt_set:
            num_hits += 1
            precision_at_i = num_hits / i
            ap += precision_at_i
    
    return ap / num_relevant


def mean_average_precision(predictions_list: List[List[int]], 
                          ground_truth_list: List[List[int]], 
                          k: int = None) -> float:
    """
    Compute Mean Average Precision (MAP@k).
    
    MAP = (1/|Q|) * sum(AP_i) where AP_i is average precision for query i.
    Takes into account the order of all relevant documents.
    
    Args:
        predictions_list: List of ranked predictions for each query
        ground_truth_list: List of ground truth relevant items for each query
        k: Number of top results to consider (None for all)
    
    Returns:
        float: MAP@k score (0-1)
    """
    ap_scores = []
    
    for predictions, ground_truth in zip(predictions_list, ground_truth_list):
        ap = average_precision_at_k(predictions, ground_truth, k)
        ap_scores.append(ap)
    
    return sum(ap_scores) / len(ap_scores) if ap_scores else 0.0


# =====================================================================
# 3. GENERATION METRICS - TRADITIONAL
# =====================================================================

def exact_match(predictions: List[str], references: List[str]) -> float:
    """
    Compute Exact Match (EM) score.
    
    Binary metric: 1 if prediction exactly matches reference, 0 otherwise.
    Useful for structured answers (dates, names) but poor for free-form text.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
    
    Returns:
        float: Exact Match ratio (0-1)
    """
    matches = sum(1 for pred, ref in zip(predictions, references) 
                  if pred.strip().lower() == ref.strip().lower())
    
    return matches / len(predictions) if predictions else 0.0


def meteor(prediction: str, reference: str) -> float:
    """
    Simplified METEOR (Metric for Evaluation of Translation with Explicit ORdering).
    
    Combines precision, recall with synonym handling.
    Simplified implementation (without full lemmatization/synonyms).
    
    Formula: (1-p) * ((α²+1)*Prec*Rec)/(Rec + α*Prec)
    where p is penalty factor for word order
    
    Args:
        prediction: Predicted text
        reference: Reference text
    
    Returns:
        float: METEOR score (0-1)
    """
    import difflib
    
    alpha = 0.9  # Weight for precision vs recall
    penalty_weight = 0.5  # Penalty for word order differences
    
    # Tokenize
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    
    # Count matches (simple exact match of tokens)
    matches = sum(1 for token in pred_tokens if token in ref_tokens)
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    precision = matches / len(pred_tokens)
    recall = matches / len(ref_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    # Calculate sequence matcher for word order penalty
    matcher = difflib.SequenceMatcher(None, pred_tokens, ref_tokens)
    matching_blocks = matcher.get_matching_blocks()
    num_chunks = len(matching_blocks)
    penalty = penalty_weight * (num_chunks / max(len(pred_tokens), len(ref_tokens)))
    
    f_score = ((alpha**2 + 1) * precision * recall) / (recall + alpha * precision)
    meteor_score = (1 - penalty) * f_score
    
    return max(0.0, min(1.0, meteor_score))


def meteor_batch(predictions: List[str], references: List[str]) -> float:
    """
    Compute average METEOR score over multiple predictions.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
    
    Returns:
        float: Average METEOR score (0-1)
    """
    scores = [meteor(pred, ref) for pred, ref in zip(predictions, references)]
    return sum(scores) / len(scores) if scores else 0.0


# =====================================================================
# 4. RISK-AWARE METRICS (Answerability-based)
# =====================================================================

def risk_aware_metrics(answerable_kept: int, 
                       unanswerable_kept: int,
                       unanswerable_discarded: int,
                       answerable_discarded: int) -> Dict[str, float]:
    """
    Compute risk-aware metrics for systems that can abstain from answering.
    
    Categories:
    - AK (Answerable, Kept): Correct answers provided
    - UK (Unanswerable, Kept): Hallucinations/errors
    - UD (Unanswerable, Discarded): Correctly rejected
    - AD (Answerable, Discarded): Missed opportunities
    
    Args:
        answerable_kept: Count of AK
        unanswerable_kept: Count of UK
        unanswerable_discarded: Count of UD
        answerable_discarded: Count of AD
    
    Returns:
        dict: Risk, Prudence, Alignment, and Coverage metrics
    """
    ak = answerable_kept
    uk = unanswerable_kept
    ud = unanswerable_discarded
    ad = answerable_discarded
    
    total = ak + uk + ud + ad
    
    # Risk: Hallucination rate among provided answers
    risk = uk / (ak + uk) if (ak + uk) > 0 else 0.0
    
    # Prudence: Ability to detect unanswerable questions (Recall on unanswerable)
    prudence = ud / (uk + ud) if (uk + ud) > 0 else 0.0
    
    # Alignment: Overall correctness of decision to answer/abstain
    alignment = (ak + ud) / total if total > 0 else 0.0
    
    # Coverage: System's answer rate
    coverage = (ak + uk) / total if total > 0 else 0.0
    
    return {
        "risk": risk,
        "prudence": prudence,
        "alignment": alignment,
        "coverage": coverage
    }


# =====================================================================
# 5. EFFICIENCY METRICS
# =====================================================================

def latency_metrics(ttft_milliseconds: float, 
                    total_latency_milliseconds: float) -> Dict[str, Union[float, str]]:
    """
    Compute latency metrics for generation.
    
    Args:
        ttft_milliseconds: Time to First Token in milliseconds
        total_latency_milliseconds: Total generation time in milliseconds
    
    Returns:
        dict: Latency metrics
    """
    return {
        "ttft_ms": ttft_milliseconds,
        "total_latency_ms": total_latency_milliseconds,
        "ttft_s": ttft_milliseconds / 1000,
        "total_latency_s": total_latency_milliseconds / 1000
    }


def cost_metrics(input_tokens: int,
                output_tokens: int,
                cost_per_input_token: float,
                cost_per_output_token: float) -> Dict[str, float]:
    """
    Compute cost metrics for API-based LLM usage.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cost_per_input_token: Cost per input token (e.g., $0.000001)
        cost_per_output_token: Cost per output token (usually higher)
    
    Returns:
        dict: Cost breakdown
    """
    input_cost = input_tokens * cost_per_input_token
    output_cost = output_tokens * cost_per_output_token
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "total_tokens": input_tokens + output_tokens,
        "cost_per_token": total_cost / (input_tokens + output_tokens) if (input_tokens + output_tokens) > 0 else 0
    }


def retriever_roi(retriever_cost: float,
                  context_tokens_per_query: int,
                  llm_cost_per_input_token: float,
                  num_queries: int) -> Dict[str, float]:
    """
    Compute Return On Investment (ROI) of using a retriever.
    
    If retriever is more accurate, it retrieves denser, more relevant context,
    reducing the total context size needed and thus LLM input tokens.
    
    Args:
        retriever_cost: Cost per retrieval query
        context_tokens_per_query: Average context tokens retrieved
        llm_cost_per_input_token: Cost per LLM input token
        num_queries: Number of queries processed
    
    Returns:
        dict: Cost analysis
    """
    total_retriever_cost = retriever_cost * num_queries
    llm_context_cost = context_tokens_per_query * llm_cost_per_input_token * num_queries
    total_cost = total_retriever_cost + llm_context_cost
    
    return {
        "retriever_cost": total_retriever_cost,
        "llm_context_cost": llm_context_cost,
        "total_cost": total_cost,
        "cost_per_query": total_cost / num_queries if num_queries > 0 else 0
    }


# =====================================================================
# 6. HELPER FUNCTIONS FOR EVALUATION
# =====================================================================

def evaluate_retriever(predictions_list: List[List[int]],
                      ground_truth_list: List[List[int]],
                      k_values: List[int] = None) -> Dict:
    """
    Comprehensive retriever evaluation.
    
    Args:
        predictions_list: List of ranked predictions for each query
        ground_truth_list: List of ground truth relevant items for each query
        k_values: List of k values for rank-based metrics (default: [5, 10, 20])
    
    Returns:
        dict: All retriever metrics
    """
    if k_values is None:
        k_values = [5, 10, 20]
    
    results = {
        "non_rank_based": {},
        "rank_based": {}
    }
    
    # Non-rank based (using first query only for demonstration)
    if predictions_list and ground_truth_list:
        non_ranked = unrankedMetrics(predictions_list[0], ground_truth_list[0])
        results["non_rank_based"] = non_ranked
    
    # Rank-based metrics
    results["rank_based"]["mrr"] = mean_reciprocal_rank(predictions_list, ground_truth_list)
    
    for k in k_values:
        results["rank_based"][f"ndcg@{k}"] = ndcg_at_k(predictions_list, ground_truth_list, k)
        results["rank_based"][f"map@{k}"] = mean_average_precision(predictions_list, ground_truth_list, k)
    
    return results


def evaluate_generation(predictions: List[str],
                       references: List[str]) -> Dict:
    """
    Comprehensive generation evaluation (without LLM-based metrics).
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
    
    Returns:
        dict: All generation metrics
    """
    return {
        "exact_match": exact_match(predictions, references),
        "meteor": meteor_batch(predictions, references)
    }
