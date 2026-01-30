import math
from typing import List, Dict, Union, Optional
import json


def placeholder_llm_call(model, input_text: str, warning=True) -> Dict:
    """
    Placeholder for actual LLM calls. To replace later with real API calls.
    
    Args:
        model: Model identifier (will be used when connecting to real API)
        input_text: Input prompt/text to send to LLM
        warning: Whether to print warning message
    
    Returns:
        dict: Placeholder response structure
    """
    if warning:
        print("Warning: Using placeholder model call (doesn't actually call anything)")
    return {"placeholder_var": 0, "response": "placeholder"}


# =====================================================================
# 1. LLM-AS-JUDGE METRICS
# =====================================================================

def llm_as_judge_answer_accuracy(model: str,
                                query: str,
                                response: str,
                                reference_response: str,
                                detailed: bool = False) -> Union[float, Dict]:
    """
    LLM-as-judge: Evaluate answer accuracy against a reference answer using dual templates.
    
    Uses two distinct templates to ensure robust assessment:
    - Template 1: Compare response with reference (normal order)
    - Template 2: Compare response with reference (swapped order - reference vs response)
    
    Each template rates on a scale of 0, 2, or 4:
    - 0: No match / inaccurate
    - 2: Partial match / mostly aligned
    - 4: Exact match / fully consistent
    
    The final score converts to [0,1] scale (0→0, 2→0.5, 4→1) and averages both ratings.
    
    Args:
        model: LLM model identifier (e.g., "gpt-4", "claude-3")
        query: Original user query
        response: Generated response to evaluate
        reference_response: Reference/ground truth response
        detailed: If True, return full details; if False, return only score
    
    Returns:
        float: Score 0-1 (if detailed=False)
        dict: Score and detailed information (if detailed=True)
            - "score": float 0-1 (final answer accuracy score)
            - "score1": float score from template 1
            - "score2": float score from template 2
            - "rating1": int raw rating from template 1 (0, 2, or 4)
            - "rating2": int raw rating from template 2 (0, 2, or 4)
    """
    from prompts import AnswerAccuracyJudge1Prompt, AnswerAccuracyJudge2Prompt
    
    def rating_to_score(rating):
        """Convert rating scale (0, 2, 4) to [0, 1] scale."""
        if rating == 0:
            return 0.0
        elif rating == 2:
            return 0.5
        elif rating == 4:
            return 1.0
        else:
            return None
    
    def extract_rating(llm_response: str) -> Optional[int]:
        """Extract rating from LLM response JSON."""
        try:
            if isinstance(llm_response, str):
                parsed = json.loads(llm_response)
            else:
                parsed = llm_response
            
            rating = parsed.get("rating")
            if rating in [0, 2, 4]:
                return rating
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        return None
    
    # Template 1: Normal order (response vs reference)
    prompt1 = f"""Query: {query}

        User Answer: {response}

        Reference Answer: {reference_response}

        {AnswerAccuracyJudge1Prompt.instruction}"""
    
    result1 = placeholder_llm_call(model, prompt1, warning=False)
    rating1 = extract_rating(result1.get("response", ""))
    
    # Template 2: Swapped order (reference vs response)
    prompt2 = f"""Query: {query}

        User Answer: {reference_response}

        Reference Answer: {response}

        {AnswerAccuracyJudge2Prompt.instruction}"""
    
    result2 = placeholder_llm_call(model, prompt2, warning=False)
    rating2 = extract_rating(result2.get("response", ""))
    
    # Convert ratings to [0,1] scale
    score1 = rating_to_score(rating1)
    score2 = rating_to_score(rating2)
    
    # Calculate final score
    valid_scores = [s for s in [score1, score2] if s is not None]
    
    if len(valid_scores) == 2:
        final_score = (score1 + score2) / 2
    elif len(valid_scores) == 1:
        final_score = valid_scores[0]
    else:
        final_score = 0  # Fallback if both ratings are invalid
    
    if detailed:
        return {
            "score": final_score,
            "score1": score1,
            "score2": score2,
            "rating1": rating1,
            "rating2": rating2
        }
    else:
        return {"score": final_score }


def llm_as_judge_context_support(model: str,
                                 query: str,
                                 response: str,
                                 retrieved_context: str,
                                 detailed: bool = False) -> Union[float, Dict]:
    """
    LLM-as-judge: Evaluate if the response is supported by retrieved context.
    
    Uses an LLM to assess whether the generated answer is grounded in and supported
    by the provided context. This catches hallucinations where the model invents
    facts not found in retrieved documents.
    
    Args:
        model: LLM model identifier (e.g., "gpt-4", "claude-3")
        query: Original user query
        response: Generated response to evaluate
        retrieved_context: Retrieved document chunks/context
        detailed: If True, return full reasoning; if False, return only score
    
    Returns:
        float: Score 0-1 (if detailed=False)
        dict: Score and reasoning (if detailed=True)
            - "score": float 0-1 (1=fully supported, 0=contradicted/hallucinated)
            - "reasoning": str explanation
    """
    from prompts import ContextSupportPrompt
    
    prompt = f"""Query: {query}

Retrieved Context:
{retrieved_context}

Response: {response}

{ContextSupportPrompt.instruction}"""

    result = placeholder_llm_call(model, prompt, warning=False)
    
    # Parse placeholder or real response
    try:
        if isinstance(result.get("response"), str):
            parsed = json.loads(result["response"])
        else:
            parsed = result
    except (json.JSONDecodeError, TypeError):
        # Fallback for placeholder
        parsed = {
            "supported": True,
            "score": 0.5,
            "reasoning": "Placeholder evaluation - replace with real LLM call"
        }
    
    if detailed:
        return {
            "score": parsed.get("score", 0.5),
            "supported": parsed.get("supported", False),
            "reasoning": parsed.get("reasoning", "")
        }
    else:
        return parsed.get("score", 0.5)


def llm_as_judge_answer_relevance(model: str,
                                  query: str,
                                  response: str,
                                  detailed: bool = False) -> Union[float, Dict]:
    """
    LLM-as-judge: Evaluate if response actually answers the query.
    
    Assesses whether the generated response directly addresses the user's question,
    even if perfectly grounded in context. A hallucination-free but irrelevant
    answer would score poorly here.
    
    Args:
        model: LLM model identifier
        query: Original user query
        response: Generated response to evaluate
        detailed: If True, return full reasoning; if False, return only score
    
    Returns:
        float: Score 0-1 (if detailed=False)
        dict: Score and reasoning (if detailed=True)
            - "score": float 0-1 (1=directly answers query, 0=completely irrelevant)
            - "reasoning": str explanation
    """
    from prompts import AnswerRelevancePrompt
    
    prompt = f"""Query: {query}

Response: {response}

{AnswerRelevancePrompt.instruction}"""

    result = placeholder_llm_call(model, prompt, warning=False)
    
    try:
        if isinstance(result.get("response"), str):
            parsed = json.loads(result["response"])
        else:
            parsed = result
    except (json.JSONDecodeError, TypeError):
        parsed = {
            "relevant": True,
            "score": 0.5,
            "reasoning": "Placeholder evaluation - replace with real LLM call"
        }
    
    if detailed:
        return {
            "score": parsed.get("score", 0.5),
            "relevant": parsed.get("relevant", False),
            "reasoning": parsed.get("reasoning", "")
        }
    else:
        return parsed.get("score", 0.5)


def llm_as_judge_coherence(model: str,
                           response: str,
                           detailed: bool = False) -> Union[float, Dict]:
    """
    LLM-as-judge: Evaluate coherence and fluency of the response.
    
    Assesses whether the generated text is clear, well-structured, and easy to read.
    This is independent of factuality but important for user experience.
    
    Args:
        model: LLM model identifier
        response: Generated response to evaluate
        detailed: If True, return full reasoning; if False, return only score
    
    Returns:
        float: Score 0-1 (if detailed=False)
        dict: Score and reasoning (if detailed=True)
            - "score": float 0-1 (1=highly coherent, 0=incoherent)
            - "reasoning": str explanation
    """
    from prompts import CoherencePrompt
    
    prompt = f"""Response: {response}

{CoherencePrompt.instruction}"""

    result = placeholder_llm_call(model, prompt, warning=False)
    
    try:
        if isinstance(result.get("response"), str):
            parsed = json.loads(result["response"])
        else:
            parsed = result
    except (json.JSONDecodeError, TypeError):
        parsed = {
            "coherent": True,
            "score": 0.5,
            "reasoning": "Placeholder evaluation - replace with real LLM call"
        }
    
    if detailed:
        return {
            "score": parsed.get("score", 0.5),
            "coherent": parsed.get("coherent", False),
            "reasoning": parsed.get("reasoning", "")
        }
    else:
        return parsed.get("score", 0.5)


def llm_as_judge_batch(model: str,
                       queries: List[str],
                       responses: List[str],
                       retrieved_contexts: Optional[List[str]] = None,
                       evaluation_type: str = "support") -> Dict[str, float]:
    """
    Batch evaluation using LLM-as-judge across multiple responses.
    
    Args:
        model: LLM model identifier
        queries: List of queries
        responses: List of corresponding responses
        retrieved_contexts: Optional list of context for each response
        evaluation_type: "support", "relevance", or "coherence"
    
    Returns:
        dict: Aggregated scores
            - "individual_scores": list of scores for each response
            - "mean_score": average score
            - "min_score": minimum score
            - "max_score": maximum score
    """
    scores = []
    
    if evaluation_type == "support" and retrieved_contexts:
        for query, response, context in zip(queries, responses, retrieved_contexts):
            score = llm_as_judge_context_support(model, query, response, context)
            scores.append(score)
    elif evaluation_type == "relevance":
        for query, response in zip(queries, responses):
            score = llm_as_judge_answer_relevance(model, query, response)
            scores.append(score)
    elif evaluation_type == "coherence":
        for response in responses:
            score = llm_as_judge_coherence(model, response)
            scores.append(score)
    
    return {
        "individual_scores": scores,
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "evaluation_type": evaluation_type
    }


# =====================================================================
# 2. SEMANTIC PERPLEXITY METRICS
# =====================================================================

def semantic_perplexity(logits: List[float],
                       token_probabilities: Optional[List[float]] = None) -> float:
    """
    Compute Semantic Perplexity: Model's confidence in its own generation.
    
    Perplexity = exp(cross_entropy) where cross_entropy measures the average
    negative log probability of the generated tokens.
    
    Low perplexity (close to 1) = model is confident and generation is coherent
    High perplexity (>> 1) = model is uncertain, generation may be problematic
    
    This metric is useful for detecting when the model is unsure about its output,
    potentially avoiding hallucinations through abstention mechanisms.
    
    Args:
        logits: List of log-probabilities for each generated token
        token_probabilities: Optional alternative: direct token probabilities (0-1)
    
    Returns:
        float: Perplexity score (typically 1-1000+)
    """
    if token_probabilities is not None:
        # Convert probabilities to cross-entropy
        # cross_entropy = -sum(log(p_i)) / num_tokens
        cross_entropy = -sum(math.log(max(p, 1e-10)) for p in token_probabilities) / len(token_probabilities)
    else:
        # Use logits directly (assumed to be negative log probabilities)
        # Higher logit = lower probability = higher cross-entropy
        cross_entropy = sum(logits) / len(logits) if logits else 0.0
    
    perplexity = math.exp(cross_entropy)
    
    return perplexity


def semantic_perplexity_batch(batch_logits: List[List[float]]) -> Dict[str, float]:
    """
    Compute Semantic Perplexity across multiple generations.
    
    Args:
        batch_logits: List of logit sequences, one per generation
    
    Returns:
        dict: Perplexity statistics
            - "individual_perplexities": list of perplexity scores
            - "mean_perplexity": average perplexity
            - "min_perplexity": minimum (best confidence)
            - "max_perplexity": maximum (worst confidence)
            - "std_perplexity": standard deviation
    """
    perplexities = [semantic_perplexity(logits) for logits in batch_logits]
    
    mean_perplexity = sum(perplexities) / len(perplexities) if perplexities else 0.0
    
    # Calculate standard deviation
    if len(perplexities) > 1:
        variance = sum((p - mean_perplexity) ** 2 for p in perplexities) / (len(perplexities) - 1)
        std_perplexity = math.sqrt(variance)
    else:
        std_perplexity = 0.0
    
    return {
        "individual_perplexities": perplexities,
        "mean_perplexity": mean_perplexity,
        "min_perplexity": min(perplexities) if perplexities else 0.0,
        "max_perplexity": max(perplexities) if perplexities else 0.0,
        "std_perplexity": std_perplexity,
        "num_generations": len(perplexities)
    }


def confidence_based_abstention(logits: List[float],
                               confidence_threshold: float = 2.0) -> Dict[str, Union[float, bool]]:
    """
    Use Semantic Perplexity to decide whether system should abstain from answering.
    
    If perplexity exceeds threshold, the model is too uncertain and should refuse
    to answer to avoid hallucinations.
    
    Args:
        logits: List of log-probabilities for generated tokens
        confidence_threshold: Perplexity threshold above which to abstain
                             (lower threshold = stricter abstention)
    
    Returns:
        dict: Decision and confidence metrics
            - "should_abstain": boolean (True if perplexity > threshold)
            - "perplexity": float (actual perplexity score)
            - "confidence": float (1 - normalized_perplexity, roughly)
            - "threshold": float (used threshold)
    """
    perplexity = semantic_perplexity(logits)
    
    # Normalize confidence: higher perplexity = lower confidence
    # Using a sigmoid-like normalization: 1 / (1 + perplexity/baseline)
    baseline = 1.0
    confidence = baseline / (baseline + (perplexity / baseline))
    
    return {
        "should_abstain": perplexity > confidence_threshold,
        "perplexity": perplexity,
        "confidence": confidence,
        "threshold": confidence_threshold
    }


# =====================================================================
# 3. COMPREHENSIVE LLM-BASED EVALUATION
# =====================================================================

def comprehensive_llm_evaluation(model: str,
                                query: str,
                                response: str,
                                retrieved_context: str,
                                reference_response: Optional[str] = None,
                                logits: Optional[List[float]] = None) -> Dict:
    """
    Full LLM-based evaluation combining multiple dimensions.
    
    Evaluates response across multiple aspects:
    - Answer accuracy (against reference if provided)
    - Context support (grounding)
    - Answer relevance (does it answer the question)
    - Coherence (clarity and fluency)
    - Semantic perplexity (model confidence)
    
    Args:
        model: LLM model identifier
        query: Original user query
        response: Generated response
        retrieved_context: Retrieved context chunks
        reference_response: Optional reference/ground truth response for accuracy evaluation
        logits: Optional token logits for perplexity calculation
    
    Returns:
        dict: Comprehensive evaluation report
            - "answer_accuracy": float 0-1 (if reference_response provided)
            - "context_support": float 0-1
            - "relevance": float 0-1
            - "coherence": float 0-1
            - "perplexity": float (if logits provided)
            - "overall_score": weighted average of metrics
            - "evaluation_summary": human-readable summary
    """
    # Get individual scores
    relevance = llm_as_judge_answer_relevance(model, query, response)
    coherence = llm_as_judge_coherence(model, response)
    context_support = llm_as_judge_context_support(model, query, response, retrieved_context)
    
    result = {
        "relevance": relevance,
        "coherence": coherence,
        "context_support": context_support
    }
    
    # Add answer accuracy if reference response provided
    if reference_response is not None:
        answer_accuracy = llm_as_judge_answer_accuracy(model, query, response, reference_response)
        # Extract score if it's a dict (detailed mode)
        if isinstance(answer_accuracy, dict):
            result["answer_accuracy"] = answer_accuracy.get("score", 0.5)
            result["answer_accuracy_details"] = answer_accuracy
        else:
            result["answer_accuracy"] = answer_accuracy
    
    # Add perplexity if logits provided
    if logits is not None:
        perplexity = semantic_perplexity(logits)
        # Normalize perplexity to 0-1 (lower is better, so invert)
        confidence = 1.0 / (1.0 + perplexity)
        result["confidence"] = confidence
        result["perplexity"] = perplexity
    
    # Calculate weighted overall score
    if reference_response is not None and logits is not None:
        # All metrics available
        overall_score = (0.25 * result.get("answer_accuracy", 0.5) +
                        0.25 * context_support + 
                        0.20 * relevance + 
                        0.15 * coherence + 
                        0.15 * result.get("confidence", 0.5))
    elif reference_response is not None:
        # With answer accuracy but no perplexity
        overall_score = (0.30 * result.get("answer_accuracy", 0.5) +
                        0.30 * context_support + 
                        0.20 * relevance + 
                        0.20 * coherence)
    elif logits is not None:
        # With perplexity but no answer accuracy
        overall_score = (0.35 * context_support + 
                        0.30 * relevance + 
                        0.20 * coherence + 
                        0.15 * result.get("confidence", 0.5))
    else:
        # Only basic metrics
        overall_score = (0.40 * context_support + 
                        0.35 * relevance + 
                        0.25 * coherence)
    
    result["overall_score"] = overall_score
    
    # Generate summary
    if overall_score >= 0.8:
        summary = "Excellent: Well-grounded, relevant, coherent, and accurate response."
    elif overall_score >= 0.6:
        summary = "Good: Generally accurate and relevant, minor issues."
    elif overall_score >= 0.4:
        summary = "Fair: Some concerns with grounding, relevance, or accuracy."
    else:
        summary = "Poor: Significant concerns with factuality or relevance. Consider abstention."
    
    result["evaluation_summary"] = summary
    
    return result

