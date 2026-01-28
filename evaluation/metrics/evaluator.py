"""
Flexible and extensible evaluator framework for RAG systems.

Provides a hierarchical structure of evaluators for different metric types:
- RetrieverEvaluator: Ranking and retrieval metrics
- GenerationEvaluator: Traditional text generation metrics
- LLMEvaluator: LLM-as-judge semantic metrics
- RiskAwareEvaluator: Risk, prudence, and abstention metrics
- ComprehensiveEvaluator: Orchestrates multiple evaluators
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional, Any
import json
from dataclasses import dataclass, asdict
import logging

from trad_metrics import (
    unrankedMetrics,
    mean_reciprocal_rank,
    ndcg_at_k,
    mean_average_precision,
    exact_match,
    meteor_batch,
    risk_aware_metrics,
    latency_metrics,
    cost_metrics,
    retriever_roi,
)

from llm_metrics import (
    llm_as_judge_context_support,
    llm_as_judge_answer_relevance,
    llm_as_judge_coherence,
    llm_as_judge_batch,
    semantic_perplexity,
    semantic_perplexity_batch,
    confidence_based_abstention,
    comprehensive_llm_evaluation,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Data Classes for Configuration and Results
# =====================================================================

@dataclass
class EvaluatorConfig:
    """Configuration for evaluators."""
    name: str
    enabled: bool = True
    additional_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}


@dataclass
class EvaluationResult:
    """Standardized result structure for all evaluations."""
    evaluator_name: str
    metrics: Dict[str, Union[float, Dict]]
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# =====================================================================
# Base Evaluator Class
# =====================================================================

class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.
    
    Defines common interface and utilities for all metric evaluators.
    """

    def __init__(self, config: Optional[EvaluatorConfig] = None):
        """
        Initialize evaluator.
        
        Args:
            config: Optional EvaluatorConfig for customization
        """
        if config is None:
            config = EvaluatorConfig(name=self.__class__.__name__)
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def evaluate(self, **kwargs) -> EvaluationResult:
        """
        Perform evaluation. Must be implemented by subclasses.
        
        Returns:
            EvaluationResult with metrics and status
        """
        pass

    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate input parameters. Override in subclasses for custom validation.
        
        Returns:
            bool: True if inputs are valid
        """
        return True

    def aggregate_results(self, results: List[float]) -> Dict[str, float]:
        """
        Aggregate multiple metric results into summary statistics.
        
        Args:
            results: List of individual metric scores
        
        Returns:
            dict: Aggregated statistics
        """
        if not results:
            return {}
        
        return {
            "mean": sum(results) / len(results),
            "min": min(results),
            "max": max(results),
            "count": len(results)
        }

    def create_result(self, metrics: Dict, success: bool = True, 
                     error_message: Optional[str] = None,
                     metadata: Optional[Dict] = None) -> EvaluationResult:
        """
        Create a standardized evaluation result.
        
        Args:
            metrics: Dictionary of computed metrics
            success: Whether evaluation succeeded
            error_message: Error message if failed
            metadata: Additional metadata about evaluation
        
        Returns:
            EvaluationResult object
        """
        return EvaluationResult(
            evaluator_name=self.config.name,
            metrics=metrics,
            success=success,
            error_message=error_message,
            metadata=metadata or {}
        )


# =====================================================================
# Retriever Evaluator
# =====================================================================

class RetrieverEvaluator(BaseEvaluator):
    """
    Evaluates retrieval quality using rank-based and non-rank-based metrics.
    
    Metrics:
    - Non-rank-based: accuracy, precision, recall, f1
    - Rank-based: MRR, NDCG@k, MAP@k
    """

    def __init__(self, config: Optional[EvaluatorConfig] = None):
        """
        Initialize retriever evaluator.
        
        Args:
            config: EvaluatorConfig with k_values in additional_params
        """
        if config is None:
            config = EvaluatorConfig(
                name="RetrieverEvaluator",
                additional_params={"k_values": [5, 10, 20]}
            )
        super().__init__(config)

    def evaluate(self, 
                predictions_list: List[List[int]],
                ground_truth_list: List[List[int]],
                k_values: Optional[List[int]] = None,
                include_non_ranked: bool = True) -> EvaluationResult:
        """
        Evaluate retriever performance.
        
        Args:
            predictions_list: List of ranked predictions for each query
            ground_truth_list: List of ground truth relevant items for each query
            k_values: List of k values for rank-based metrics (default from config)
            include_non_ranked: Whether to include non-rank-based metrics
        
        Returns:
            EvaluationResult with retriever metrics
        """
        try:
            # Validate inputs
            if not self.validate_inputs(
                predictions_list=predictions_list,
                ground_truth_list=ground_truth_list
            ):
                return self.create_result({}, success=False, 
                                        error_message="Invalid inputs")
            
            # Use config k_values if not provided
            if k_values is None:
                k_values = self.config.additional_params.get("k_values", [5, 10, 20])
            
            metrics = {}
            
            # Non-rank-based metrics
            if include_non_ranked and predictions_list and ground_truth_list:
                non_ranked = unrankedMetrics(
                    predictions_list[0], ground_truth_list[0]
                )
                metrics["non_rank_based"] = non_ranked
            
            # Rank-based metrics
            metrics["rank_based"] = {
                "mrr": mean_reciprocal_rank(predictions_list, ground_truth_list)
            }
            
            for k in k_values:
                metrics["rank_based"][f"ndcg@{k}"] = ndcg_at_k(
                    predictions_list, ground_truth_list, k
                )
                metrics["rank_based"][f"map@{k}"] = mean_average_precision(
                    predictions_list, ground_truth_list, k
                )
            
            return self.create_result(
                metrics,
                metadata={"k_values": k_values, "num_queries": len(predictions_list)}
            )
        
        except Exception as e:
            self.logger.error(f"Error in retriever evaluation: {str(e)}")
            return self.create_result({}, success=False, error_message=str(e))

    def validate_inputs(self, predictions_list: List, 
                       ground_truth_list: List) -> bool:
        """Validate retriever inputs."""
        if not predictions_list or not ground_truth_list:
            return False
        if len(predictions_list) != len(ground_truth_list):
            return False
        return True


# =====================================================================
# Generation Evaluator
# =====================================================================

class GenerationEvaluator(BaseEvaluator):
    """
    Evaluates text generation quality using traditional metrics.
    
    Metrics:
    - Exact Match (EM)
    - METEOR
    """

    def __init__(self, config: Optional[EvaluatorConfig] = None):
        if config is None:
            config = EvaluatorConfig(name="GenerationEvaluator")
        super().__init__(config)

    def evaluate(self,
                predictions: List[str],
                references: List[str]) -> EvaluationResult:
        """
        Evaluate generation quality.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
        
        Returns:
            EvaluationResult with generation metrics
        """
        try:
            if not self.validate_inputs(predictions=predictions, 
                                       references=references):
                return self.create_result({}, success=False,
                                        error_message="Invalid inputs")
            
            metrics = {
                "exact_match": exact_match(predictions, references),
                "meteor": meteor_batch(predictions, references)
            }
            
            return self.create_result(
                metrics,
                metadata={"num_samples": len(predictions)}
            )
        
        except Exception as e:
            self.logger.error(f"Error in generation evaluation: {str(e)}")
            return self.create_result({}, success=False, error_message=str(e))

    def validate_inputs(self, predictions: List, 
                       references: List) -> bool:
        """Validate generation inputs."""
        if not predictions or not references:
            return False
        if len(predictions) != len(references):
            return False
        return all(isinstance(p, str) and isinstance(r, str) 
                  for p, r in zip(predictions, references))


# =====================================================================
# LLM Evaluator
# =====================================================================

class LLMEvaluator(BaseEvaluator):
    """
    Evaluates responses using LLM-as-judge semantic metrics.
    
    Metrics:
    - Context support (grounding)
    - Answer relevance
    - Coherence
    - Semantic perplexity
    - Comprehensive evaluation combining all aspects
    """

    def __init__(self, model: str, config: Optional[EvaluatorConfig] = None):
        """
        Initialize LLM evaluator.
        
        Args:
            model: LLM model identifier (e.g., "gpt-4", "claude-3")
            config: Optional EvaluatorConfig
        """
        if config is None:
            config = EvaluatorConfig(
                name="LLMEvaluator",
                additional_params={"model": model}
            )
        else:
            config.additional_params["model"] = model
        
        super().__init__(config)
        self.model = model

    def evaluate_context_support(self,
                                query: str,
                                response: str,
                                context: str,
                                detailed: bool = False) -> Union[float, Dict]:
        """
        Evaluate if response is supported by retrieved context.
        
        Args:
            query: User query
            response: Generated response
            context: Retrieved context
            detailed: Return full reasoning
        
        Returns:
            Score 0-1 or detailed dict
        """
        return llm_as_judge_context_support(
            self.model, query, response, context, detailed
        )

    def evaluate_relevance(self,
                          query: str,
                          response: str,
                          detailed: bool = False) -> Union[float, Dict]:
        """
        Evaluate if response answers the query.
        
        Args:
            query: User query
            response: Generated response
            detailed: Return full reasoning
        
        Returns:
            Score 0-1 or detailed dict
        """
        return llm_as_judge_answer_relevance(self.model, query, response, detailed)

    def evaluate_coherence(self,
                          response: str,
                          detailed: bool = False) -> Union[float, Dict]:
        """
        Evaluate coherence and fluency of response.
        
        Args:
            response: Generated response
            detailed: Return full reasoning
        
        Returns:
            Score 0-1 or detailed dict
        """
        return llm_as_judge_coherence(self.model, response, detailed)

    def evaluate_comprehensive(self,
                              query: str,
                              response: str,
                              context: str,
                              logits: Optional[List[float]] = None) -> Dict:
        """
        Comprehensive evaluation across all dimensions.
        
        Args:
            query: User query
            response: Generated response
            context: Retrieved context
            logits: Optional token logits for perplexity
        
        Returns:
            Comprehensive evaluation report
        """
        return comprehensive_llm_evaluation(self.model, query, response, 
                                           context, logits)

    def evaluate_batch(self,
                      queries: List[str],
                      responses: List[str],
                      contexts: Optional[List[str]] = None,
                      evaluation_type: str = "relevance") -> EvaluationResult:
        """
        Batch evaluation across multiple responses.
        
        Args:
            queries: List of queries
            responses: List of responses
            contexts: Optional list of contexts (required for 'support' type)
            evaluation_type: "support", "relevance", or "coherence"
        
        Returns:
            EvaluationResult with batch metrics
        """
        try:
            if evaluation_type == "support" and contexts is None:
                return self.create_result({}, success=False,
                                        error_message="Contexts required for support evaluation")
            
            if not self.validate_inputs(queries=queries, responses=responses):
                return self.create_result({}, success=False,
                                        error_message="Invalid inputs")
            
            metrics = llm_as_judge_batch(
                self.model, queries, responses, contexts, evaluation_type
            )
            
            return self.create_result(
                metrics,
                metadata={"evaluation_type": evaluation_type, 
                         "num_samples": len(responses)}
            )
        
        except Exception as e:
            self.logger.error(f"Error in LLM batch evaluation: {str(e)}")
            return self.create_result({}, success=False, error_message=str(e))

    def evaluate(self, **kwargs) -> EvaluationResult:
        """
        Generic evaluate method. Routes to appropriate evaluation based on args.
        
        Supports: evaluate_comprehensive, evaluate_batch
        """
        if "query" in kwargs and "response" in kwargs and "context" in kwargs:
            # Comprehensive evaluation
            result = self.evaluate_comprehensive(
                kwargs["query"], kwargs["response"], kwargs["context"],
                kwargs.get("logits")
            )
            return self.create_result(result)
        elif "queries" in kwargs and "responses" in kwargs:
            # Batch evaluation
            return self.evaluate_batch(
                kwargs["queries"], kwargs["responses"],
                kwargs.get("contexts"),
                kwargs.get("evaluation_type", "relevance")
            )
        else:
            return self.create_result({}, success=False,
                                    error_message="Invalid arguments for evaluation")

    def validate_inputs(self, queries: List = None, 
                       responses: List = None, **kwargs) -> bool:
        """Validate LLM evaluator inputs."""
        if queries is not None and responses is not None:
            if len(queries) != len(responses):
                return False
        return True


# =====================================================================
# Semantic Perplexity Evaluator
# =====================================================================

class PerplexityEvaluator(BaseEvaluator):
    """
    Evaluates model confidence using semantic perplexity.
    
    Metrics:
    - Semantic perplexity
    - Model confidence scores
    - Abstention decisions based on confidence thresholds
    """

    def __init__(self, config: Optional[EvaluatorConfig] = None):
        if config is None:
            config = EvaluatorConfig(
                name="PerplexityEvaluator",
                additional_params={"confidence_threshold": 2.0}
            )
        super().__init__(config)

    def evaluate(self,
                logits_list: List[List[float]]) -> EvaluationResult:
        """
        Evaluate perplexity across multiple generations.
        
        Args:
            logits_list: List of logit sequences for each generation
        
        Returns:
            EvaluationResult with perplexity metrics
        """
        try:
            if not self.validate_inputs(logits_list=logits_list):
                return self.create_result({}, success=False,
                                        error_message="Invalid inputs")
            
            metrics = semantic_perplexity_batch(logits_list)
            
            return self.create_result(
                metrics,
                metadata={"num_generations": len(logits_list)}
            )
        
        except Exception as e:
            self.logger.error(f"Error in perplexity evaluation: {str(e)}")
            return self.create_result({}, success=False, error_message=str(e))

    def evaluate_abstention(self,
                          logits: List[float],
                          confidence_threshold: Optional[float] = None) -> Dict:
        """
        Determine if system should abstain based on confidence.
        
        Args:
            logits: Token logits for generated response
            confidence_threshold: Optional threshold (uses config default if None)
        
        Returns:
            Abstention decision and confidence metrics
        """
        if confidence_threshold is None:
            confidence_threshold = self.config.additional_params.get(
                "confidence_threshold", 2.0
            )
        
        return confidence_based_abstention(logits, confidence_threshold)

    def validate_inputs(self, logits_list: List) -> bool:
        """Validate perplexity inputs."""
        if not logits_list:
            return False
        return all(isinstance(logits, list) and logits for logits in logits_list)


# =====================================================================
# Risk-Aware Evaluator
# =====================================================================

class RiskAwareEvaluator(BaseEvaluator):
    """
    Evaluates risk metrics for systems that can abstain.
    
    Metrics:
    - Risk (hallucination rate)
    - Prudence (unanswerable detection rate)
    - Alignment (overall correctness)
    - Coverage (answer rate)
    """

    def __init__(self, config: Optional[EvaluatorConfig] = None):
        if config is None:
            config = EvaluatorConfig(name="RiskAwareEvaluator")
        super().__init__(config)

    def evaluate(self,
                answerable_kept: int,
                unanswerable_kept: int,
                unanswerable_discarded: int,
                answerable_discarded: int) -> EvaluationResult:
        """
        Evaluate risk-aware metrics.
        
        Args:
            answerable_kept: Count of correct answers provided
            unanswerable_kept: Count of hallucinations
            unanswerable_discarded: Count of correctly rejected questions
            answerable_discarded: Count of missed opportunities
        
        Returns:
            EvaluationResult with risk metrics
        """
        try:
            metrics = risk_aware_metrics(
                answerable_kept, unanswerable_kept,
                unanswerable_discarded, answerable_discarded
            )
            
            total = answerable_kept + unanswerable_kept + unanswerable_discarded + answerable_discarded
            
            return self.create_result(
                metrics,
                metadata={
                    "answerable_kept": answerable_kept,
                    "unanswerable_kept": unanswerable_kept,
                    "unanswerable_discarded": unanswerable_discarded,
                    "answerable_discarded": answerable_discarded,
                    "total_samples": total
                }
            )
        
        except Exception as e:
            self.logger.error(f"Error in risk-aware evaluation: {str(e)}")
            return self.create_result({}, success=False, error_message=str(e))


# =====================================================================
# Efficiency Evaluator
# =====================================================================

class EfficiencyEvaluator(BaseEvaluator):
    """
    Evaluates efficiency metrics for RAG systems.
    
    Metrics:
    - Latency (TTFT, total time)
    - Cost (input/output costs)
    - Retriever ROI
    """

    def __init__(self, config: Optional[EvaluatorConfig] = None):
        if config is None:
            config = EvaluatorConfig(name="EfficiencyEvaluator")
        super().__init__(config)

    def evaluate_latency(self,
                        ttft_ms: float,
                        total_latency_ms: float) -> EvaluationResult:
        """
        Evaluate latency metrics.
        
        Args:
            ttft_ms: Time to First Token in milliseconds
            total_latency_ms: Total generation time in milliseconds
        
        Returns:
            EvaluationResult with latency metrics
        """
        try:
            metrics = latency_metrics(ttft_ms, total_latency_ms)
            return self.create_result(metrics)
        except Exception as e:
            self.logger.error(f"Error in latency evaluation: {str(e)}")
            return self.create_result({}, success=False, error_message=str(e))

    def evaluate_cost(self,
                     input_tokens: int,
                     output_tokens: int,
                     cost_per_input_token: float,
                     cost_per_output_token: float) -> EvaluationResult:
        """
        Evaluate cost metrics.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost_per_input_token: Cost per input token
            cost_per_output_token: Cost per output token
        
        Returns:
            EvaluationResult with cost metrics
        """
        try:
            metrics = cost_metrics(
                input_tokens, output_tokens,
                cost_per_input_token, cost_per_output_token
            )
            return self.create_result(metrics)
        except Exception as e:
            self.logger.error(f"Error in cost evaluation: {str(e)}")
            return self.create_result({}, success=False, error_message=str(e))

    def evaluate_retriever_roi(self,
                              retriever_cost: float,
                              context_tokens_per_query: int,
                              llm_cost_per_input_token: float,
                              num_queries: int) -> EvaluationResult:
        """
        Evaluate retriever ROI.
        
        Args:
            retriever_cost: Cost per retrieval query
            context_tokens_per_query: Average context tokens retrieved
            llm_cost_per_input_token: LLM cost per token
            num_queries: Number of queries
        
        Returns:
            EvaluationResult with ROI metrics
        """
        try:
            metrics = retriever_roi(
                retriever_cost, context_tokens_per_query,
                llm_cost_per_input_token, num_queries
            )
            return self.create_result(metrics)
        except Exception as e:
            self.logger.error(f"Error in ROI evaluation: {str(e)}")
            return self.create_result({}, success=False, error_message=str(e))

    def evaluate(self, **kwargs) -> EvaluationResult:
        """Generic evaluate method routing to appropriate efficiency metric."""
        if "ttft_ms" in kwargs and "total_latency_ms" in kwargs:
            return self.evaluate_latency(kwargs["ttft_ms"], kwargs["total_latency_ms"])
        elif all(k in kwargs for k in ["input_tokens", "output_tokens"]):
            return self.evaluate_cost(
                kwargs["input_tokens"],
                kwargs["output_tokens"],
                kwargs.get("cost_per_input_token", 0),
                kwargs.get("cost_per_output_token", 0)
            )
        else:
            return self.create_result({}, success=False,
                                    error_message="Invalid arguments")


# =====================================================================
# Comprehensive Evaluator
# =====================================================================

class ComprehensiveEvaluator:
    """
    Orchestrates multiple specialized evaluators into a unified framework.
    
    Manages evaluation across all dimensions (retrieval, generation, 
    LLM-as-judge, risk, efficiency) and provides unified reporting.
    """

    def __init__(self, config: Optional[Dict[str, EvaluatorConfig]] = None,
                llm_model: Optional[str] = None):
        """
        Initialize comprehensive evaluator.
        
        Args:
            config: Optional dict of evaluator-specific configurations
            llm_model: Model identifier for LLM evaluator
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize individual evaluators
        retriever_config = config.get("retriever") if config else None
        self.retriever_evaluator = RetrieverEvaluator(retriever_config)
        
        generation_config = config.get("generation") if config else None
        self.generation_evaluator = GenerationEvaluator(generation_config)
        
        llm_config = config.get("llm") if config else None
        self.llm_evaluator = LLMEvaluator(
            llm_model or "gpt-4", llm_config
        ) if llm_model else None
        
        perplexity_config = config.get("perplexity") if config else None
        self.perplexity_evaluator = PerplexityEvaluator(perplexity_config)
        
        risk_config = config.get("risk") if config else None
        self.risk_evaluator = RiskAwareEvaluator(risk_config)
        
        efficiency_config = config.get("efficiency") if config else None
        self.efficiency_evaluator = EfficiencyEvaluator(efficiency_config)

    def evaluate_full_pipeline(self,
                              query: str,
                              response: str,
                              context: str,
                              predictions: Optional[List[List[int]]] = None,
                              ground_truth: Optional[List[List[int]]] = None,
                              references: Optional[List[str]] = None,
                              logits: Optional[List[float]] = None,
                              **kwargs) -> Dict[str, Any]:
        """
        Comprehensive evaluation of full RAG pipeline.
        
        Args:
            query: User query
            response: Generated response
            context: Retrieved context
            predictions: Optional ranked retrieval predictions
            ground_truth: Optional ground truth for retrieval
            references: Optional reference answers
            logits: Optional token logits for perplexity
            **kwargs: Additional parameters for specific evaluators
        
        Returns:
            Unified evaluation report with all metrics
        """
        report = {
            "timestamp": None,
            "results": {},
            "summary": {},
            "success": True
        }
        
        try:
            # Import datetime here to avoid top-level dependency
            from datetime import datetime
            report["timestamp"] = datetime.now().isoformat()
            
            # Retrieval evaluation
            if predictions is not None and ground_truth is not None:
                k_values = kwargs.get("k_values", [5, 10, 20])
                result = self.retriever_evaluator.evaluate(
                    predictions, ground_truth, k_values
                )
                report["results"]["retrieval"] = asdict(result)
            
            # Generation evaluation
            if references is not None:
                predictions_text = [response] if isinstance(response, str) else response
                references_text = references if isinstance(references, list) else [references]
                result = self.generation_evaluator.evaluate(
                    predictions_text, references_text
                )
                report["results"]["generation"] = asdict(result)
            
            # LLM evaluation
            if self.llm_evaluator:
                result = self.llm_evaluator.evaluate(
                    query=query,
                    response=response,
                    context=context,
                    logits=logits
                )
                report["results"]["llm_judge"] = asdict(result)
            
            # Perplexity evaluation
            if logits is not None:
                result = self.perplexity_evaluator.evaluate([logits])
                report["results"]["perplexity"] = asdict(result)
            
            # Risk-aware evaluation
            if all(k in kwargs for k in ["answerable_kept", "unanswerable_kept"]):
                result = self.risk_evaluator.evaluate(
                    kwargs["answerable_kept"],
                    kwargs["unanswerable_kept"],
                    kwargs["unanswerable_discarded"],
                    kwargs["answerable_discarded"]
                )
                report["results"]["risk"] = asdict(result)
            
            # Efficiency evaluation
            if "ttft_ms" in kwargs and "total_latency_ms" in kwargs:
                result = self.efficiency_evaluator.evaluate_latency(
                    kwargs["ttft_ms"],
                    kwargs["total_latency_ms"]
                )
                report["results"]["latency"] = asdict(result)
            
            if all(k in kwargs for k in ["input_tokens", "output_tokens"]):
                result = self.efficiency_evaluator.evaluate_cost(
                    kwargs["input_tokens"],
                    kwargs["output_tokens"],
                    kwargs.get("cost_per_input_token", 0),
                    kwargs.get("cost_per_output_token", 0)
                )
                report["results"]["cost"] = asdict(result)
            
            # Generate summary
            report["summary"] = self._generate_summary(report["results"])
        
        except Exception as e:
            self.logger.error(f"Error in comprehensive evaluation: {str(e)}")
            report["success"] = False
            report["error"] = str(e)
        
        return report

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary of evaluation results.
        
        Args:
            results: Dictionary of individual evaluation results
        
        Returns:
            Summary report with key insights
        """
        summary = {
            "total_evaluators_run": len(results),
            "successful_evaluations": sum(
                1 for r in results.values()
                if isinstance(r, dict) and r.get("success", True)
            ),
            "key_metrics": {}
        }
        
        # Extract key metrics from results
        if "llm_judge" in results and "metrics" in results["llm_judge"]:
            metrics = results["llm_judge"]["metrics"]
            if "overall_score" in metrics:
                summary["key_metrics"]["overall_quality"] = metrics["overall_score"]
        
        if "retrieval" in results and "metrics" in results["retrieval"]:
            metrics = results["retrieval"]["metrics"]
            if "rank_based" in metrics:
                summary["key_metrics"]["retrieval_quality"] = metrics["rank_based"].get("mrr", 0)
        
        if "perplexity" in results and "metrics" in results["perplexity"]:
            metrics = results["perplexity"]["metrics"]
            if "mean_perplexity" in metrics:
                summary["key_metrics"]["confidence"] = 1.0 / (1.0 + metrics["mean_perplexity"])
        
        return summary

    def to_json(self, report: Dict[str, Any], pretty: bool = True) -> str:
        """
        Serialize evaluation report to JSON.
        
        Args:
            report: Evaluation report dictionary
            pretty: Whether to pretty-print JSON
        
        Returns:
            JSON string representation
        """
        try:
            return json.dumps(report, indent=2 if pretty else None, default=str)
        except TypeError:
            return json.dumps(report, default=str)


# =====================================================================
# Example Usage
# =====================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example: Initialization
    evaluator = ComprehensiveEvaluator(llm_model="gpt-4")
    
    # Example: Full pipeline evaluation
    report = evaluator.evaluate_full_pipeline(
        query="What is machine learning?",
        response="Machine learning is a subset of AI...",
        context="Machine learning involves algorithms...",
        references=["Machine learning is a field of AI..."],
        ttft_ms=150.0,
        total_latency_ms=1200.0
    )
    
    print(evaluator.to_json(report))
