import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import sys

root = Path("../../").resolve()
parent = root / "experiments" / "metrics_comp"

sys.path.append(root.as_posix())
from evaluation.metrics import evaluate_single_qa, compute_aggregate_metrics
from src.llm.manager import LLMManager


def load_rag_results_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSONL file with RAG results."""
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def convert_context_to_chunks(context_list: List[str]) -> List[Dict[str, str]]:
    """Convert list of context strings to chunks dict format."""
    chunks = []
    for i, text in enumerate(context_list):
        chunks.append({
            "chunk_id": str(i),
            "content": text,
            "similarity": 1.0 - (0.1 * i)  # Dummy similarity decreasing with position
        })
    return chunks


def initialize_llm_judge(
    provider: str = "ollama",
    model: str = "mistral:latest",
    base_url: Optional[str] = None,
    **kwargs
):
    """
    Initialize an LLM judge model.
    
    Args:
        provider: "ollama", "openrouter", "openai", or "llamacpp"
        model: Model name/path
        base_url: Optional base URL for the provider
        **kwargs: Additional provider-specific arguments
        
    Returns:
        LLM instance for use as judge
    """
    try:
        if provider.lower() == "ollama":
            base_url = base_url or "http://localhost:11434/v1"
            judge = LLMManager.from_ollama(model, base_url=base_url)
            print(f"✓ Loaded Ollama judge: {model} from {base_url}")
            return judge.provider
            
        elif provider.lower() == "openrouter":
            api_key = kwargs.get("api_key") or kwargs.get("openrouter_api_key")
            if not api_key:
                raise ValueError("openrouter_api_key required for OpenRouter provider")
            judge = LLMManager.from_openrouter(model, api_key=api_key)
            print(f"✓ Loaded OpenRouter judge: {model}")
            return judge.provider
            
        elif provider.lower() == "openai":
            api_key = kwargs.get("api_key") or kwargs.get("openai_api_key")
            if not api_key:
                raise ValueError("openai_api_key required for OpenAI provider")
            judge = LLMManager.from_openai(model, api_key=api_key)
            print(f"✓ Loaded OpenAI judge: {model}")
            return judge.provider
            
        elif provider.lower() == "llamacpp":
            gguf_path = kwargs.get("gguf_path") or model
            judge = LLMManager.from_direct_llamacpp(gguf_path)
            print(f"✓ Loaded llama-cpp judge from: {gguf_path}")
            return judge.provider
            
        else:
            raise ValueError(f"Unknown provider: {provider}. Choose from: ollama, openrouter, openai, llamacpp")
    
    except Exception as e:
        print(f"✗ Failed to initialize LLM judge: {e}")
        return None


def compute_dataset_metrics(
    dataset_path: Path,
    output_path: Path = None,
    bert_device: str = "cpu",
    llm_judge = None
) -> Dict[str, Any]:
    """
    Compute metrics for entire RAG dataset.
    
    Args:
        dataset_path: Path to JSONL file with RAG results
        output_path: Optional path to save detailed results
        bert_device: Device for BERTScore ("cpu" or "cuda")
        llm_judge: Optional LLM instance for judge evaluation
        
    Returns:
        Dict with aggregated metrics
    """
    
    # Load dataset
    data = load_rag_results_jsonl(dataset_path)
    print(f"Loaded {len(data)} examples from {dataset_path}")
    
    all_eval_results = []
    
    for i, example in enumerate(data):
        try:
            # Extract fields from dataset
            question = example.get("question", "")
            gold_answer = example.get("answer", "")
            reference_page = example.get("reference_page", "")
            generated_answer = example.get("generated_answer", "")
            context_list = example.get("context_used", [])
            
            # Skip if missing critical fields
            if not question or not generated_answer:
                print(f"Skipping example {i}: missing question or generated_answer")
                continue
            
            # Convert context list to chunks format
            retrieved_chunks = convert_context_to_chunks(context_list)
            
            # Join context for faithfulness and other text-based metrics
            context_str = "\n\n".join(context_list)
            
            # Evaluate this QA pair
            eval_result = evaluate_single_qa(
                question=question,
                gold_answer=gold_answer,
                gold_chunk_id=str(reference_page),  # Using page as chunk_id for now
                gold_chunk_content=context_str,     # Use full context as proxy for gold content
                generated_answer=generated_answer,
                retrieved_chunks=retrieved_chunks,
                context_used=context_str,
                llm_judge=llm_judge,
                top_k=len(context_list),
                bert_device=bert_device
            )
            
            # Add original fields to result for traceability
            eval_result["question"] = question
            eval_result["gold_answer"] = gold_answer
            eval_result["generated_answer"] = generated_answer
            eval_result["reference_page"] = reference_page
            
            all_eval_results.append(eval_result)
            
            if (i + 1) % 5 == 0:
                print(f"Processed {i + 1}/{len(data)} examples")
                
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            continue
    
    # Compute aggregated metrics
    aggregate = compute_aggregate_metrics(all_eval_results)
    
    # Save detailed results if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "aggregate_metrics": aggregate,
                "detailed_results": all_eval_results[:10]  # Save first 10 for inspection
            }, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")
    
    return {
        "aggregate": aggregate,
        "total_evaluated": len(all_eval_results),
        "detailed_results": all_eval_results
    }


if __name__ == "__main__":
    # Dataset path
    dataset_path = parent / "output/qwen3.5:0.8b-context.jsonl"
    output_path = parent / "metrics_results.json"
    
    # Initialize LLM judge
    print("\n" + "="*60)
    print("INITIALIZING LLM JUDGE")
    print("="*60)
    llm_judge = initialize_llm_judge(
        provider="ollama",
        model="mistral:latest",
        base_url="http://localhost:11434/v1"
    )
    
    # Compute metrics
    print("\n" + "="*60)
    print("EVALUATING DATASET")
    print("="*60)
    results = compute_dataset_metrics(
        dataset_path=dataset_path,
        output_path=output_path,
        bert_device="cpu",
        llm_judge=llm_judge
    )
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total evaluated: {results['total_evaluated']}")
    print("\nAggregated Metrics:")
    print(json.dumps(results['aggregate'], indent=2))


