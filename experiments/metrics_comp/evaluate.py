import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from evaluation.metrics import evaluate_single_qa, compute_aggregate_metrics


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


def main():
    """Example usage."""
    # Dataset path
    dataset_path = Path("experiments/metrics_comp/output/qwen3.5:0.8b-context.jsonl")
    output_path = Path("experiments/metrics_comp/metrics_results.json")
    
    # Compute metrics
    results = compute_dataset_metrics(
        dataset_path=dataset_path,
        output_path=output_path,
        bert_device="cpu"  # Use "cuda" if available
    )
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total evaluated: {results['total_evaluated']}")
    print("\nAggregated Metrics:")
    print(json.dumps(results['aggregate'], indent=2))


if __name__ == "__main__":
    main()