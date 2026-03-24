#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def display_entry(entry, index):
    """Display a single entry with nice formatting."""
    print(f"\n{'='*80}")
    print(f"Entry {index + 1}")
    print(f"{'='*80}\n")
    
    print("QUESTION:")
    print(f"  {entry.get('question', 'N/A')}\n")
    
    print("RETRIEVED CONTEXT:")
    context = entry.get('context_used', [])
    if isinstance(context, list) and context:
        for i, ctx in enumerate(context[:3], 1):
            if isinstance(ctx, str):
                text = ctx[:300] + "..." if len(ctx) > 300 else ctx
                print(f"  [{i}] {text}")
        if len(context) > 3:
            print(f"  ... and {len(context) - 3} more context items\n")
        else:
            print()
    else:
        print("  N/A\n")
    
    print("REFERENCE ANSWER:")
    print(f"  {entry.get('answer', 'N/A')}\n")
    
    print("GENERATED ANSWER:")
    answer = entry.get('generated_answer', 'N/A')
    if isinstance(answer, str) and len(answer) > 500:
        lines = answer.split('\n')
        for line in lines[:10]:
            print(f"  {line}")
        print(f"  ...\n  [Content truncated, total length: {len(answer)} chars]\n")
    else:
        print(f"  {answer}\n")


def get_rating(prompt_text, allow_none=True):
    """Get a rating from user (0-10 or 'none')."""
    while True:
        response = input(prompt_text).strip()
        
        if not response:
            if allow_none:
                return "none"
            else:
                print("  Please enter a value (0-10) or press Enter for 'none'")
                continue
        
        if response.lower() == "none":
            return "none"
        
        try:
            rating = int(response)
            if 0 <= rating <= 10:
                return rating
            else:
                print("  Please enter a number between 0 and 10")
        except ValueError:
            print("  Invalid input. Enter 0-10, 'none', or press Enter")


def get_evaluations():
    """Prompt user for all evaluation metrics."""
    print("\n" + "-" * 80)
    print("EVALUATION")
    print("-" * 80 + "\n")
    
    evaluations = {}
    
    print("1. RETRIEVER (Recherche d'information)\n")
    
    print("   Context Recall (Rappel du contexte):")
    print("   Capacité à extraire toutes les informations nécessaires pour répondre.")
    evaluations["context_recall"] = get_rating("   Rate (0-10, or press Enter for 'none'): ")
    
    print("\n   Context Precision (Précision du contexte):")
    print("   Capacité à ne remonter que des éléments utiles et pertinents.")
    evaluations["context_precision"] = get_rating("   Rate (0-10, or press Enter for 'none'): ")
    
    print("\n2. GENERATOR (Génération de la réponse)\n")
    
    print("   Faithfulness / Groundedness (Fidélité):")
    print("   Capacité à formuler une réponse issue uniquement du contexte fourni (mesure directe des hallucinations).")
    evaluations["faithfulness"] = get_rating("   Rate (0-10, or press Enter for 'none'): ")
    
    print("\n   Linguistic Quality (Qualité linguistique):")
    print("   Clarté, fluidité et respect des consignes de formatage.")
    evaluations["linguistic_quality"] = get_rating("   Rate (0-10, or press Enter for 'none'): ")
    
    print("\n3. SYSTÈME COMPLET (End-to-End)\n")
    
    print("   Answer Relevance (Pertinence de la réponse):")
    print("   Capacité à répondre directement à la question de l'utilisateur, sans digression ni manque.")
    evaluations["answer_relevance"] = get_rating("   Rate (0-10, or press Enter for 'none'): ")
    
    print("\n   Robustness to Off-Topic (Robustesse au hors-sujet):")
    print("   Capacité du système à refuser de répondre lorsque le contexte ne contient pas l'information requise.")
    evaluations["robustness"] = get_rating("   Rate (0-10, or press Enter for 'none'): ")
    
    return evaluations


def evaluate_file(input_file, output_file=None):
    """Process JSONL file and add evaluations."""
    if output_file is None:
        output_file = input_file
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: File {input_file} not found")
        sys.exit(1)
    
    entries = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    print(f"Loaded {len(entries)} entries from {input_file}\n")
    
    entry_index = 0
    for i, entry in enumerate(entries):
        level = entry.get('level', 0)
        if level == 0:
            continue
        
        display_entry(entry, entry_index)
        evaluations = get_evaluations()
        entry["evaluation"] = evaluations
        
        remaining_entries = sum(1 for e in entries[i+1:] if e.get('level', 0) != 0)
        if remaining_entries > 0:
            proceed = input("\n\nPress Enter to continue to next entry (or 'q' to quit): ").strip().lower()
            if proceed == 'q':
                break
        
        entry_index += 1
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\n\nResults saved to {output_file}")
    print(f"Evaluated {len([e for e in entries if 'evaluation' in e])} entries")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_rag.py <input_jsonl_file> [output_jsonl_file]")
        print("\nExample:")
        print("  python evaluate_rag.py data.jsonl")
        print("  python evaluate_rag.py data.jsonl evaluated_data.jsonl")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    evaluate_file(input_file, output_file)
