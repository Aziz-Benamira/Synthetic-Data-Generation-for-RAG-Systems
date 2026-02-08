"""
CHALLENGING TEST - Ambiguous Content Demo
==========================================

This test creates DELIBERATELY AMBIGUOUS chunks to challenge the system:
- Ambiguous references ("it", "this", "that")
- Incomplete explanations
- Contradictory information
- Missing context
- Vague terminology

Goal: Force the critic to reject bad QAs and trigger retry loops.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.ollama_client import create_ollama_client, OLLAMA_MODELS
from src.chunking.semantic_chunker import SemanticChunk
from src.agents.question_generator import QuestionGenerator
from src.agents.answer_generator import AnswerGenerator
from src.agents.critic_agent import CriticAgent

# AMBIGUOUS CHUNKS - Designed to produce bad QAs
CHALLENGING_CHUNKS = [
    {
        "id": "ambiguous_1",
        "content": """
        Le système fonctionne de manière optimale. Il peut atteindre 95% de performance.
        Cette valeur dépend des conditions externes. Dans certains cas, on observe 82%.
        Les facteurs incluent la température et d'autres paramètres.
        """,
        "title": "Performance du système",
        "issues": ["Vague references (Il, Cette valeur)", "Numbers without context", "Missing causality"]
    },
    {
        "id": "ambiguous_2", 
        "content": """
        La méthode proposée résout le problème. Elle est plus efficace que les approches 
        précédentes. Cependant, il faut considérer les limitations. Dans notre étude,
        nous avons observé des résultats intéressants qui confirment notre hypothèse.
        """,
        "title": "Évaluation de la méthode",
        "issues": ["No concrete problem mentioned", "No concrete method described", "No actual results"]
    },
    {
        "id": "ambiguous_3",
        "content": """
        Le taux d'erreur est de 0.15 dans le cas général. Pour les cas spécifiques,
        il varie entre 0.08 et 0.32. Ces valeurs sont importantes pour l'analyse.
        On note également que 67% des échantillons répondent aux critères.
        """,
        "title": "Analyse des erreurs", 
        "issues": ["Numbers without chunk context", "No explanation of 'cas général'", "Vague 'critères'"]
    },
    {
        "id": "incomplete_1",
        "content": """
        L'algorithme commence par initialiser les paramètres. Ensuite, il itère sur
        les données. Le processus continue jusqu'à convergence. C'est simple et efficace.
        """,
        "title": "Description de l'algorithme",
        "issues": ["No actual algorithm steps", "How questions will fail", "Oral language 'C'est'"]
    },
    {
        "id": "contradictory_1",
        "content": """
        La méthode A est supérieure à la méthode B en termes de précision. Cependant,
        les résultats montrent que la méthode B obtient de meilleurs scores. 
        Il faut donc choisir avec précaution. Les deux approches ont leurs avantages.
        """,
        "title": "Comparaison des méthodes",
        "issues": ["Contradiction: A > B but B has better scores", "Why questions will expose this"]
    }
]

def print_section(title, char="="):
    print(f"\n{char * 90}")
    print(f"  {title}")
    print(f"{char * 90}\n")

def run_challenging_test(use_seif_critic=False):
    """Run test with challenging chunks"""
    
    version = "SEIF" if use_seif_critic else "CURRENT"
    print_section(f"CHALLENGING TEST - {version} VERSION")
    
    # Setup
    if use_seif_critic:
        print("📝 Temporarily swapping to Seif's critic...")
        import shutil
        critic_original = Path("src/agents/critic_agent.py")
        critic_seif = Path("seif_changes_review/critic_agent_seif.py")
        critic_backup = Path("src/agents/critic_agent_backup_challenge.py")
        
        shutil.copy2(critic_original, critic_backup)
        shutil.copy2(critic_seif, critic_original)
        
        # Clear cache
        if 'critic_agent' in sys.modules:
            del sys.modules['critic_agent']
        print("✅ Using Seif's critic\n")
    
    try:
        # Import (fresh if swapped)
        from src.agents.critic_agent import CriticAgent
        
        # Create clients
        print("🔌 Connecting to Ollama...")
        client = create_ollama_client()
        print("✅ Connected!\n")
        
        # Create agents
        question_gen = QuestionGenerator(
            llm_client=client,
            model_name=OLLAMA_MODELS["generator"],
            temperature=0.8,  # Higher temp for variety
            language="fr"
        )
        
        answer_gen = AnswerGenerator(
            llm_client=client,
            model_name=OLLAMA_MODELS["generator"],
            temperature=0.7,
            language="fr"
        )
        
        critic = CriticAgent(
            llm_client=client,
            model_name=OLLAMA_MODELS["critic"],
            temperature=0.3,  # Lower temp for consistency
            language="fr"
        )
        
        results = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "chunks": []
        }
        
        total_qa = 0
        total_passed = 0
        total_rejected = 0
        total_retries = 0
        
        # Process each challenging chunk
        for i, chunk_data in enumerate(CHALLENGING_CHUNKS, 1):
            print(f"{'─' * 90}")
            print(f"CHUNK {i}/{len(CHALLENGING_CHUNKS)}: {chunk_data['id']}")
            print(f"Title: {chunk_data['title']}")
            print(f"Expected Issues: {', '.join(chunk_data['issues'])}")
            print(f"{'─' * 90}\n")
            
            # Create semantic chunk
            chunk = SemanticChunk(
                chunk_id=chunk_data['id'],
                content=chunk_data['content'],
                semantic_type="text",
                page_range=(1, 1),
                chapter_title="Test Chapter",
                section_title=chunk_data['title']
            )
            
            chunk_result = {
                "chunk_id": chunk_data['id'],
                "title": chunk_data['title'],
                "expected_issues": chunk_data['issues'],
                "qa_pairs": []
            }
            
            # Generate 2 questions
            print(f"   📝 Generating questions...")
            questions = question_gen.generate_from_chunk(chunk, num_questions=2)
            print(f"      → {len(questions)} questions generated\n")
            
            for q_idx, question in enumerate(questions, 1):
                print(f"   Q{q_idx}: {question.question}")
                
                # Generate answer
                print(f"   💬 Generating answer...")
                answer = answer_gen.generate_answer(question, chunk)
                print(f"      → Answer: {answer[:100]}...\n")
                
                # Create QA pair
                from src.agents.answer_generator import QAPair
                qa_pair = QAPair.from_question_and_answer(question, answer)
                total_qa += 1
                
                # Evaluate with critic (with retry loop)
                print(f"   🔍 Critic evaluation...")
                max_retries = 2
                attempt = 0
                final_eval = None
                
                while attempt <= max_retries:
                    evaluation = critic.evaluate(qa_pair, chunk)
                    attempt += 1
                    
                    if evaluation.decision.value == "pass":
                        total_passed += 1
                        if attempt > 1:
                            print(f"      ✅ PASSED (after {attempt-1} retries)")
                            total_retries += (attempt - 1)
                        else:
                            print(f"      ✅ PASSED")
                        final_eval = evaluation
                        break
                    else:
                        if attempt <= max_retries:
                            print(f"      🔄 RETRY {attempt}/{max_retries}")
                            print(f"         Reasons: {', '.join(evaluation.failed_criteria)}")
                            
                            # Format feedback
                            feedback = critic.format_feedback_for_retry(evaluation)
                            print(f"         Feedback: {feedback[:80]}...")
                            
                            # Regenerate
                            new_question = question_gen.regenerate_with_feedback(
                                chunk, qa_pair.question, feedback
                            )
                            if new_question:
                                new_answer = answer_gen.regenerate_with_feedback(
                                    new_question, chunk, answer, feedback
                                )
                                qa_pair = QAPair.from_question_and_answer(new_question, new_answer)
                                print(f"         New Q: {qa_pair.question[:60]}...")
                            else:
                                break
                        else:
                            total_rejected += 1
                            print(f"      ❌ REJECTED (after {max_retries} retries)")
                            print(f"         Final reasons: {', '.join(evaluation.failed_criteria)}")
                            final_eval = evaluation
                
                # Store result
                if final_eval:
                    chunk_result["qa_pairs"].append({
                        "question": qa_pair.question,
                        "answer": qa_pair.answer,
                        "passed": final_eval.decision.value == "pass",
                        "score": final_eval.overall_score,
                        "failed_criteria": final_eval.failed_criteria,
                        "retries": attempt - 1
                    })
                
                print()
            
            results["chunks"].append(chunk_result)
            print()
        
        # Summary
        print_section("RESULTS SUMMARY")
        
        rejection_rate = (total_rejected / total_qa * 100) if total_qa > 0 else 0
        
        print(f"📊 QA PAIRS:")
        print(f"   Total generated: {total_qa}")
        print(f"   ✅ PASSED: {total_passed} ({total_passed/total_qa*100:.1f}%)")
        print(f"   ❌ REJECTED: {total_rejected} ({rejection_rate:.1f}%)")
        print()
        
        print(f"🔄 RETRY LOOPS:")
        print(f"   Total retries triggered: {total_retries}")
        print(f"   Average retries per QA: {total_retries/total_qa:.2f}")
        print()
        
        results["summary"] = {
            "total_qa": total_qa,
            "passed": total_passed,
            "rejected": total_rejected,
            "rejection_rate": rejection_rate,
            "total_retries": total_retries
        }
        
        # Save results
        output_file = f"challenge_results_{version.lower()}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved: {output_file}\n")
        
        return results
        
    finally:
        # Restore original if swapped
        if use_seif_critic:
            import shutil
            critic_original = Path("src/agents/critic_agent.py")
            critic_backup = Path("src/agents/critic_agent_backup_challenge.py")
            
            if critic_backup.exists():
                shutil.copy2(critic_backup, critic_original)
                critic_backup.unlink()
                print("🔄 Restored original critic_agent.py")

if __name__ == "__main__":
    print_section("CHALLENGING TEST - AMBIGUOUS CONTENT", "🔴")
    print("This test uses deliberately ambiguous/incomplete chunks to challenge the system.")
    print("Goal: Force critic rejections and trigger retry loops.")
    print("\n⏱️  Starting in 2 seconds...")
    import time
    time.sleep(2)
    
    print_section("TEST 1: CURRENT VERSION (before Seif's merge)")
    # Note: We already merged, so this uses a backup
    results_current = run_challenging_test(use_seif_critic=False)
    
    print("\n⏸️  Test 1 complete. Starting test 2 in 3 seconds...")
    import time
    time.sleep(3)
    
    print_section("TEST 2: SEIF'S VERSION")
    results_seif = run_challenging_test(use_seif_critic=True)
    
    print_section("FINAL COMPARISON")
    print(f"CURRENT: {results_current['summary']['rejected']}/{results_current['summary']['total_qa']} rejected ({results_current['summary']['rejection_rate']:.1f}%)")
    print(f"SEIF:    {results_seif['summary']['rejected']}/{results_seif['summary']['total_qa']} rejected ({results_seif['summary']['rejection_rate']:.1f}%)")
    print()
    print(f"CURRENT: {results_current['summary']['total_retries']} total retries")
    print(f"SEIF:    {results_seif['summary']['total_retries']} total retries")
    print()
    print("✅ Challenging test complete!")
