"""
Test CriticV4 - Orchestrateur 2-Phases sur GPU
Tests avec un chunk réel de cours (MI201)

Étape 4 du plan d'implémentation hybride.
"""

import os
import sys
import json
import time
import logging

os.environ['PYTHONUNBUFFERED'] = '1'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))


# ─────────────────────────────────────────────────────────────────────────────
# Données de test (chunk réel de cours de mécanique des fluides)
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_GOOD = """
La loi de Bernoulli exprime la conservation de l'énergie dans un écoulement de fluide 
parfait incompressible en régime permanent. Elle s'écrit :

    P + ½ρv² + ρgh = constante

où P est la pression statique (Pa), ρ la masse volumique (kg/m³), v la vitesse 
d'écoulement (m/s), g l'accélération gravitationnelle (9.81 m/s²) et h la hauteur 
par rapport à un plan de référence (m).

Conditions d'application :
1. Fluide parfait (non visqueux)
2. Écoulement incompressible (ρ = cte)
3. Régime permanent (∂v/∂t = 0)
4. Le long d'une ligne de courant

Applications : tube de Pitot (mesure de vitesse), venturimètre (mesure de débit),
effet Coandă, portance des ailes d'avion.
"""

QUESTION_GOOD = "Quelles sont les quatre conditions d'application nécessaires pour utiliser la loi de Bernoulli ?"
ANSWER_GOOD = (
    "La loi de Bernoulli s'applique uniquement lorsque quatre conditions sont réunies : "
    "(1) le fluide doit être parfait, c'est-à-dire non visqueux ; "
    "(2) l'écoulement doit être incompressible (masse volumique constante) ; "
    "(3) le régime doit être permanent (vitesse ne varie pas dans le temps) ; "
    "(4) l'équation s'applique uniquement le long d'une ligne de courant."
)

# Question trop vague, réponse partielle
QUESTION_VAGUE = "C'est quoi Bernoulli ?"
ANSWER_VAGUE = "Bernoulli c'est une formule pour les fluides."

# Question hors-contexte
QUESTION_OOC = "Quel est l'impact des réseaux de neurones sur la mécanique quantique ?"
ANSWER_OOC = "Les réseaux de neurones permettent de simuler la mécanique quantique grâce au deep learning."

# Réponse avec hallucination
ANSWER_HALLUCINATED = (
    "La loi de Bernoulli a été publiée en 1738 par Daniel Bernoulli dans son livre 'Hydrodynamica'. "
    "Elle nécessite que le fluide soit parfait et que l'écoulement soit permanent. "
    "Elle s'applique aussi bien aux gaz qu'aux liquides à des vitesses supersoniques. "
    "Le coefficient de viscosité dynamique doit être inférieur à 0.001 Pa·s."
)


def run_test(name: str, chunk: str, question: str, answer: str, critic) -> dict:
    """Run a single test and return result."""
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST: {name}")
    logger.info(f"{'='*60}")
    logger.info(f"Question: {question[:100]}")
    logger.info(f"Answer:   {answer[:100]}...")

    t0 = time.time()
    result = critic.evaluate(
        chunk_content=chunk,
        question=question,
        answer=answer,
    )
    elapsed = time.time() - t0

    decision = result["decision"].upper()
    score = result["global_score"]
    rejection_phase = result.get("rejection_phase")
    feedback = result["feedback"]
    concepts = result.get("key_concepts", [])[:5]

    logger.info(f"\nRésultat: {decision}")
    logger.info(f"Score global: {score:.3f}/1.0")
    if rejection_phase:
        logger.info(f"Rejeté par: {rejection_phase}")
    logger.info(f"Key concepts: {concepts}")
    logger.info(f"Feedback: {feedback[:150]}...")
    logger.info(f"Durée: {elapsed:.1f}s")

    # Phase details
    if result.get("phase1"):
        p1 = result["phase1"]
        logger.info(f"\n[Phase 1] {p1['decision'].upper()} (score={p1['global_score']:.2f})")
        if result["phase1"].get("contextual"):
            ca = p1["contextual"]
            logger.info(f"  ContextualAnswerability: score={ca.get('score', '?')}/3, {ca.get('decision')}")
        if p1.get("pedagogical"):
            pv = p1["pedagogical"]
            logger.info(f"  PedagogicalValue: score={pv.get('score', '?'):.2f}/1.0, {pv.get('decision')}")

    if result.get("phase2_completeness"):
        pc = result["phase2_completeness"]
        logger.info(f"[Phase 2a] AnswerCompleteness: score={pc.get('score', '?')}/3, {pc.get('decision')}")

    if result.get("phase2_anchoring"):
        pa = result["phase2_anchoring"]
        logger.info(f"[Phase 2b] AnswerAnchoring: score={pa.get('score', '?')}/3, {pa.get('decision')}")

    return {
        "test_name": name,
        "decision": result["decision"],
        "global_score": score,
        "rejection_phase": rejection_phase,
        "elapsed_s": round(elapsed, 1),
        "concepts": concepts,
    }


def main():
    logger.info("Chargement du modèle LLM...")
    t_start = time.time()

    model_path = "/home/ensta/ensta-ben-amira/models/deepseek-r1-distill-qwen-32b/DeepSeek-R1-Distill-Qwen-32B-IQ3_M.gguf"

    from src.llm import LLMManager
    llm_manager = LLMManager.from_direct_llamacpp(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=4096,
    )
    llm = llm_manager.provider.llm
    logger.info(f"Modèle chargé en {time.time() - t_start:.1f}s")

    from src.critic_v4 import CriticV4
    critic = CriticV4(llm=llm, temperature=0.1, max_tokens=1000)
    logger.info("CriticV4 initialisé !")

    # ─────────────────── TESTS ───────────────────────────────────
    results = []
    tests = [
        ("BON_QA",     CHUNK_GOOD, QUESTION_GOOD,  ANSWER_GOOD),
        ("QA_VAGUE",   CHUNK_GOOD, QUESTION_VAGUE, ANSWER_VAGUE),
        ("QA_OOC",     CHUNK_GOOD, QUESTION_OOC,   ANSWER_OOC),
        ("HALLUCINE",  CHUNK_GOOD, QUESTION_GOOD,  ANSWER_HALLUCINATED),
    ]

    for name, chunk, q, a in tests:
        try:
            r = run_test(name, chunk, q, a, critic)
            results.append({"status": "ok", **r})
        except Exception as exc:
            logger.error(f"ERREUR test {name}: {exc}", exc_info=True)
            results.append({"status": "error", "test_name": name, "error": str(exc)})

    # ─────────────────── BILAN ───────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("BILAN DES TESTS")
    logger.info("=" * 60)
    total_ok = 0
    total_err = 0

    expected_decisions = {
        "BON_QA":    "pass",
        "QA_VAGUE":  "reject",
        "QA_OOC":    "reject",
        "HALLUCINE": "reject",
    }

    passed = 0
    for r in results:
        if r["status"] == "error":
            total_err += 1
            logger.warning(f"  ✗ {r['test_name']}: ERROR → {r.get('error', '?')[:80]}")
        else:
            total_ok += 1
            expected = expected_decisions.get(r["test_name"], "?")
            got = r["decision"]
            ok = (got == expected)
            passed += ok
            status_icon = "✓" if ok else "✗"
            logger.info(
                f"  {status_icon} {r['test_name']}: {got.upper()} "
                f"(expected={expected.upper()}, score={r['global_score']:.3f}, "
                f"time={r['elapsed_s']}s)"
            )

    total_time = sum(r.get("elapsed_s", 0) for r in results if r["status"] == "ok")
    logger.info(f"\n{passed}/{total_ok} tests corrects | {total_err} erreurs | Durée totale: {total_time:.1f}s")

    # Sauvegarde des résultats
    output_path = "/home/ensta/ensta-ben-amira/projects/Agentic_AI/test_critic_v4_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nRésultats sauvegardés → {output_path}")

    # Code de retour
    if total_err > 0:
        sys.exit(2)
    if passed < total_ok:
        logger.warning("Certains tests ont eu des résultats inattendus (voir bilan ci-dessus)")
    logger.info("Tests terminés avec succès !")


if __name__ == "__main__":
    main()
