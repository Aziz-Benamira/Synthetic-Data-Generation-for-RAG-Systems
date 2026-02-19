"""
Test End-to-End Pipeline V4
===========================
Étape 7 : test complet sur 5 chunks réels de chunks_mi201.json
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

MODEL_PATH = "/home/ensta/ensta-ben-amira/models/deepseek-r1-distill-qwen-32b/DeepSeek-R1-Distill-Qwen-32B-IQ3_M.gguf"
CHUNKS_PATH = "/home/ensta/ensta-ben-amira/projects/Agentic_AI/experiments/critic_v2_baseline/data/chunks_mi201.json"
OUTPUT_PATH = "/home/ensta/ensta-ben-amira/projects/Agentic_AI/output/gold_dataset_v4_test.jsonl"


def main():
    logger.info("=" * 60)
    logger.info("TEST END-TO-END PIPELINE V4")
    logger.info("=" * 60)

    # ── Chargement LLM ────────────────────────────────────────────
    logger.info("Chargement du modèle LLM...")
    t0 = time.time()

    from src.llm import LLMManager
    llm_manager = LLMManager.from_direct_llamacpp(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,
        n_ctx=4096,
    )
    llm = llm_manager.provider.llm
    logger.info(f"Modèle chargé en {time.time() - t0:.1f}s")

    # ── Lancer le pipeline sur 5 chunks ──────────────────────────
    from src.orchestrator.pipeline_v4 import PipelineV4, PipelineV4Config

    config = PipelineV4Config(
        chunks_path=CHUNKS_PATH,
        output_path=OUTPUT_PATH,
        max_chunks=5,
        min_chunk_length=300,
        max_q_retries=3,
        max_a_retries=2,
        checkpoint_every=0,   # pas de checkpoint pour le test
    )

    pipeline = PipelineV4(config=config, llm=llm)
    dataset = pipeline.run()

    # ── Afficher les résultats ────────────────────────────────────
    logger.info(f"\n{len(dataset)} entrées Gold générées")

    for i, entry in enumerate(dataset, 1):
        logger.info(f"\n--- Entrée {i} ---")
        logger.info(f"  Chunk     : {entry['chunk_id']}")
        logger.info(f"  Question  : {entry['question'][:120]}")
        logger.info(f"  Réponse   : {entry['answer'][:120]}...")
        logger.info(f"  Scores    : global={entry['global_score']:.3f} | "
                    f"phase1={entry['phase1_score']:.2f} | "
                    f"completeness={entry['phase2_completeness_score']:.1f}/3 | "
                    f"anchoring={entry['phase2_anchoring_score']:.1f}/3")
        logger.info(f"  Tentatives: Q={entry['q_attempts']} | A={entry['a_attempts']}")
        logger.info(f"  Concepts  : {entry['key_concepts'][:5]}")

    # ── Sauvegarder un JSON lisible en plus du JSONL ──────────────
    json_out = OUTPUT_PATH.replace(".jsonl", ".json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    logger.info(f"\nDataset sauvegardé → {OUTPUT_PATH}")
    logger.info(f"Dataset lisible   → {json_out}")

    if len(dataset) == 0:
        logger.warning("ATTENTION: Aucune entrée Gold produite sur 5 chunks !")
        logger.warning("Vérifier les seuils de validation ou augmenter max_retries.")
    else:
        avg_score = sum(e["global_score"] for e in dataset) / len(dataset)
        logger.info(f"Score global moyen : {avg_score:.3f}")
        logger.info("✓ Pipeline V4 opérationnel !")


if __name__ == "__main__":
    main()
