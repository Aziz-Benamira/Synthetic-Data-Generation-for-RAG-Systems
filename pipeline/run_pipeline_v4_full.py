"""
Pipeline V4 — Exécution complète sur tous les chunks de chunks_mi201.json
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

MODEL_PATH  = "/home/ensta/ensta-ben-amira/models/deepseek-r1-distill-qwen-32b/DeepSeek-R1-Distill-Qwen-32B-IQ3_M.gguf"
CHUNKS_PATH = "/home/ensta/ensta-ben-amira/projects/Agentic_AI/experiments/critic_v2_baseline/data/chunks_mi201.json"
OUTPUT_PATH = "/home/ensta/ensta-ben-amira/projects/Agentic_AI/output/gold_dataset_v4_full.jsonl"

def main():
    logger.info("=" * 60)
    logger.info("PIPELINE V4 — RUN COMPLET (100 chunks)")
    logger.info("=" * 60)

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

    from src.orchestrator.pipeline_v4 import PipelineV4, PipelineV4Config
    config = PipelineV4Config(
        chunks_path=CHUNKS_PATH,
        output_path=OUTPUT_PATH,
        max_chunks=None,          # tous les chunks
        min_chunk_length=300,
        max_q_retries=3,
        max_a_retries=2,
        checkpoint_every=10,      # sauvegarde tous les 10 chunks
    )

    pipeline = PipelineV4(config=config, llm=llm)
    dataset = pipeline.run()

    # Sauvegarde JSON lisible en plus
    json_out = OUTPUT_PATH.replace(".jsonl", ".json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{len(dataset)} entrées Gold sauvegardées")
    logger.info(f"  JSONL : {OUTPUT_PATH}")
    logger.info(f"  JSON  : {json_out}")

    if dataset:
        avg = sum(e["global_score"] for e in dataset) / len(dataset)
        logger.info(f"  Score global moyen : {avg:.3f}")

if __name__ == "__main__":
    main()
