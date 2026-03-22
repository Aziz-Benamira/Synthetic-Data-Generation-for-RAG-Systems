"""
annotate_difficulty.py — Post-traitement d'un Gold Dataset existant

Ajoute les champs difficulty_level, difficulty_label, difficulty_justification
à chaque entrée d'un Gold Dataset déjà généré, sans relancer la génération QA.

Usage:
    python3 scripts/annotate_difficulty.py \\
        --gold output/gold_dataset_v4_full.jsonl \\
        --chunks /path/to/chunks_mi201.json \\
        --output output/gold_dataset_v4_full_with_difficulty.jsonl

    # Annoter seulement les entrées qui n'ont pas encore de niveau
    python3 scripts/annotate_difficulty.py \\
        --gold output/gold_dataset_v4_full.jsonl \\
        --chunks /path/to/chunks_mi201.json \\
        --skip-existing

Variables d'environnement:
    MODEL_PATH — chemin vers le GGUF (défaut: ~/models/deepseek-r1-distill-qwen-32b/...)
    CHUNKS_PATH — chemin vers chunks JSON (si non fourni en argument)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Chemins par défaut ────────────────────────────────────────────────────────

DEFAULT_MODEL = os.path.expanduser(
    "~/models/deepseek-r1-distill-qwen-32b/DeepSeek-R1-Distill-Qwen-32B-IQ3_M.gguf"
)
DEFAULT_CHUNKS = (
    "/home/ensta/ensta-ben-amira/projects/Agentic_AI/"
    "experiments/critic_v2_baseline/data/chunks_mi201.json"
)


# ── Chargement des données ────────────────────────────────────────────────────

def load_gold_dataset(path: str) -> list:
    """Charge un gold dataset JSONL ou JSON."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Gold dataset introuvable: {path}")

    entries = []
    if p.suffix == ".jsonl":
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip().replace("\x00", "")
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Ligne ignorée (JSON invalide): {e}")
    else:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        entries = data if isinstance(data, list) else data.get("entries", [])

    logger.info(f"Gold dataset chargé: {len(entries)} entrées depuis {path}")
    return entries


def build_chunk_index(chunks_path: str) -> dict:
    """Construit un index chunk_id → content depuis le fichier chunks JSON."""
    p = Path(chunks_path)
    if not p.exists():
        raise FileNotFoundError(f"Fichier chunks introuvable: {chunks_path}")

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    chunks = data if isinstance(data, list) else data.get("chunks", [])
    index = {c["chunk_id"]: c.get("content", "") for c in chunks if "chunk_id" in c}
    logger.info(f"Index chunks: {len(index)} entrées depuis {chunks_path}")
    return index


# ── Sauvegarde ────────────────────────────────────────────────────────────────

def save_dataset(entries: list, output_path: str):
    """Sauvegarde en JSONL et JSON lisible."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # JSONL
    with open(out, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # JSON lisible
    json_out = out.with_suffix(".json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    logger.info(f"Dataset sauvegardé: {out} + {json_out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Annote un Gold Dataset existant avec les niveaux de difficulté (Bloom 1–5)"
    )
    parser.add_argument(
        "--gold", required=True,
        help="Chemin vers le gold dataset (.jsonl ou .json)"
    )
    parser.add_argument(
        "--chunks", default=DEFAULT_CHUNKS,
        help="Chemin vers le fichier chunks JSON (pour retrouver le contenu source)"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Chemin vers le modèle GGUF"
    )
    parser.add_argument(
        "--output", default=None,
        help="Fichier de sortie (défaut: <gold>_with_difficulty.jsonl)"
    )
    parser.add_argument(
        "--n-ctx", type=int, default=4096,
        help="Taille du contexte LLM (défaut: 4096)"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Sauter les entrées qui ont déjà un difficulty_level"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Traiter seulement les N premières entrées (pour test)"
    )
    args = parser.parse_args()

    # Résoudre le chemin de sortie
    if args.output is None:
        gold_path = Path(args.gold)
        args.output = str(
            gold_path.parent / (gold_path.stem + "_with_difficulty.jsonl")
        )

    # Ajouter le répertoire racine du projet au path Python
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    # ── Charger les données ───────────────────────────────────────────────────
    entries = load_gold_dataset(args.gold)
    chunk_index = build_chunk_index(args.chunks)

    if args.limit:
        entries = entries[:args.limit]
        logger.info(f"Limité à {args.limit} entrées (--limit)")

    # Compter combien d'entrées sont à traiter
    to_process = [
        e for e in entries
        if not (args.skip_existing and e.get("difficulty_level") is not None)
    ]
    already_done = len(entries) - len(to_process)
    if already_done > 0:
        logger.info(f"{already_done} entrées déjà annotées → ignorées (--skip-existing)")
    logger.info(f"{len(to_process)} entrées à annoter")

    if not to_process:
        logger.info("Rien à faire. Dataset déjà complet.")
        save_dataset(entries, args.output)
        return

    # ── Charger le modèle ─────────────────────────────────────────────────────
    logger.info(f"Chargement du modèle: {args.model}")
    t0 = time.time()
    from llama_cpp import Llama
    llm = Llama(
        model_path=args.model,
        n_gpu_layers=-1,
        n_ctx=args.n_ctx,
        verbose=False,
        chat_format="chatml",
    )
    logger.info(f"Modèle chargé en {time.time() - t0:.1f}s")

    # ── Grader ────────────────────────────────────────────────────────────────
    from src.critic_v4.metrics import DifficultyGrader
    grader = DifficultyGrader(llm=llm, temperature=0.1, max_tokens=600)

    stats = {"success": 0, "fallback": 0, "error": 0}
    t_start = time.time()

    for i, entry in enumerate(entries):
        # Sauter si déjà annoté et --skip-existing
        if args.skip_existing and entry.get("difficulty_level") is not None:
            continue

        question = entry.get("question", "")
        chunk_id = entry.get("chunk_id", "")
        chunk_content = chunk_index.get(chunk_id, "")

        if not chunk_content:
            logger.warning(f"[{i+1}] chunk_id '{chunk_id}' introuvable dans l'index → contenu vide")

        logger.info(f"[{i+1}/{len(entries)}] {chunk_id} — {question[:60]}...")

        try:
            result = grader.grade(question=question, chunk_content=chunk_content)
            entry["difficulty_level"] = result["level"]
            entry["difficulty_label"] = result["label"]
            entry["difficulty_justification"] = result["justification"]
            stats["success"] += 1
            logger.info(
                f"  → Level {result['level']} ({result['label']}) | "
                f"{result['justification'][:80]}..."
            )
        except Exception as exc:
            # En cas d'erreur, on conserve l'entrée sans niveau et on continue
            entry.setdefault("difficulty_level", None)
            entry.setdefault("difficulty_label", None)
            entry.setdefault("difficulty_justification", None)
            stats["error"] += 1
            logger.error(f"  ✗ Erreur: {exc}")

        # Checkpoint tous les 20 pour ne pas perdre le travail
        if (i + 1) % 20 == 0:
            save_dataset(entries, args.output)
            logger.info(f"  💾 Checkpoint sauvegardé ({i+1}/{len(entries)})")

    # ── Résumé ────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info("\n" + "=" * 60)
    logger.info("BILAN ANNOTATION DIFFICULTÉ")
    logger.info("=" * 60)
    logger.info(f"  Total entrées    : {len(entries)}")
    logger.info(f"  ✓ Annotées       : {stats['success']}")
    logger.info(f"  ✗ Erreurs        : {stats['error']}")
    logger.info(f"  Durée            : {elapsed:.0f}s ({elapsed/max(len(to_process),1):.1f}s/entrée)")

    # Distribution des niveaux
    levels = [e.get("difficulty_level") for e in entries if e.get("difficulty_level")]
    if levels:
        from collections import Counter
        dist = Counter(levels)
        labels = {1: "Factuel", 2: "Compréhension", 3: "Application", 4: "Analyse", 5: "Synthèse"}
        logger.info("\n  Distribution des niveaux:")
        for lvl in sorted(dist):
            pct = dist[lvl] / len(levels) * 100
            logger.info(f"    Level {lvl} ({labels.get(lvl, '?'):15s}): {dist[lvl]:3d}  ({pct:.1f}%)")

    save_dataset(entries, args.output)
    logger.info(f"\n  Fichier de sortie: {args.output}")
    logger.info("=" * 60)

    # Libérer le modèle
    del llm


if __name__ == "__main__":
    main()
