"""
Pipeline V4 - Génération Gold Dataset
======================================

Flux complet : Chunks JSON → Question (V3) → Réponse (V3) → CriticV4 → Gold Dataset

Architecture :
    Pour chaque chunk :
    ┌─────────────────────────────────────────────────────────┐
    │  1. QuestionGeneratorV3                                 │
    │     ├─ Hint diversité (ScopedMemory)                    │
    │     ├─ Génère question candidate                        │
    │     ├─ Valide Phase 1 (CriticV4)  ← max 3 tentatives   │
    │     └─ PASS → enregistre concepts dans ScopedMemory    │
    │                                                         │
    │  2. AnswerGeneratorV3                                   │
    │     ├─ Génère réponse candidate                         │
    │     ├─ Valide Phase 2 (CriticV4)  ← max 2 tentatives   │
    │     └─ PASS → ajoute au Gold Dataset                   │
    │                                                         │
    │  Si REJECT après retries → skip ce chunk               │
    └─────────────────────────────────────────────────────────┘

Usage :
    from src.orchestrator.pipeline_v4 import PipelineV4, PipelineV4Config

    config = PipelineV4Config(
        chunks_path="experiments/critic_v2_baseline/data/chunks_mi201.json",
        output_path="output/gold_dataset_v4.jsonl",
        max_chunks=20,
    )
    pipeline = PipelineV4(config=config, llm=llm)
    dataset = pipeline.run()
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineV4Config:
    """
    Configuration du pipeline V4.

    Paramètres clés :
    - chunks_path        : fichier JSON des chunks pré-calculés
    - output_path        : où sauvegarder le dataset final (JSONL)
    - max_chunks         : limite pour les tests (None = tout traiter)
    - min_chunk_length   : ignorer les chunks trop courts
    - max_q_retries      : tentatives max de génération de question
    - max_a_retries      : tentatives max de génération de réponse
    - checkpoint_every   : sauvegarder l'état tous les N chunks
    """
    chunks_path: str
    output_path: str = "output/gold_dataset_v4.jsonl"
    max_chunks: Optional[int] = None
    min_chunk_length: int = 300
    semantic_types: Optional[List[str]] = None   # None = tous les types
    max_q_retries: int = 3
    max_a_retries: int = 2
    q_temperature: float = 0.7
    a_temperature: float = 0.3
    checkpoint_every: int = 10
    enable_difficulty_grading: bool = False   # Phase 3 optionnelle (Bloom 1–5)


# ─────────────────────────────────────────────────────────────────────────────
# Entry du dataset
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GoldEntry:
    """Une entrée validée dans le Gold Dataset."""
    question: str
    answer: str

    # Source
    chunk_id: str
    source_file: str
    chapter: str
    section: str
    page_range: List[int]

    # Scores qualité
    phase1_score: float            # score question (0-1)
    phase2_completeness_score: float
    phase2_anchoring_score: float
    phase2_score: float            # score réponse (0-1)
    global_score: float            # score combiné

    # Méta
    q_attempts: int                # tentatives pour la question
    a_attempts: int                # tentatives pour la réponse
    key_concepts: List[str]
    timestamp: str = ""

    # Difficulté (Phase 3, optionnelle)
    difficulty_level: Optional[int] = None       # 1–5 (taxonomie de Bloom)
    difficulty_label: Optional[str] = None       # Factuel / Compréhension / Application / Analyse / Synthèse
    difficulty_justification: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not d["timestamp"]:
            d["timestamp"] = datetime.now().isoformat()
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline V4
# ─────────────────────────────────────────────────────────────────────────────

class PipelineV4:
    """
    Pipeline end-to-end de génération de Gold Dataset.

    Différences vs Pipeline V1/V2/V3 :
    - Validation intégrée dans les générateurs (boucle de retry interne)
    - ScopedMemory pour la diversité des questions
    - Un seul LLM partagé (pas de modèle séparé pour le critic)
    - Format chunks JSON (pas de parsing PDF à la volée)
    """

    def __init__(self, config: PipelineV4Config, llm: Any):
        """
        Args:
            config: Configuration du pipeline
            llm: Instance LLM partagée (llm_manager.provider.llm)
        """
        self.config = config
        self.llm = llm
        self.dataset: List[GoldEntry] = []
        self._init_components()

    def _init_components(self):
        """Initialise tous les composants (lazy, partagent le même LLM)."""
        from src.utils.scoped_memory import ScopedMemory
        from src.critic_v4 import QuestionEvaluator
        from src.critic_v4.metrics import AnswerCompleteness, AnswerAnchoring, DifficultyGrader
        from src.agents import QuestionGeneratorV3, AnswerGeneratorV3

        self.memory = ScopedMemory()

        self.q_evaluator = QuestionEvaluator(
            llm=self.llm,
            temperature=0.1,
            max_tokens=1000,
        )
        self.a_completeness = AnswerCompleteness(llm=self.llm, temperature=0.1, max_tokens=1000)
        self.a_anchoring = AnswerAnchoring(llm=self.llm, temperature=0.1, max_tokens=1000)

        self.q_gen = QuestionGeneratorV3(
            llm=self.llm,
            scoped_memory=self.memory,
            question_evaluator=self.q_evaluator,
            temperature=self.config.q_temperature,
            max_tokens=300,
            max_retries=self.config.max_q_retries,
        )
        self.a_gen = AnswerGeneratorV3(
            llm=self.llm,
            completeness_evaluator=self.a_completeness,
            anchoring_evaluator=self.a_anchoring,
            temperature=self.config.a_temperature,
            max_tokens=600,
            max_retries=self.config.max_a_retries,
        )

        # Phase 3 — Difficulty Grader (instancié uniquement si activé)
        self.difficulty_grader = (
            DifficultyGrader(llm=self.llm, temperature=0.1, max_tokens=600)
            if self.config.enable_difficulty_grading
            else None
        )

        components = "ScopedMemory, QuestionGeneratorV3, AnswerGeneratorV3"
        if self.difficulty_grader:
            components += ", DifficultyGrader"
        logger.info(f"Composants initialisés : {components}")

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> List[Dict[str, Any]]:
        """
        Lance le pipeline complet.

        Returns:
            Liste des entrées Gold Dataset (dicts)
        """
        logger.info("=" * 60)
        logger.info("PIPELINE V4 — Génération Gold Dataset")
        logger.info("=" * 60)
        t_start = time.time()

        # 1. Charger les chunks
        chunks = self._load_chunks()
        logger.info(f"→ {len(chunks)} chunks chargés depuis {self.config.chunks_path}")

        # 2. Filtrer
        chunks = self._filter_chunks(chunks)
        logger.info(f"→ {len(chunks)} chunks après filtrage")

        if not chunks:
            logger.warning("Aucun chunk à traiter !")
            return []

        # 3. Traiter chunk par chunk
        stats = {"pass": 0, "reject_q": 0, "reject_a": 0, "error": 0, "fallback_q": 0, "fallback_a": 0}

        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            logger.info(f"\n{'─'*50}")
            logger.info(f"[{i+1}/{len(chunks)}] {chunk_id} | {chunk.get('chapter','?')[:40]}")

            try:
                entry = self._process_chunk(chunk)
                if entry is not None:
                    self.dataset.append(entry)
                    stats["pass"] += 1
                    if entry.q_attempts > 1:
                        stats["fallback_q"] += 1
                    if entry.a_attempts > 1:
                        stats["fallback_a"] += 1
                    logger.info(
                        f"  ✓ GOLD (q_score={entry.phase1_score:.2f}, "
                        f"a_score={entry.phase2_score:.2f}, "
                        f"global={entry.global_score:.2f})"
                    )
                else:
                    if self._last_reject_phase == "question":
                        stats["reject_q"] += 1
                        logger.info("  ✗ Skip — question rejetée après retries")
                    else:
                        stats["reject_a"] += 1
                        logger.info("  ✗ Skip — réponse rejetée après retries")

            except Exception as exc:
                stats["error"] += 1
                logger.error(f"  ✗ Erreur: {exc}", exc_info=False)

            # Checkpoint
            if self.config.checkpoint_every and (i + 1) % self.config.checkpoint_every == 0:
                self._save(is_checkpoint=True)
                logger.info(f"  💾 Checkpoint sauvegardé ({len(self.dataset)} entrées)")

        # 4. Sauvegarder le dataset final
        self._save(is_checkpoint=False)

        elapsed = time.time() - t_start
        total = len(chunks)
        logger.info("\n" + "=" * 60)
        logger.info("BILAN PIPELINE V4")
        logger.info("=" * 60)
        logger.info(f"  Chunks traités   : {total}")
        logger.info(f"  ✓ Gold entries   : {stats['pass']} ({stats['pass']/total*100:.1f}%)")
        logger.info(f"  ✗ Rejet question : {stats['reject_q']}")
        logger.info(f"  ✗ Rejet réponse  : {stats['reject_a']}")
        logger.info(f"  ⚠ Erreurs        : {stats['error']}")
        logger.info(f"  Durée totale     : {elapsed:.0f}s ({elapsed/total:.0f}s/chunk)")
        logger.info(f"  Fichier sortie   : {self.config.output_path}")
        logger.info("=" * 60)

        return [e.to_dict() for e in self.dataset]

    # ── Private ──────────────────────────────────────────────────────────────

    _last_reject_phase: str = ""

    def _process_chunk(self, chunk: Dict[str, Any]) -> Optional[GoldEntry]:
        """
        Traite un chunk : génère question → réponse → construit l'entrée Gold.

        Returns:
            GoldEntry si la paire passe, None si rejetée après retries.
        """
        # ── Générer et valider la question ────────────────────────
        q_result = self.q_gen.generate(chunk)

        if not q_result["question"]:
            self._last_reject_phase = "question"
            return None

        if q_result["status"] == "fallback":
            # Question retournée mais pas validée — on tente quand même la réponse
            logger.info(
                f"  ⚠ Question fallback (phase1_score={q_result['phase1_score']:.2f}), "
                "on tente la réponse quand même"
            )

        question = q_result["question"]
        phase1_score = q_result["phase1_score"] or 0.0

        # ── Générer et valider la réponse ─────────────────────────
        a_result = self.a_gen.generate(question=question, chunk=chunk)

        if not a_result["answer"]:
            self._last_reject_phase = "answer"
            return None

        if a_result["status"] == "fallback":
            # Réponse retournée mais pas validée
            logger.info(
                f"  ⚠ Réponse fallback (phase2_score={a_result['phase2_score']:.2f})"
            )

        # ── Score global ──────────────────────────────────────────
        # Phase1 (40%) + Phase2 (60%)
        phase2_score = a_result["phase2_score"] or 0.0
        global_score = round(0.4 * phase1_score + 0.6 * phase2_score, 3)

        # ── Construire l'entrée Gold ──────────────────────────────
        entry = GoldEntry(
            question=question,
            answer=a_result["answer"],
            chunk_id=chunk.get("chunk_id", ""),
            source_file=str(self.config.chunks_path),
            chapter=chunk.get("chapter", ""),
            section=chunk.get("section", ""),
            page_range=list(chunk.get("page_range", [0, 0])),
            phase1_score=round(phase1_score, 3),
            phase2_completeness_score=round(a_result.get("phase2_completeness_score") or 0.0, 3),
            phase2_anchoring_score=round(a_result.get("phase2_anchoring_score") or 0.0, 3),
            phase2_score=round(phase2_score, 3),
            global_score=global_score,
            q_attempts=q_result["attempts"],
            a_attempts=a_result["attempts"],
            key_concepts=q_result.get("key_concepts", []),
            timestamp=datetime.now().isoformat(),
        )

        # ── Phase 3 : Difficulty Grading (optionnelle) ────────────
        if self.difficulty_grader is not None:
            try:
                diff = self.difficulty_grader.grade(
                    question=question,
                    chunk_content=chunk.get("content", ""),
                )
                entry.difficulty_level = diff["level"]
                entry.difficulty_label = diff["label"]
                entry.difficulty_justification = diff["justification"]
                logger.info(
                    f"  📊 Difficulty: Level {diff['level']} — {diff['label']}"
                )
            except Exception as exc:
                # Non-bloquant : on conserve l'entrée sans difficulté
                logger.warning(f"  ⚠ Difficulty grading échoué (non bloquant): {exc}")

        return entry

    def _load_chunks(self) -> List[Dict[str, Any]]:
        """Charge les chunks depuis un fichier JSON."""
        path = Path(self.config.chunks_path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier chunks introuvable: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Supporter les deux formats : liste directe ou {"metadata":..., "chunks":[...]}
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "chunks" in data:
            return data["chunks"]
        else:
            raise ValueError(f"Format JSON non reconnu dans {path}")

    def _filter_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filtre les chunks selon la config."""
        filtered = [c for c in chunks if len(c.get("content", "")) >= self.config.min_chunk_length]

        if self.config.semantic_types:
            filtered = [c for c in filtered if c.get("semantic_type") in self.config.semantic_types]

        if self.config.max_chunks:
            filtered = filtered[:self.config.max_chunks]

        return filtered

    def _save(self, is_checkpoint: bool = False):
        """Sauvegarde le dataset en JSONL."""
        path = Path(self.config.output_path)
        if is_checkpoint:
            path = path.with_suffix("") / f"checkpoint_{len(self.dataset)}.jsonl"

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for entry in self.dataset:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
