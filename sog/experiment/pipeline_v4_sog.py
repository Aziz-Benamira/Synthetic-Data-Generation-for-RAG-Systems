"""
PipelineV4SoG
=============

Extension of PipelineV4 with Synthesize-on-Graph (SoG) knowledge graph context.

Pipeline flow (per chunk):
    1. QuestionGeneratorV3.generate(chunk)                    → question
    2. SoGRetriever.retrieve(question)                        → graph_context
    3. AnswerGeneratorV3SoG.generate(q, chunk, graph_ctx, …)  → answer
    4. CriticV4 scores (embedded in generators)               → GoldEntrySoG

Output format is a superset of the standard GoldEntry JSON — all original
fields are preserved, with three new fields added:
    graph_entities : list[str]   entities retrieved from the graph
    graph_relations: list[str]   entity pairs (a ↔ b)
    sog_mode       : str         "combined" | "graph_only"

This file configures sys.path at import time so that:
  - experiment/src/ (SoG modules)  is importable as flat modules
  - Agentic_AI/     (original src)  is importable via `from src.xxx import ...`
"""

import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

# ── sys.path setup ────────────────────────────────────────────────────────────
_EXP_DIR    = Path(__file__).parent
_SOG_SRC    = str(_EXP_DIR / "src")
_AGENTIC_AI = "/home/ensta/ensta-ben-amira/projects/Agentic_AI"

# Remove '' / '.' so experiment/src/ is never mistaken for the `src` package.
sys.path = [p for p in sys.path if p not in ("", ".")]

if _SOG_SRC not in sys.path:
    sys.path.insert(0, _SOG_SRC)       # flat SoG imports: sog_retriever, context_graph …

if _AGENTIC_AI not in sys.path:
    sys.path.insert(1, _AGENTIC_AI)    # src.orchestrator.pipeline_v4, src.agents …

# ── Imports ───────────────────────────────────────────────────────────────────
from src.orchestrator.pipeline_v4 import PipelineV4, GoldEntry
from src.orchestrator.pipeline_v4 import PipelineV4Config as _BasePipelineV4Config

from answer_generator_v3_sog import AnswerGeneratorV3SoG
from sog_retriever import SoGRetriever
from context_graph import ContextGraph

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_graph(path: str) -> ContextGraph:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ContextGraph.from_dict(data)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineV4SoGConfig(_BasePipelineV4Config):
    """
    PipelineV4Config + SoG parameters.

    New fields:
        sog_graph_path : absolute/relative path to the pre-built graph JSON
        sog_mode       : "combined"   — chunk text + graph context
                         "graph_only" — only graph context, no chunk text
                         "disabled"   — SoG off, identical to plain PipelineV4
        sog_top_k      : number of seed paragraphs for initial cosine retrieval
        sog_depth      : BFS depth for multi-hop graph traversal
        sog_top_w      : width at each BFS hop (top-W most similar neighbours)
        sog_multihop   : True  → use retrieve_multihop() (Section 3.2 BFS)
                         False → use retrieve()          (flat 1-hop expansion)
    """
    sog_graph_path: str = ""
    sog_mode: str = "combined"   # "combined" | "graph_only" | "disabled"
    sog_top_k: int = 3
    sog_depth: int = 2
    sog_top_w: int = 3
    sog_multihop: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# GoldEntrySoG — superset of GoldEntry with graph metadata
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GoldEntrySoG(GoldEntry):
    """GoldEntry extended with SoG metadata (fully JSON-serialisable)."""
    graph_entities: List[str]  = field(default_factory=list)
    graph_relations: List[str] = field(default_factory=list)
    sog_mode: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# PipelineV4SoG
# ─────────────────────────────────────────────────────────────────────────────

class PipelineV4SoG(PipelineV4):
    """
    PipelineV4 with SoG knowledge graph context injection.

    Usage:
        from pipeline_v4_sog import PipelineV4SoG, PipelineV4SoGConfig

        config = PipelineV4SoGConfig(
            chunks_path="data/tipler_chunks.json",
            output_path="output/gold_sog.jsonl",
            sog_graph_path="data/Tipler_Llewellyn_context_graph.json",
            sog_mode="combined",
            max_chunks=10,
        )
        pipeline = PipelineV4SoG(config=config, llm=llm, embed_fn=embed_fn)
        dataset = pipeline.run()
    """

    def __init__(self, config: PipelineV4SoGConfig, llm: Any, embed_fn=None, batch_embed_fn=None):
        # Stash before super().__init__() triggers _init_components()
        self._sog_config = config
        self._embed_fn = embed_fn
        self._batch_embed_fn = batch_embed_fn
        super().__init__(config=config, llm=llm)

    # ── Component initialisation ─────────────────────────────────────────────

    def _init_components(self):
        """
        Calls parent _init_components() then:
          1. Replaces self.a_gen with AnswerGeneratorV3SoG
          2. Instantiates SoGRetriever (if enabled)
        """
        super()._init_components()

        cfg           = getattr(self, "_sog_config", self.config)
        embed_fn      = getattr(self, "_embed_fn", None)
        batch_embed_fn = getattr(self, "_batch_embed_fn", None)

        # ── PATCH: re-create all CriticV4 evaluators with higher max_tokens ─
        # DeepSeek R1 emits a <think>...</think> block before JSON output.
        # The parent hardcodes max_tokens=1000 which is exhausted by the think
        # block → empty JSON response → "Expecting value at char 0" crash.
        # We need at least 2000 tokens:  ~900 think + ~300 JSON output + margin.
        from src.critic_v4 import QuestionEvaluator
        from src.critic_v4.metrics import AnswerCompleteness, AnswerAnchoring
        from src.agents import QuestionGeneratorV3

        self.q_evaluator = QuestionEvaluator(
            llm=self.llm,
            temperature=0.1,
            max_tokens=2000,       # ← was 1000 in parent
        )
        self.a_completeness = AnswerCompleteness(
            llm=self.llm, temperature=0.1, max_tokens=2000,
        )
        self.a_anchoring = AnswerAnchoring(
            llm=self.llm, temperature=0.1, max_tokens=2000,
        )
        logger.info("  CriticV4 evaluators max_tokens=2000 (DeepSeek R1 think-budget) ✓")

        self.q_gen = QuestionGeneratorV3(
            llm=self.llm,
            scoped_memory=self.memory,
            question_evaluator=self.q_evaluator,
            temperature=self.config.q_temperature,
            max_tokens=1500,
            max_retries=self.config.max_q_retries,
        )
        logger.info("  QuestionGeneratorV3 max_tokens=1500 (DeepSeek R1 think-budget) ✓")

        # ── Replace standard AnswerGeneratorV3 with the SoG-aware subclass ───
        # a_completeness / a_anchoring were already created by parent.
        self.a_gen = AnswerGeneratorV3SoG(
            llm=self.llm,
            completeness_evaluator=self.a_completeness,
            anchoring_evaluator=self.a_anchoring,
            temperature=self.config.a_temperature,
            max_tokens=700,
            max_retries=self.config.max_a_retries,
        )
        logger.info("  AnswerGeneratorV3SoG initialisé ✓")

        # SoG retriever
        if cfg.sog_mode != "disabled" and cfg.sog_graph_path and embed_fn is not None:
            logger.info(f"  Chargement graphe SoG: {cfg.sog_graph_path}")
            graph = _load_graph(cfg.sog_graph_path)
            self.retriever = SoGRetriever(
                graph, embed_fn, precompute=True, batch_embed_fn=batch_embed_fn
            )
            logger.info(
                f"  SoGRetriever prêt ✓ "
                f"(mode={cfg.sog_mode}, top_k={cfg.sog_top_k}, "
                f"depth={cfg.sog_depth}, multihop={cfg.sog_multihop})"
            )
        else:
            self.retriever = None
            logger.info(f"  SoGRetriever désactivé (mode={cfg.sog_mode})")

    # ── Core per-chunk processing ─────────────────────────────────────────────

    def _process_chunk(self, chunk: Dict[str, Any]) -> Optional[GoldEntrySoG]:
        """
        Override: injects graph context between question generation and answer
        generation.  All critic scoring logic is unchanged (stays in generators).
        """
        cfg = self._sog_config

        # 1. Generate & validate question (identical to parent)
        q_result = self.q_gen.generate(chunk)

        if not q_result["question"]:
            self._last_reject_phase = "question"
            return None

        if q_result["status"] == "fallback":
            logger.info(
                f"  ⚠ Question fallback (phase1_score={q_result['phase1_score']:.2f})"
            )

        question    = q_result["question"]
        phase1_score = q_result["phase1_score"] or 0.0

        # 2. Graph traversal
        graph_context  = ""
        graph_entities: List[str] = []
        graph_relations: List[str] = []

        if self.retriever is not None:
            try:
                if cfg.sog_multihop:
                    retrieved = self.retriever.retrieve_multihop(
                        question,
                        top_k=cfg.sog_top_k,
                        depth=cfg.sog_depth,
                        top_w=cfg.sog_top_w,
                    )
                else:
                    retrieved = self.retriever.retrieve(
                        question, top_k=cfg.sog_top_k, depth=cfg.sog_depth
                    )
                graph_context  = retrieved["formatted"]
                graph_entities = retrieved["entities"]
                graph_relations = retrieved["relations"]
                logger.info(
                    f"  SoG: {len(graph_entities)} entités, "
                    f"{len(graph_relations)} relations, "
                    f"{len(retrieved['passages'])} passages"
                )
            except Exception as exc:
                logger.warning(f"  SoG retrieval échoué ({exc}) — on continue sans graphe")

        # 3. Generate & validate answer with graph context
        a_result = self.a_gen.generate(
            question=question,
            chunk=chunk,
            graph_context=graph_context,
            sog_mode=cfg.sog_mode,
        )

        if not a_result["answer"]:
            self._last_reject_phase = "answer"
            return None

        if a_result["status"] == "fallback":
            logger.info(
                f"  ⚠ Réponse fallback (phase2_score={a_result['phase2_score']:.2f})"
            )

        # 4. Scores & GoldEntrySoG
        phase2_score = a_result["phase2_score"] or 0.0
        global_score = round(0.4 * phase1_score + 0.6 * phase2_score, 3)

        return GoldEntrySoG(
            question=question,
            answer=a_result["answer"],
            chunk_id=chunk.get("chunk_id", ""),
            source_file=str(self.config.chunks_path),
            chapter=chunk.get("chapter", ""),
            section=chunk.get("section", ""),
            page_range=list(chunk.get("page_range", [0, 0])),
            phase1_score=round(phase1_score, 3),
            phase2_completeness_score=round(
                a_result.get("phase2_completeness_score") or 0.0, 3
            ),
            phase2_anchoring_score=round(
                a_result.get("phase2_anchoring_score") or 0.0, 3
            ),
            phase2_score=round(phase2_score, 3),
            global_score=global_score,
            q_attempts=q_result["attempts"],
            a_attempts=a_result["attempts"],
            key_concepts=q_result.get("key_concepts", []),
            timestamp=datetime.now().isoformat(),
            graph_entities=graph_entities,
            graph_relations=graph_relations,
            sog_mode=cfg.sog_mode,
        )
