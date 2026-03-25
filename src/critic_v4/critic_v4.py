
"""
CriticV4 - Orchestrateur Principal 2-Phases

Architecture complète:
┌──────────────────────────────────────────────────────────────┐
│                        CRITIC V4                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  [INPUT] chunk + question + answer                           │
│                │                                             │
│  PHASE 1: Question Filtering                                 │
│  ├─ ContextualAnswerability (chunk a-t-il l'info?)           │
│  └─ PedagogicalValue (question est-elle bonne?)              │
│       REJECT → feedback → retry Question Generator          │
│       PASS ↓                                                 │
│                                                              │
│  PHASE 2: Answer Validation                                  │
│  ├─ AnswerCompleteness (réponse couvre tous les aspects?)    │
│  └─ AnswerAnchoring (réponse bien ancrée dans le chunk?)     │
│       REJECT → feedback → retry Answer Generator            │
│       PASS ↓                                                 │
│                                                              │
│  [OUTPUT] décision + scores + feedback + concepts            │
└──────────────────────────────────────────────────────────────┘

Usage:
    from src.critic_v4 import CriticV4
    from src.llm import LLMManager

    llm = LLMManager.from_direct_llamacpp(model_path=...).provider.llm
    critic = CriticV4(llm=llm)

    result = critic.evaluate(
        chunk_content="...",
        question="...",
        answer="...",
    )
    # result["decision"] == "pass" ou "reject"
    # result["feedback"]  → message pour le générateur
"""

import logging
from typing import Dict, Any, Optional

from .question_evaluator import QuestionEvaluator
from .metrics import AnswerCompleteness, AnswerAnchoring

logger = logging.getLogger(__name__)


class CriticV4:
    """
    Orchestrateur principal du Critic V4.

    Combine Phase 1 (filtrage question) + Phase 2 (validation réponse)
    pour produire une décision PASS/REJECT sur une paire (question, réponse).

    Optimisation:
    - Court-circuit Phase 1: si question rejetée, Phase 2 est skippée
    - Court-circuit Phase 2: si AnswerCompleteness REJECT, AnswerAnchoring peut être skippé

    Poids pour le score global final:
    - Phase 1 (question): 40%
      - ContextualAnswerability: 60% de 40% = 24%
      - PedagogicalValue: 40% de 40% = 16%
    - Phase 2 (réponse): 60%
      - AnswerCompleteness: 60% de 60% = 36%
      - AnswerAnchoring: 40% de 60% = 24%
    """

    WEIGHT_PHASE1 = 0.4
    WEIGHT_PHASE2 = 0.6
    WEIGHT_COMPLETENESS = 0.6
    WEIGHT_ANCHORING = 0.4

    def __init__(
        self,
        llm: Any = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        short_circuit_phase1: bool = True,
        short_circuit_phase2: bool = True,
    ):
        """
        Initialise CriticV4.

        Args:
            llm: Instance LLM partagée entre toutes les métriques
            temperature: Température pour les évaluations LLM (bas = déterministe)
            max_tokens: Tokens max par appel LLM
            short_circuit_phase1: Skip Phase 2 si Phase 1 REJECT
            short_circuit_phase2: Skip AnswerAnchoring si AnswerCompleteness REJECT
        """
        self.llm = llm
        self.short_circuit_phase1 = short_circuit_phase1
        self.short_circuit_phase2 = short_circuit_phase2

        # Phase 1: QuestionEvaluator (ContextualAnswerability + PedagogicalValue)
        self.question_evaluator = QuestionEvaluator(
            llm=llm,
            temperature=temperature,
            max_tokens=max_tokens,
            short_circuit=short_circuit_phase1,
        )

        # Phase 2: Answer Validation
        self.completeness_evaluator = AnswerCompleteness(
            llm=llm,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.anchoring_evaluator = AnswerAnchoring(
            llm=llm,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def evaluate(
        self,
        chunk_content: str,
        question: str,
        answer: str,
        llm: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Évalue une paire (question, réponse) complète.

        Args:
            chunk_content: Chunk de cours source
            question: Question générée
            answer: Réponse générée
            llm: Instance LLM optionnelle (override self.llm)

        Returns:
            Dict avec:
            - decision: "pass" ou "reject"
            - global_score: score final pondéré (0.0 à 1.0)
            - phase1: résultat complet Phase 1 (QuestionEvaluator)
            - phase2_completeness: résultat AnswerCompleteness (ou None)
            - phase2_anchoring: résultat AnswerAnchoring (ou None)
            - feedback: feedback consolidé (phase qui a échoué)
            - key_concepts: concepts clés (depuis Phase 1, pour ScopedMemory)
            - rejection_phase: "phase1" | "phase2_completeness" | "phase2_anchoring" | None
        """
        active_llm = llm or self.llm
        if active_llm is None:
            raise ValueError("Un LLM doit être fourni soit dans __init__ soit dans evaluate()")

        logger.info("=" * 60)
        logger.info(f"CriticV4: évaluation de QA pair")
        logger.info(f"  Question: {question[:80]}...")
        logger.info(f"  Answer:   {answer[:80]}...")

        # ── PHASE 1: Question Filtering ──────────────────────────
        logger.info("→ Phase 1: Question Filtering")
        phase1_result = self.question_evaluator.evaluate(
            chunk_content=chunk_content,
            question=question,
            llm=active_llm,
        )

        if phase1_result["decision"] == "reject":
            logger.info(f"  Phase 1 REJECT → court-circuit, skip Phase 2")
            return {
                "decision": "reject",
                "global_score": phase1_result["global_score"] * self.WEIGHT_PHASE1,
                "phase1": phase1_result,
                "phase2_completeness": None,
                "phase2_anchoring": None,
                "feedback": phase1_result["feedback"],
                "key_concepts": phase1_result.get("key_concepts", []),
                "rejection_phase": "phase1",
            }

        # ── PHASE 2A: Answer Completeness ────────────────────────
        logger.info("→ Phase 2a: Answer Completeness")
        completeness_result = self.completeness_evaluator.evaluate(
            question=question,
            answer=answer,
            llm=active_llm,
        )

        if self.short_circuit_phase2 and completeness_result["decision"] == "reject":
            logger.info("  Phase 2a REJECT → court-circuit, skip AnswerAnchoring")
            phase2_score = completeness_result["normalized_score"] * self.WEIGHT_COMPLETENESS
            global_score = (
                phase1_result["global_score"] * self.WEIGHT_PHASE1
                + phase2_score * self.WEIGHT_PHASE2
            )
            return {
                "decision": "reject",
                "global_score": round(global_score, 3),
                "phase1": phase1_result,
                "phase2_completeness": completeness_result,
                "phase2_anchoring": None,
                "feedback": completeness_result["feedback"],
                "key_concepts": phase1_result.get("key_concepts", []),
                "rejection_phase": "phase2_completeness",
            }

        # ── PHASE 2B: Answer Anchoring ───────────────────────────
        logger.info("→ Phase 2b: Answer Anchoring")
        anchoring_result = self.anchoring_evaluator.evaluate(
            chunk_content=chunk_content,
            question=question,
            answer=answer,
            llm=active_llm,
        )

        # ── Décision finale ──────────────────────────────────────
        completeness_pass = completeness_result["decision"] == "pass"
        anchoring_pass = anchoring_result["decision"] == "pass"
        final_decision = "pass" if (completeness_pass and anchoring_pass) else "reject"

        # Score Phase 2 pondéré
        phase2_score = (
            completeness_result["normalized_score"] * self.WEIGHT_COMPLETENESS
            + anchoring_result["normalized_score"] * self.WEIGHT_ANCHORING
        )

        # Score global
        global_score = (
            phase1_result["global_score"] * self.WEIGHT_PHASE1
            + phase2_score * self.WEIGHT_PHASE2
        )

        # Feedback et phase de rejet
        rejection_phase = None
        if final_decision == "reject":
            if not anchoring_pass:
                feedback = anchoring_result["feedback"]
                rejection_phase = "phase2_anchoring"
            else:
                feedback = completeness_result["feedback"]
                rejection_phase = "phase2_completeness"
        else:
            feedback = (
                f"QA pair acceptée (score global={global_score:.2f}/1.0). "
                f"Phase1={phase1_result['global_score']:.2f}, "
                f"Complétude={completeness_result['score']:.1f}/3, "
                f"Ancrage={anchoring_result['score']:.1f}/3."
            )

        result = {
            "decision": final_decision,
            "global_score": round(global_score, 3),
            "phase1": phase1_result,
            "phase2_completeness": completeness_result,
            "phase2_anchoring": anchoring_result,
            "feedback": feedback,
            "key_concepts": phase1_result.get("key_concepts", []),
            "rejection_phase": rejection_phase,
        }

        logger.info(
            f"CriticV4 final: {final_decision.upper()} "
            f"(global={global_score:.2f}, "
            f"p1={phase1_result['global_score']:.2f}, "
            f"compl={completeness_result['score']:.1f}/3, "
            f"anchor={anchoring_result['score']:.1f}/3)"
        )

        return result

    def evaluate_phase1_only(
        self,
        chunk_content: str,
        question: str,
        llm: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Évalue uniquement la Phase 1 (filtrage question sans réponse).

        Utile pour filtrer les questions AVANT de générer les réponses,
        économisant du temps de calcul.

        Returns:
            Résultat de QuestionEvaluator (decision, feedback, key_concepts...)
        """
        active_llm = llm or self.llm
        if active_llm is None:
            raise ValueError("Un LLM doit être fourni soit dans __init__ soit dans evaluate()")

        return self.question_evaluator.evaluate(
            chunk_content=chunk_content,
            question=question,
            llm=active_llm,
        )
