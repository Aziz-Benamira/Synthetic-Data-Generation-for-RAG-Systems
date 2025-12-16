"""
Critic Agent - Constitutional AI Quality Filter
================================================

Evaluates QA pairs against 5 quality criteria using Constitutional AI rubrics.
Decision is BINARY: PASS or REJECT (no reformulation loop).

Design Principles:
- Constitutional AI approach: explicit rubrics, not implicit judgments
- Each criterion has clear pass/fail conditions
- Explanations are provided for transparency
- REJECT means DELETE, not "try again"

The 5 Evaluation Criteria:
1. ANCHORING: Answer is derivable from the chunk
2. LOCAL_ANSWERABILITY: Question answerable from chunk alone
3. FACTUAL_ACCURACY: No factual errors or hallucinations
4. COMPLETENESS: Answer addresses the question fully
5. CLARITY: Question and answer are clear and unambiguous

Input: QAPair + SemanticChunk (source)
Output: CriticEvaluation with PASS/REJECT decision
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import json
import re


class CriterionResult(Enum):
    """Result for a single criterion"""
    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"


class FinalDecision(Enum):
    """Final decision for a QA pair"""
    PASS = "pass"      # Include in dataset
    REJECT = "reject"  # Discard


@dataclass
class CriterionEvaluation:
    """Evaluation of a single criterion"""
    criterion: str
    result: CriterionResult
    score: float  # 0.0 to 1.0
    explanation: str
    evidence: List[str] = field(default_factory=list)  # Supporting evidence from text


@dataclass
class CriticEvaluation:
    """Complete evaluation of a QA pair"""
    # Source info
    question: str
    answer: str
    chunk_id: str
    
    # Individual criterion evaluations
    criteria_evaluations: Dict[str, CriterionEvaluation]
    
    # Final decision
    decision: FinalDecision
    overall_score: float  # 0.0 to 1.0
    
    # Summary
    passed_criteria: List[str]
    failed_criteria: List[str]
    rejection_reasons: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "chunk_id": self.chunk_id,
            "decision": self.decision.value,
            "overall_score": self.overall_score,
            "passed_criteria": self.passed_criteria,
            "failed_criteria": self.failed_criteria,
            "rejection_reasons": self.rejection_reasons,
            "criteria_details": {
                name: {
                    "result": eval.result.value,
                    "score": eval.score,
                    "explanation": eval.explanation
                }
                for name, eval in self.criteria_evaluations.items()
            }
        }


# =============================================================================
# CONSTITUTIONAL AI RUBRICS
# =============================================================================

RUBRICS_FR = {
    "anchoring": {
        "name": "Ancrage",
        "description": "La réponse est-elle ENTIÈREMENT dérivable du contenu du chunk?",
        "pass_conditions": [
            "Chaque affirmation de la réponse peut être retrouvée dans le chunk",
            "Aucune information externe n'est ajoutée",
            "Les citations correspondent au texte source"
        ],
        "fail_conditions": [
            "La réponse contient des informations absentes du chunk",
            "Des connaissances externes sont utilisées",
            "Les citations ne correspondent pas au texte"
        ]
    },
    "local_answerability": {
        "name": "Répondabilité Locale", 
        "description": "La question peut-elle être répondue avec UNIQUEMENT ce chunk?",
        "pass_conditions": [
            "Le chunk contient toutes les informations nécessaires",
            "Pas besoin de contexte externe pour comprendre",
            "La question ne fait pas référence à d'autres sections"
        ],
        "fail_conditions": [
            "La réponse nécessite des informations d'autres parties du document",
            "La question fait référence à des concepts non définis dans le chunk",
            "Le chunk ne contient qu'une partie de la réponse"
        ]
    },
    "factual_accuracy": {
        "name": "Exactitude Factuelle",
        "description": "La réponse est-elle factuellement correcte par rapport au chunk?",
        "pass_conditions": [
            "Aucune erreur factuelle détectable",
            "Les chiffres, noms, formules sont corrects",
            "Pas de contradiction avec le texte source"
        ],
        "fail_conditions": [
            "Erreurs dans les chiffres ou formules",
            "Noms ou termes incorrects",
            "Contradiction avec le contenu du chunk"
        ]
    },
    "completeness": {
        "name": "Complétude",
        "description": "La réponse répond-elle complètement à la question?",
        "pass_conditions": [
            "Tous les aspects de la question sont adressés",
            "La réponse n'est pas tronquée",
            "Les éléments clés sont présents"
        ],
        "fail_conditions": [
            "Des parties de la question sont ignorées",
            "La réponse est incomplète ou tronquée",
            "Des éléments essentiels manquent"
        ]
    },
    "clarity": {
        "name": "Clarté",
        "description": "La question et la réponse sont-elles claires et non ambiguës?",
        "pass_conditions": [
            "La question est compréhensible sans contexte externe",
            "La réponse est formulée clairement",
            "Pas d'ambiguïté dans les termes utilisés"
        ],
        "fail_conditions": [
            "La question est vague ou ambiguë",
            "La réponse est confuse ou mal structurée",
            "Les termes utilisés sont ambigus"
        ]
    }
}

RUBRICS_EN = {
    "anchoring": {
        "name": "Anchoring",
        "description": "Is the answer ENTIRELY derivable from the chunk content?",
        "pass_conditions": [
            "Every claim in the answer can be found in the chunk",
            "No external information is added",
            "Citations match the source text"
        ],
        "fail_conditions": [
            "Answer contains information not in the chunk",
            "External knowledge is used",
            "Citations don't match the text"
        ]
    },
    "local_answerability": {
        "name": "Local Answerability",
        "description": "Can the question be answered with ONLY this chunk?",
        "pass_conditions": [
            "Chunk contains all necessary information",
            "No external context needed",
            "Question doesn't reference other sections"
        ],
        "fail_conditions": [
            "Answer requires information from other parts",
            "Question references undefined concepts",
            "Chunk only contains partial answer"
        ]
    },
    "factual_accuracy": {
        "name": "Factual Accuracy",
        "description": "Is the answer factually correct relative to the chunk?",
        "pass_conditions": [
            "No detectable factual errors",
            "Numbers, names, formulas are correct",
            "No contradiction with source text"
        ],
        "fail_conditions": [
            "Errors in numbers or formulas",
            "Incorrect names or terms",
            "Contradiction with chunk content"
        ]
    },
    "completeness": {
        "name": "Completeness",
        "description": "Does the answer fully address the question?",
        "pass_conditions": [
            "All aspects of the question are addressed",
            "Answer is not truncated",
            "Key elements are present"
        ],
        "fail_conditions": [
            "Parts of the question are ignored",
            "Answer is incomplete or truncated",
            "Essential elements are missing"
        ]
    },
    "clarity": {
        "name": "Clarity",
        "description": "Are the question and answer clear and unambiguous?",
        "pass_conditions": [
            "Question is understandable without external context",
            "Answer is clearly formulated",
            "No ambiguity in terms used"
        ],
        "fail_conditions": [
            "Question is vague or ambiguous",
            "Answer is confusing or poorly structured",
            "Terms used are ambiguous"
        ]
    }
}


# =============================================================================
# PROMPTS FOR CRITIC EVALUATION
# =============================================================================

SYSTEM_PROMPT_FR = """Tu es un évaluateur TRÈS STRICT de qualité pour datasets de Question-Réponse.

⚠️ ATTENTION: Tu dois être EXTRÊMEMENT EXIGEANT. En cas de doute, PÉNALISE.
Le but est d'avoir un dataset PARFAIT. Il vaut mieux rejeter un bon QA que d'accepter un mauvais.

=== LES 5 CRITÈRES (TOUS OBLIGATOIRES) ===

1. ANCRAGE (anchoring) - VÉRIFIE MOT PAR MOT
   ✗ FAIL (score 0.0-0.5) si:
     - La réponse ajoute des EXEMPLES non présents dans le chunk
     - La réponse utilise des TERMES ou CONCEPTS absents du chunk
     - La réponse fait des DÉDUCTIONS ou INFÉRENCES non explicites
     - La réponse ajoute des EXPLICATIONS non présentes
   ✓ PASS (score 0.8-1.0) UNIQUEMENT si:
     - CHAQUE phrase de la réponse est DIRECTEMENT dans le chunk
     - Aucune paraphrase qui change le sens

2. RÉPONDABILITÉ LOCALE (local_answerability)
   ✗ FAIL si:
     - La question demande une COMPARAISON avec autre chose
     - La question demande des CONSÉQUENCES ou IMPLICATIONS non explicites
     - La question est trop GÉNÉRALE pour ce chunk spécifique
   ✓ PASS UNIQUEMENT si:
     - Le chunk contient EXPLICITEMENT la réponse complète

3. EXACTITUDE FACTUELLE (factual_accuracy)
   ✗ FAIL si:
     - Toute INTERPRÉTATION du texte source
     - Toute REFORMULATION qui change le sens
     - Tout ajout de NUANCE non présente
   ✓ PASS UNIQUEMENT si:
     - La réponse est FIDÈLE mot pour mot au chunk

4. COMPLÉTUDE (completeness)
   ✗ FAIL (score 0.3-0.5) si:
     - La réponse est TROP COURTE (moins de 2 phrases pour une question complexe)
     - La réponse ne répond qu'à UNE PARTIE de la question
     - La réponse est TRIVIALE et n'apporte pas de valeur
   ✗ FAIL (score 0.0-0.3) si:
     - La réponse est une seule phrase vague
     - La réponse répète juste la question

5. CLARTÉ (clarity)
   ✗ FAIL si:
     - La question utilise des termes VAGUES ("le truc", "ça", "chose")
     - La formulation est ORALE ou FAMILIÈRE
     - La structure est CONFUSE ou MAL ORGANISÉE
   ✓ PASS UNIQUEMENT si:
     - Style ACADÉMIQUE et PRÉCIS

=== EXEMPLES DE REJECTIONS (PATTERNS GÉNÉRIQUES) ===

EXEMPLE 1 - REJET pour ANCRAGE (exemple inventé):
Chunk: "[Définition X]"
Question: "Comment fonctionne X?"
Réponse: "X fonctionne ainsi. Par exemple, [exemple non présent dans le chunk]..."
→ REJET: anchoring=0.3 car l'exemple N'EST PAS dans le chunk source!
PATTERN: Toute réponse qui AJOUTE un exemple, une illustration, ou un cas concret NON PRÉSENT dans le chunk.

EXEMPLE 2 - REJET pour COMPLÉTUDE (réponse trop courte):
Question: "Qu'est-ce que [concept] et comment fonctionne-t-il?"
Réponse: "C'est [définition minimale en une phrase]."
→ REJET: completeness=0.4 car la réponse est TRIVIALE, ne répond qu'à la première partie.
PATTERN: Réponse qui ne fait que RÉPÉTER la définition sans développer quand la question demande plus.

EXEMPLE 3 - REJET pour CLARTÉ (formulation orale):
Question: "C'est quoi le truc avec [concept]?" ou "Comment ça marche ce machin?"
→ REJET: clarity=0.2 car formulation VAGUE et ORALE.
PATTERN: Questions utilisant "truc", "machin", "ça", "chose", ou style conversationnel.

EXEMPLE 4 - REJET pour ANCRAGE (déduction/inférence):
Chunk: "[Fait A]. [Fait B]."
Réponse: "Cela suggère que...", "On peut en déduire que...", "Cela implique que..."
→ REJET: anchoring=0.4 car les mots "suggère", "déduit", "implique" indiquent une INFÉRENCE.
PATTERN: Toute réponse qui fait une CONCLUSION LOGIQUE non explicitement écrite dans le chunk.

EXEMPLE 5 - REJET pour EXACTITUDE (reformulation incorrecte):
Chunk: "A est toujours B dans les conditions C"
Réponse: "A est B" (sans mentionner les conditions C)
→ REJET: factual_accuracy=0.5 car la nuance/condition est perdue.
PATTERN: Reformulation qui SIMPLIFIE TROP et perd des détails importants.

EXEMPLE 6 - REJET pour LOCAL_ANSWERABILITY:
Question: "Comment [concept X] se compare-t-il à [concept Y du chapitre précédent]?"
→ REJET: local_answerability=0.3 car nécessite info EXTERNE au chunk.
PATTERN: Questions qui font référence à d'autres parties du document ou à des connaissances externes.

=== RÈGLES DE SCORING ===
- 1.0 = PARFAIT (rare, réservé aux cas irréprochables)
- 0.8-0.9 = Très bon (quelques imperfections mineures)
- 0.7 = Acceptable (seuil minimum)
- 0.5-0.6 = Médiocre (problèmes notables)
- 0.3-0.4 = Mauvais (problèmes majeurs)
- 0.0-0.2 = Inacceptable

Tu dois être CALIBRÉ: un score de 1.0 est EXCEPTIONNEL, pas la norme!"""

SYSTEM_PROMPT_EN = """You are an expert quality evaluator for Question-Answer datasets intended for RAG system evaluation.

YOUR ROLE:
You must evaluate each Question-Answer pair against 5 strict criteria.
You are DEMANDING and only let HIGH QUALITY QAs pass.

THE 5 CRITERIA:

1. ANCHORING
   - Is the answer ENTIRELY derivable from the provided chunk?
   - No external information should be present
   - Score: 0.0-1.0

2. LOCAL ANSWERABILITY
   - Can the question be answered with ONLY this chunk?
   - No need for other parts of the document
   - Score: 0.0-1.0

3. FACTUAL ACCURACY
   - Is the answer correct relative to the chunk?
   - No errors, no hallucinations
   - Score: 0.0-1.0

4. COMPLETENESS
   - Does the answer address ALL aspects of the question?
   - Score: 0.0-1.0

5. CLARITY
   - Are the question and answer clear and unambiguous?
   - Score: 0.0-1.0

DECISION RULES:
- Score ≥ 0.7 for a criterion = PASS
- Score < 0.7 for a criterion = FAIL
- If ALL criteria PASS → Final decision: PASS
- If AT LEAST ONE criterion FAIL → Final decision: REJECT

Be STRICT but FAIR. The goal is to have a high-quality dataset."""


USER_PROMPT_TEMPLATE = """Évalue la paire Question-Réponse suivante selon les 5 critères.

=== CONTEXTE SOURCE (chunk) ===
Chunk ID: {chunk_id}
Chapitre: {chapter}
Section: {section}

Contenu du chunk:
---
{chunk_content}
---

=== PAIRE QA À ÉVALUER ===
Question: {question}
Réponse: {answer}
Citations fournies: {quotes}

=== INSTRUCTIONS ===
Évalue chaque critère avec un score de 0.0 à 1.0 et une explication.
Puis donne ta décision finale: PASS ou REJECT.

=== FORMAT DE SORTIE (JSON STRICT) ===
{{
  "criteria": {{
    "anchoring": {{
      "score": 0.0-1.0,
      "result": "pass|fail",
      "explanation": "Explication concise"
    }},
    "local_answerability": {{
      "score": 0.0-1.0,
      "result": "pass|fail",
      "explanation": "Explication concise"
    }},
    "factual_accuracy": {{
      "score": 0.0-1.0,
      "result": "pass|fail",
      "explanation": "Explication concise"
    }},
    "completeness": {{
      "score": 0.0-1.0,
      "result": "pass|fail",
      "explanation": "Explication concise"
    }},
    "clarity": {{
      "score": 0.0-1.0,
      "result": "pass|fail",
      "explanation": "Explication concise"
    }}
  }},
  "decision": "pass|reject",
  "overall_score": 0.0-1.0,
  "rejection_reasons": ["raison1", "raison2"] 
}}

Génère UNIQUEMENT le JSON, sans commentaires."""


# =============================================================================
# CRITIC AGENT CLASS
# =============================================================================

class CriticAgent:
    """
    Constitutional AI Critic for evaluating QA pair quality.
    
    Design:
    - Uses explicit rubrics for each criterion
    - Binary decision: PASS or REJECT
    - No reformulation loop - REJECT means discard
    - Provides detailed explanations for transparency
    """
    
    # Threshold for passing a criterion
    PASS_THRESHOLD = 0.7
    
    def __init__(
        self,
        llm_client: Any,
        model_name: str = "llama-3.3-70b-versatile",
        language: str = "fr",
        temperature: float = 0.2,  # Low temperature for consistent evaluation
        strict_mode: bool = True   # If True, ALL criteria must pass
    ):
        """
        Initialize the critic agent.
        
        Args:
            llm_client: LLM API client
            model_name: Model to use for evaluation
            language: "fr" or "en" for prompts
            temperature: LLM temperature (lower = more consistent)
            strict_mode: If True, all 5 criteria must pass. If False, majority vote.
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.language = language
        self.temperature = temperature
        self.strict_mode = strict_mode
        
        # Select prompts and rubrics based on language
        self.system_prompt = SYSTEM_PROMPT_FR if language == "fr" else SYSTEM_PROMPT_EN
        self.rubrics = RUBRICS_FR if language == "fr" else RUBRICS_EN
    
    def evaluate(
        self,
        qa_pair: Any,  # QAPair object
        chunk: Any     # SemanticChunk object
    ) -> CriticEvaluation:
        """
        Evaluate a single QA pair against all criteria.
        
        Args:
            qa_pair: QAPair object to evaluate
            chunk: Source SemanticChunk
            
        Returns:
            CriticEvaluation with decision and detailed scores
        """
        # Format quotes for prompt
        quotes_str = "; ".join(qa_pair.supporting_quotes) if qa_pair.supporting_quotes else "Aucune"
        
        # Build user prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            chunk_id=chunk.chunk_id,
            chapter=chunk.chapter_title,
            section=chunk.section_title,
            chunk_content=chunk.content,
            question=qa_pair.question,
            answer=qa_pair.answer,
            quotes=quotes_str
        )
        
        # Call LLM
        response = self._call_llm(user_prompt)
        
        # Parse response
        evaluation = self._parse_response(response, qa_pair, chunk)
        
        return evaluation
    
    def _call_llm(self, user_prompt: str) -> str:
        """Call the LLM API."""
        if hasattr(self.llm_client, 'chat'):
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=1500
            )
            return response.choices[0].message.content
        else:
            raise ValueError("Unsupported LLM client type")
    
    def _parse_response(
        self,
        response: str,
        qa_pair: Any,
        chunk: Any
    ) -> CriticEvaluation:
        """Parse LLM response into CriticEvaluation object."""
        try:
            # Try to parse JSON
            data = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except:
                    # Fallback to conservative evaluation
                    data = self._create_fallback_evaluation()
            else:
                data = self._create_fallback_evaluation()
        
        # Parse criteria evaluations
        criteria_evaluations = {}
        criteria_data = data.get("criteria", {})
        
        for criterion_name in ["anchoring", "local_answerability", "factual_accuracy", "completeness", "clarity"]:
            crit_data = criteria_data.get(criterion_name, {})
            score = float(crit_data.get("score", 0.5))
            result_str = crit_data.get("result", "uncertain")
            
            # Determine result based on score if not explicitly provided
            if result_str == "pass" or score >= self.PASS_THRESHOLD:
                result = CriterionResult.PASS
            elif result_str == "fail" or score < self.PASS_THRESHOLD:
                result = CriterionResult.FAIL
            else:
                result = CriterionResult.UNCERTAIN
            
            criteria_evaluations[criterion_name] = CriterionEvaluation(
                criterion=criterion_name,
                result=result,
                score=score,
                explanation=crit_data.get("explanation", ""),
                evidence=[]
            )
        
        # Determine passed and failed criteria
        passed = [name for name, eval in criteria_evaluations.items() if eval.result == CriterionResult.PASS]
        failed = [name for name, eval in criteria_evaluations.items() if eval.result == CriterionResult.FAIL]
        
        # Calculate overall score
        overall_score = sum(eval.score for eval in criteria_evaluations.values()) / len(criteria_evaluations)
        
        # Determine final decision
        if self.strict_mode:
            # All criteria must pass
            decision = FinalDecision.PASS if len(failed) == 0 else FinalDecision.REJECT
        else:
            # Majority vote (at least 4 out of 5)
            decision = FinalDecision.PASS if len(passed) >= 4 else FinalDecision.REJECT
        
        # Get rejection reasons
        rejection_reasons = data.get("rejection_reasons", [])
        if not rejection_reasons and failed:
            rejection_reasons = [f"Critère '{c}' non satisfait" for c in failed]
        
        return CriticEvaluation(
            question=qa_pair.question,
            answer=qa_pair.answer,
            chunk_id=chunk.chunk_id,
            criteria_evaluations=criteria_evaluations,
            decision=decision,
            overall_score=overall_score,
            passed_criteria=passed,
            failed_criteria=failed,
            rejection_reasons=rejection_reasons
        )
    
    def _create_fallback_evaluation(self) -> dict:
        """Create a conservative fallback evaluation when parsing fails."""
        return {
            "criteria": {
                "anchoring": {"score": 0.5, "result": "uncertain", "explanation": "Évaluation impossible"},
                "local_answerability": {"score": 0.5, "result": "uncertain", "explanation": "Évaluation impossible"},
                "factual_accuracy": {"score": 0.5, "result": "uncertain", "explanation": "Évaluation impossible"},
                "completeness": {"score": 0.5, "result": "uncertain", "explanation": "Évaluation impossible"},
                "clarity": {"score": 0.5, "result": "uncertain", "explanation": "Évaluation impossible"}
            },
            "decision": "reject",
            "overall_score": 0.5,
            "rejection_reasons": ["Évaluation automatique impossible - rejet par précaution"]
        }
    
    def evaluate_batch(
        self,
        qa_pairs: List[Tuple[Any, Any]],  # [(QAPair, SemanticChunk), ...]
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[CriticEvaluation], Dict[str, Any]]:
        """
        Evaluate multiple QA pairs and return statistics.
        
        Args:
            qa_pairs: List of (QAPair, SemanticChunk) tuples
            progress_callback: Optional callback(current, total)
            
        Returns:
            (list of evaluations, statistics dict)
        """
        evaluations = []
        total = len(qa_pairs)
        
        for i, (qa_pair, chunk) in enumerate(qa_pairs):
            evaluation = self.evaluate(qa_pair, chunk)
            evaluations.append(evaluation)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        # Calculate statistics
        passed = [e for e in evaluations if e.decision == FinalDecision.PASS]
        rejected = [e for e in evaluations if e.decision == FinalDecision.REJECT]
        
        # Failure reasons breakdown
        failure_counts = {}
        for e in rejected:
            for criterion in e.failed_criteria:
                failure_counts[criterion] = failure_counts.get(criterion, 0) + 1
        
        stats = {
            "total": total,
            "passed": len(passed),
            "rejected": len(rejected),
            "pass_rate": len(passed) / total if total > 0 else 0,
            "average_score": sum(e.overall_score for e in evaluations) / total if total > 0 else 0,
            "failure_breakdown": failure_counts,
            "criterion_averages": {
                criterion: sum(e.criteria_evaluations[criterion].score for e in evaluations) / total
                for criterion in ["anchoring", "local_answerability", "factual_accuracy", "completeness", "clarity"]
            } if total > 0 else {}
        }
        
        return evaluations, stats


# =============================================================================
# FILTERING UTILITIES
# =============================================================================

def filter_qa_pairs(
    evaluations: List[CriticEvaluation],
    qa_pairs: List[Any]
) -> Tuple[List[Any], List[Any]]:
    """
    Filter QA pairs based on critic evaluations.
    
    Returns:
        (passed_qa_pairs, rejected_qa_pairs)
    """
    passed = []
    rejected = []
    
    for eval, qa in zip(evaluations, qa_pairs):
        if eval.decision == FinalDecision.PASS:
            passed.append(qa)
        else:
            rejected.append(qa)
    
    return passed, rejected


def get_evaluation_summary(evaluations: List[CriticEvaluation]) -> str:
    """Generate a human-readable summary of evaluations."""
    total = len(evaluations)
    passed = sum(1 for e in evaluations if e.decision == FinalDecision.PASS)
    rejected = total - passed
    
    lines = [
        f"=== RÉSUMÉ ÉVALUATION ===",
        f"Total évalués: {total}",
        f"✅ Acceptés: {passed} ({100*passed/total:.1f}%)",
        f"❌ Rejetés: {rejected} ({100*rejected/total:.1f}%)",
        "",
        "Scores moyens par critère:"
    ]
    
    if total > 0:
        for criterion in ["anchoring", "local_answerability", "factual_accuracy", "completeness", "clarity"]:
            avg = sum(e.criteria_evaluations[criterion].score for e in evaluations) / total
            lines.append(f"  - {criterion}: {avg:.2f}")
    
    return "\n".join(lines)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("Critic Agent Module - Constitutional AI Quality Filter")
    print("=" * 60)
    print()
    print("5 Evaluation Criteria:")
    for key, rubric in RUBRICS_FR.items():
        print(f"  {key}: {rubric['name']}")
        print(f"    → {rubric['description']}")
    print()
    print("Decision Rules:")
    print("  - Score ≥ 0.7 → PASS")
    print("  - Score < 0.7 → FAIL")
    print("  - All criteria PASS → Final: PASS")
    print("  - Any criterion FAIL → Final: REJECT")
    print()
    print("Usage:")
    print("  from critic_agent import CriticAgent")
    print("  critic = CriticAgent(llm_client)")
    print("  evaluation = critic.evaluate(qa_pair, chunk)")
    print("  if evaluation.decision == FinalDecision.PASS:")
    print("      # Include in dataset")
