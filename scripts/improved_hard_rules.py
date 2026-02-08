"""
IMPROVED HARD RULES - Tuned for Production
===========================================

Based on Seif's hard rules with adjustments:
1. Numbers rule: More lenient (allow ±1 tolerance for rounding)
2. Why/How rule: Expanded causal markers, only strict for clear why/how
3. Short answer rule: Adjusted thresholds (15+ words, <40 chars)
4. Question repetition: Increased overlap threshold to 80%
5. Oral language: Expanded markers
6. NEW RULE: Vague pronouns without referents

Changes:
- Rule 1: Allow small numerical differences (rounding errors)
- Rule 2: More causal markers, check answer too
- Rule 3: 15 words question, 40 chars answer (was 10/50)
- Rule 4: 80% overlap threshold (was 70%)
- Rule 5: More oral markers
- Rule 6: NEW - detect "il", "elle", "ça", "cela" without clear referent
