"""
sog_retriever.py
================
Pont entre le QuestionGenerator et l'AnswerGenerator.

Reçoit une question validée, interroge le graphe de connaissances SoG,
et retourne les passages contextuels pertinents (texte + entités + relations)
à injecter dans le prompt de l'AnswerGenerator.

Deux modes de récupération :
  - retrieve()          : cosinus + expansion 1-hop plate (mode original)
  - retrieve_multihop() : BFS guidé par similarité sémantique (Section 3.2 du
                          papier SoG).  A chaque hop, on sélectionne les W
                          voisins les plus *similaires* au nœud racine, ce qui
                          produit des chaînes conceptuelles cohérentes plutôt
                          qu'une expansion aléatoire.

Usage:
    from sog_retriever import SoGRetriever

    retriever = SoGRetriever(graph, embed_fn)
    # mode simple (backward-compat)
    context = retriever.retrieve(question, top_k=5, depth=1)
    # mode multi-hop BFS (recommandé)
    context = retriever.retrieve_multihop(question, top_k=3, depth=2, top_w=3)
    # context["passages"]   → list[str]
    # context["entities"]   → list[str]  (entités clés trouvées)
    # context["relations"]  → list[str]  (paires entité-entité)
    # context["formatted"]  → str        (prêt à injecter dans le prompt)
"""
from __future__ import annotations

import numpy as np
from context_graph import ContextGraph, Paragraph
from cross_document_sampling import EmbeddingIndex, _traverse_from_root


class SoGRetriever:
    """Récupère du contexte enrichi depuis le graphe de connaissances."""

    def __init__(
        self,
        graph: ContextGraph,
        embed_fn,
        precompute: bool = True,
        batch_embed_fn=None,
    ):
        """
        Args:
            graph:          ContextGraph chargé depuis le cache JSON.
            embed_fn:       callable(str)       -> np.ndarray  (single text)
            precompute:     Pre-compute all paragraph embeddings at init.
            batch_embed_fn: callable(list[str]) -> list[np.ndarray]  (optional)
                            If provided, used at precompute time for a single
                            batched encode call instead of N individual calls.
        """
        self.graph = graph
        self.index = EmbeddingIndex(embed_fn)

        # Collect unique paragraphs
        self._all_paras: list[Paragraph] = []
        seen: set[str] = set()
        for paras in self.graph.mapping.values():
            for p in paras:
                if p.para_id not in seen:
                    seen.add(p.para_id)
                    self._all_paras.append(p)

        # Pre-compute paragraph embeddings
        self._para_vecs: np.ndarray | None = None
        if precompute and self._all_paras:
            import logging
            _logger = logging.getLogger(__name__)
            _logger.info(f"Pré-calcul des embeddings pour {len(self._all_paras)} paragraphes...")

            if batch_embed_fn is not None:
                # Single batched call — much faster on GPU
                texts = [p.text for p in self._all_paras]
                raw   = batch_embed_fn(texts)  # list[ndarray] or ndarray (N, dim)
                if isinstance(raw, np.ndarray) and raw.ndim == 2:
                    vecs_raw = [raw[i] for i in range(len(raw))]
                else:
                    vecs_raw = list(raw)
                vecs = []
                for v in vecs_raw:
                    v = np.array(v, dtype=np.float32)
                    n = float(np.linalg.norm(v))
                    vecs.append(v / n if n > 0 else v)
            else:
                # Per-text fallback
                vecs = []
                for p in self._all_paras:
                    v = self.index._fn(p.text).astype(np.float32)
                    n = float(np.linalg.norm(v))
                    vecs.append(v / n if n > 0 else v)

            self._para_vecs = np.stack(vecs)  # shape: (N, dim)
            _logger.info(f"  → Embeddings pré-calculés ✓ ({self._para_vecs.shape})")

            # Pre-populate EmbeddingIndex cache so _traverse_from_root() can
            # reuse the same vectors without calling embed_fn again.
            for para, vec in zip(self._all_paras, vecs):
                self.index._cache[para.para_id] = vec

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        depth: int = 1,
    ) -> dict:
        """
        Étant donné une question validée :
          1. Embed la question.
          2. Score tous les paragraphes du graphe par similarité cosinus.
          3. Récupère les top_k paragraphes les plus pertinents.
          4. Étend via les voisins dans le graphe (profondeur `depth`).
          5. Retourne passages, entités, relations et texte formaté.

        Returns:
            dict avec:
            - passages   : list[str]  — textes des paragraphes pertinents
            - entities   : list[str]  — entités clés trouvées sur le chemin
            - relations  : list[str]  — relations entité↔entité traversées
            - formatted  : str        — bloc prêt à injecter dans le prompt
        """
        q_vec = self._embed_text(question)

        # Score tous les paragraphes — utilise la matrice pré-calculée si dispo
        if self._para_vecs is not None:
            scores = self._para_vecs @ q_vec  # (N,)
            top_indices = np.argsort(-scores)[:top_k]
            top_paras = [self._all_paras[i] for i in top_indices]
        else:
            scored: list[tuple[float, Paragraph]] = []
            seen_ids: set[str] = set()
            for paras in self.graph.mapping.values():
                for para in paras:
                    if para.para_id in seen_ids:
                        continue
                    seen_ids.add(para.para_id)
                    p_vec = self._embed_text(para.text)
                    score = float(np.dot(q_vec, p_vec))
                    scored.append((score, para))
            scored.sort(key=lambda x: -x[0])
            top_paras = [p for _, p in scored[:top_k]]

        # Expansion via voisins dans le graphe
        expanded: list[Paragraph] = list(top_paras)
        expanded_ids: set[str] = {p.para_id for p in top_paras}
        found_entities: list[str] = []
        found_relations: list[str] = []

        for para in top_paras:
            for entity in para.entities:
                if entity not in found_entities:
                    found_entities.append(entity)
                for neighbour_entity in self.graph.neighbors(entity):
                    rel = f"{entity} ↔ {neighbour_entity}"
                    if rel not in found_relations:
                        found_relations.append(rel)
                    if neighbour_entity not in found_entities:
                        found_entities.append(neighbour_entity)
                    for nb_para in self.graph.mapping.get(neighbour_entity, []):
                        if nb_para.para_id not in expanded_ids:
                            expanded.append(nb_para)
                            expanded_ids.add(nb_para.para_id)

        # Construire la liste de passages (top_k d'abord, puis expansion)
        top_ids = {p.para_id for p in top_paras}
        passages = [p.text for p in top_paras]
        for p in expanded:
            if p.para_id not in top_ids:
                passages.append(p.text)

        # Limiter à un nombre raisonnable
        max_passages = top_k * 3
        passages = passages[:max_passages]
        found_entities = found_entities[:20]
        found_relations = found_relations[:15]

        # Formater pour injection dans le prompt
        formatted = self._format_context(passages, found_entities, found_relations)

        return {
            "passages": passages,
            "entities": found_entities,
            "relations": found_relations,
            "formatted": formatted,
        }

    def retrieve_multihop(
        self,
        question: str,
        top_k: int = 3,
        depth: int = 2,
        top_w: int = 3,
    ) -> dict:
        """
        Récupération multi-hop guidée par similarité sémantique (Section 3.2 SoG).

        Algorithme :
          1. Embed la question → q_vec.
          2. Trouver les top_k paragraphes «graine» par similarité cosinus
             (même logique que retrieve()).
          3. Pour chaque paragraphe graine et chacune de ses entités, lancer
             un BFS (_traverse_from_root) de profondeur `depth`.
             À chaque saut, on garde les top_w voisins les plus similaires
             au paragraphe graine → chaîne conceptuelle cohérente.
          4. Collecter tous les nœuds des chemins → passages uniques.
          5. Les entités et relations tracent le chemin parcouru dans le graphe.

        Ce mode est supérieur à retrieve() car :
          - Il ne prend pas n'importe quel voisin ; il choisit le plus similaire.
          - Il peut atteindre des concepts à 2 sauts (multi-hop).
          - Les entités retournées forment des chaînes causales / thématiques.

        Args:
            question : la question validée par le QuestionGenerator.
            top_k    : nombre de paragraphes graines (ancrage initial).
            depth    : profondeur BFS (2 = 2 sauts depuis la graine).
            top_w    : largeur à chaque saut (W voisins gardés).

        Returns:
            dict identique à retrieve() : passages, entities, relations, formatted.
        """
        q_vec = self._embed_text(question)

        # ── 1. Seed paragraphs ────────────────────────────────────────────────
        if self._para_vecs is not None:
            scores     = self._para_vecs @ q_vec
            top_idx    = np.argsort(-scores)[:top_k]
            seed_paras = [self._all_paras[i] for i in top_idx]
        else:
            scored: list[tuple[float, Paragraph]] = []
            seen_ids: set[str] = set()
            for paras in self.graph.mapping.values():
                for para in paras:
                    if para.para_id in seen_ids:
                        continue
                    seen_ids.add(para.para_id)
                    p_vec = self._embed_text(para.text)
                    scored.append((float(np.dot(q_vec, p_vec)), para))
            scored.sort(key=lambda x: -x[0])
            seed_paras = [p for _, p in scored[:top_k]]

        # ── 2. BFS traversal from each seed entity ────────────────────────────
        all_paths = []
        for seed_para in seed_paras:
            for entity in seed_para.entities:
                if entity in self.graph.nodes:
                    paths = _traverse_from_root(
                        root_entity=entity,
                        root_para=seed_para,
                        graph=self.graph,
                        index=self.index,
                        depth=depth,
                        top_w=top_w,
                    )
                    all_paths.extend(paths)

        # ── 3. Collect passages, entities, relations from paths ───────────────
        seen_para_ids: set[str] = set()
        passages:      list[str] = []
        found_entities: list[str] = []
        found_relations: list[str] = []

        # Prefer longer (multi-hop) paths — they carry richer bridging context
        all_paths.sort(key=lambda p: -len(p))

        for path in all_paths:
            for i, node in enumerate(path):
                # Passage
                if node.paragraph.para_id not in seen_para_ids:
                    seen_para_ids.add(node.paragraph.para_id)
                    passages.append(node.paragraph.text)
                # Entity
                if node.entity not in found_entities:
                    found_entities.append(node.entity)
                # Relation (consecutive nodes along the path)
                if i > 0:
                    prev = path[i - 1].entity
                    rel  = f"{prev} ↔ {node.entity}"
                    rev  = f"{node.entity} ↔ {prev}"
                    if rel not in found_relations and rev not in found_relations:
                        found_relations.append(rel)

        # Also surface 1-hop neighbors of found entities for breadth
        for entity in list(found_entities[:10]):
            for nb in self.graph.neighbors(entity):
                if nb not in found_entities:
                    found_entities.append(nb)
                rel = f"{entity} ↔ {nb}"
                rev = f"{nb} ↔ {entity}"
                if rel not in found_relations and rev not in found_relations:
                    found_relations.append(rel)
                for nb_para in self.graph.mapping.get(nb, []):
                    if nb_para.para_id not in seen_para_ids:
                        seen_para_ids.add(nb_para.para_id)
                        passages.append(nb_para.text)

        # Limit outputs to keep prompt size manageable
        max_passages = top_k * 5
        passages        = passages[:max_passages]
        found_entities  = found_entities[:20]
        found_relations = found_relations[:15]

        formatted = self._format_context(passages, found_entities, found_relations)
        return {
            "passages":  passages,
            "entities":  found_entities,
            "relations": found_relations,
            "formatted": formatted,
        }

    def _embed_text(self, text: str) -> np.ndarray:
        """Embed et normalise un texte."""
        vec = self.index._fn(text).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec

    @staticmethod
    def _format_context(
        passages: list[str],
        entities: list[str],
        relations: list[str],
    ) -> str:
        """Formate le contexte enrichi pour injection dans le prompt."""
        parts = []

        if entities:
            parts.append(
                "=== ENTITÉS CLÉS (graphe de connaissances) ===\n"
                + ", ".join(entities[:15])
            )

        if relations:
            parts.append(
                "=== RELATIONS IDENTIFIÉES ===\n"
                + "\n".join(f"  • {r}" for r in relations[:10])
            )

        parts.append("=== PASSAGES PERTINENTS (graphe de connaissances) ===")
        for i, p in enumerate(passages, 1):
            # Tronquer les passages trop longs
            text = p[:600] if len(p) > 600 else p
            parts.append(f"\n[Passage {i}]\n{text}")

        return "\n\n".join(parts)
