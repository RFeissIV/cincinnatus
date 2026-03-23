#!/usr/bin/env python3
"""
CINCINNATUS — Reasoning Engine v1 FINAL
=========================================
Cross-domain scientific knowledge graph reasoning with provenance.

Pipeline:
  Question -> Mistral (Best-of-N entity extraction)
  -> Entity matching + alias resolution
  -> LLM-guided graph traversal (Amayuelas Agent pattern, Section 4.2.1)
  -> Bidirectional BFS path search from source to LLM-selected targets
  -> Mistral (provenance-traced explanation)

The graph-guided traversal shows the LLM real neighbors from the adjacency
list, and the LLM decides which branches to follow. This eliminates entity
matching errors because every target comes directly from the graph.

Author: Richard A. Feiss IV, Ph.D. | MNPRO, University of Minnesota
"""

import os, sys, json, time, argparse, subprocess, math
import numpy as np
from collections import defaultdict, deque
from pathlib import Path

DATA_DIR = Path(os.path.expanduser("~/agronomic-ai/data"))
OLLAMA_MODEL = "mistral"
TOP_K_FAISS = 50
MAX_PATHS = 10
MAX_PATH_LEN = 4
MAX_EXPLAIN_PATHS = 5
BEST_OF_N = 3
MULTI_SEED_K = 3
BIDIR_MAX_EXPAND = 15000

def ollama_generate(prompt, temperature=0.1, max_tokens=500):
    try:
        result = subprocess.run(["ollama","run",OLLAMA_MODEL,prompt], capture_output=True, text=True, timeout=120)
        return result.stdout.strip()
    except subprocess.TimeoutExpired: return "[LLM timeout]"
    except FileNotFoundError: return "[Ollama not installed]"

def ollama_json(prompt, temperature=0.1):
    raw = ollama_generate(prompt, temperature)
    try:
        s, e = raw.find("["), raw.rfind("]")+1
        if s >= 0 and e > s: return json.loads(raw[s:e])
    except (json.JSONDecodeError, ValueError): pass
    return [raw.strip().strip('"').strip("'")]

class CincinnatusGraph:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.entity2id = {}; self.id2entity = {}
        self.rel2id = {}; self.id2rel = {}
        self.embeddings = None; self.emb_normed = None
        self.adj = defaultdict(list); self.adj_rev = defaultdict(list)
        self.faiss_index = None

    def load(self):
        t_start = time.time()
        print("  Loading entity2id.json...", end=" ", flush=True)
        t0 = time.time()
        with open(self.data_dir/"entity2id.json") as f: self.entity2id = json.load(f)
        self.id2entity = {v:k for k,v in self.entity2id.items()}
        print(f"{len(self.entity2id):,} entities ({time.time()-t0:.1f}s)")
        print("  Loading rel2id.json...", end=" ", flush=True)
        with open(self.data_dir/"rel2id.json") as f: self.rel2id = json.load(f)
        self.id2rel = {v:k for k,v in self.rel2id.items()}
        print(f"{len(self.rel2id)} relations")
        emb_path = self.data_dir/"sheaf_embeddings.npy"
        if not emb_path.exists(): emb_path = self.data_dir/"ensemble_embeddings.npy"
        print(f"  Loading embeddings: {emb_path.name}...", end=" ", flush=True)
        t0 = time.time()
        self.embeddings = np.load(str(emb_path))
        print(f"{self.embeddings.shape} ({time.time()-t0:.1f}s)")
        print("  Normalizing embeddings...", end=" ", flush=True)
        t0 = time.time()
        norms = np.maximum(np.linalg.norm(self.embeddings, axis=1, keepdims=True), 1e-8)
        self.emb_normed = (self.embeddings / norms).astype(np.float32)
        print(f"done ({time.time()-t0:.1f}s)")
        print("  Building FAISS index...", end=" ", flush=True)
        t0 = time.time()
        try:
            import faiss
            self.faiss_index = faiss.IndexFlatIP(self.emb_normed.shape[1])
            self.faiss_index.add(self.emb_normed)
            print(f"FAISS ({time.time()-t0:.1f}s)")
        except ImportError:
            print("faiss not available, numpy fallback")
        print("  Loading kuzu_edges.parquet...", end=" ", flush=True)
        t0 = time.time()
        try:
            import polars as pl
            df = pl.read_parquet(str(self.data_dir/"kuzu_edges.parquet"))
            for row in df.iter_rows(named=True):
                h,t,p,s = row["from"],row["to"],row["predicate"],row["source"]
                c = row.get("confidence",1.0)
                self.adj[h].append((p,t,s,c)); self.adj_rev[t].append((p,h,s,c))
            print(f"{len(df):,} triples ({time.time()-t0:.1f}s)")
        except ImportError:
            import pyarrow.parquet as pq
            tbl = pq.read_table(str(self.data_dir/"kuzu_edges.parquet")).to_pydict()
            for i in range(len(tbl["from"])):
                h,t,p,s = tbl["from"][i],tbl["to"][i],tbl["predicate"][i],tbl["source"][i]
                c = tbl.get("confidence",[1.0]*len(tbl["from"]))[i]
                self.adj[h].append((p,t,s,c)); self.adj_rev[t].append((p,h,s,c))
            print(f"{len(tbl['from']):,} triples ({time.time()-t0:.1f}s)")
        # Build inverted index for fast entity lookup
        print("  Building entity index...", end=" ", flush=True)
        t0 = time.time()
        self.name_exact = {}      # lowercase name -> [(original_name, eid)]
        self.word_index = defaultdict(set)  # word stem -> set of lowercase names
        for name, eid in self.entity2id.items():
            nl = name.lower()
            self.name_exact.setdefault(nl, []).append((name, eid))
            # Index word stems for fuzzy matching
            for w in nl.split():
                if len(w) > 3:
                    self.word_index[w[:5]].add(nl)
        print(f"{len(self.name_exact):,} unique names ({time.time()-t0:.1f}s)")
        print(f"  Total load time: {time.time()-t_start:.1f}s")

    def find_entities(self, query_terms):
        """Fast entity matching using inverted index.
        Priority: exact match (O(1)) > substring (indexed) > word overlap (indexed)
        """
        matches = []
        for q_raw in query_terms:
            q = q_raw.lower().strip()
            if not q:
                continue

            # 1. Exact match — O(1) dict lookup
            if q in self.name_exact:
                for name, eid in self.name_exact[q]:
                    matches.append((name, eid, 0.0))

            # 2. Substring match — check names containing the query
            #    Use word stems to narrow candidates instead of scanning all 12M
            q_stems = {w[:min(5, len(w))] for w in q.split() if len(w) > 3}
            if q_stems:
                # Find candidate names that share at least one stem
                candidates = set()
                for stem in q_stems:
                    candidates |= self.word_index.get(stem, set())
                # Check substring on candidates only (typically < 1000, not 12M)
                for nl in candidates:
                    if nl == q:
                        continue  # already handled by exact match
                    if q in nl:
                        coverage = len(q) / len(nl)
                        for name, eid in self.name_exact.get(nl, []):
                            matches.append((name, eid, 1.0 - coverage))
                    else:
                        # Word-level overlap
                        nw = set(nl.split())
                        ns = {w[:min(5, len(w))] for w in nw if len(w) > 3}
                        ov = q_stems & ns
                        if len(ov) >= 2 or (len(ov) >= 1 and len(q_stems) <= 2):
                            ratio = len(ov) / max(len(q_stems), 1)
                            for name, eid in self.name_exact.get(nl, []):
                                matches.append((name, eid, 1.5 - ratio))

        matches.sort(key=lambda x: x[2])
        seen = set(); result = []
        for name, eid, _ in matches:
            if eid not in seen:
                seen.add(eid); result.append((name, eid))
            if len(result) >= 20:
                break
        return result

    def resolve_aliases(self, entity_name):
        ALIAS_RELS = {"has_name","has_common_name","synonym_of","same_as"}
        canonical = []
        for variant in set([entity_name, entity_name.lower(), entity_name.upper(), entity_name.capitalize()]):
            for pred, source, src_db, conf in self.adj_rev.get(variant, []):
                if pred in ALIAS_RELS:
                    n_edges = len(self.adj.get(source, []))
                    eid = self.entity2id.get(source, -1)
                    if eid >= 0:  # Only include if entity has valid ID
                        canonical.append((source, eid, n_edges))
        canonical.sort(key=lambda x: x[2], reverse=True)
        return canonical

    def embedding_neighbors(self, entity_id, top_k=TOP_K_FAISS):
        if entity_id >= len(self.emb_normed): return []
        query = self.emb_normed[entity_id:entity_id+1]
        if self.faiss_index is not None:
            scores, indices = self.faiss_index.search(query, top_k+1)
            return [(self.id2entity[idx], idx, float(s)) for s, idx in zip(scores[0], indices[0])
                    if idx != entity_id and idx in self.id2entity][:top_k]
        else:
            sims = (self.emb_normed @ query.T).flatten()
            top_idx = np.argsort(sims)[::-1][:top_k+1]
            return [(self.id2entity[int(i)], int(i), float(sims[i])) for i in top_idx
                    if i != entity_id and i in self.id2entity][:top_k]

    DIRECTIONAL_RELS = {"causes","activates","inhibits","upregulates","downregulates",
        "catalyzes","catalyzed_by","treats","kills","metabolizes","regulates",
        "transports","precedes","eats","preys_on","parasite_of",
        "pathogen_of","pollinates","visits_flowers_of","adverse_outcome"}

    REL_WEIGHTS = {"causes":1.0,"activates":1.0,"inhibits":1.0,"upregulates":1.0,
        "downregulates":1.0,"kills":1.0,"adverse_outcome":1.0,"catalyzes":0.9,
        "catalyzed_by":0.9,"regulates":0.9,"metabolizes":0.9,"treats":0.9,
        "transports":0.9,"associated_with":0.7,"interacts_with":0.7,
        "gene_associated_with_disease":0.7,"involves_compound":0.7,
        "participates_in":0.7,"has_phenotype":0.7,"has_mechanism":0.7,
        "has_function":0.7,"eats":0.8,"preys_on":0.8,"parasite_of":0.8,
        "pathogen_of":0.8,"has_host":0.8,"pollinates":0.8,"mutualist_of":0.7,
        "visits_flowers_of":0.7,"is_a":0.5,"part_of":0.5,"child_of":0.5,
        "broader_than":0.5,"located_in":0.5,"found_in_organism":0.5,
        "detected_in":0.5,"has_name":0.1,"has_common_name":0.1,"synonym_of":0.1,
        "same_as":0.1,"has_value":0.1,"has_unit":0.1,"in_namespace":0.1,
        "has_property":0.3,"has_trait":0.3,"exposure_route":0.4,"has_direction":0.2,
        "has_kinetic_param":0.3,"has_optimal_condition":0.3,"has_dispersal_vector":0.4,
        "detected_taxon":0.4,"samples_biome":0.4,"studies_species":0.4,
        "resistant_to":0.6,"precedes":0.6}
    DEFAULT_REL_WEIGHT = 0.5

    def _get_neighbors(self, node):
        nbs = []
        for pred,nb,src,conf in self.adj.get(node,[]):
            nbs.append((nb, node, pred, nb, src, conf))
        for pred,nb,src,conf in self.adj_rev.get(node,[]):
            if pred not in self.DIRECTIONAL_RELS:
                nbs.append((nb, nb, pred, node, src, conf))
        return nbs

    def find_paths_bidirectional(self, source, target, max_len=MAX_PATH_LEN, max_paths=MAX_PATHS):
        if source not in self.adj and source not in self.adj_rev: return []
        if target not in self.adj and target not in self.adj_rev: return []
        fp = {source: None}; fq = deque([source]); fd = {source: 0}
        bp = {target: None}; bq = deque([target]); bd = {target: 0}
        meets = []; mh = max_len//2+1; exp = 0
        while (fq or bq) and exp < BIDIR_MAX_EXPAND and len(meets) < max_paths*2:
            if fq:
                n = fq.popleft(); exp += 1
                if fd.get(n,0) < mh:
                    for nb,h,p,t,s,c in self._get_neighbors(n):
                        if nb not in fp:
                            fp[nb] = (n,(h,p,t,s,c)); fd[nb] = fd[n]+1; fq.append(nb)
                            if nb in bp: meets.append(nb)
            if bq:
                n = bq.popleft(); exp += 1
                if bd.get(n,0) < mh:
                    for nb,h,p,t,s,c in self._get_neighbors(n):
                        if nb not in bp:
                            bp[nb] = (n,(h,p,t,s,c)); bd[nb] = bd[n]+1; bq.append(nb)
                            if nb in fp: meets.append(nb)
            if meets and exp > 100: break
        if not meets: return []
        found = []; seen = set()
        for m in meets[:max_paths*3]:
            fe = []; n = m; vis = {m}
            while fp.get(n) is not None:
                pn, edge = fp[n]
                if pn in vis: break
                vis.add(pn); fe.append(edge); n = pn
            fe.reverse()
            be = []; n = m; vis2 = {m}
            while bp.get(n) is not None:
                pn, edge = bp[n]
                if pn in vis2: break
                vis2.add(pn); be.append(edge); n = pn
            ae = fe + be
            if not ae or len(ae) > max_len: continue
            pk = tuple((h,p,t) for h,p,t,s,c in ae)
            if pk in seen: continue
            seen.add(pk)
            pn = [source]
            for h,p,t,s,c in ae:
                nx = t if t != pn[-1] else h
                if nx != pn[-1]: pn.append(nx)
            sc = 1.0
            for i,(h,p,t,s,c) in enumerate(ae):
                sc *= max(c,0.01) * self.REL_WEIGHTS.get(p, self.DEFAULT_REL_WEIGHT) * (0.9**(i+1))
            # Hub penalty: downgrade paths through high-degree nodes
            for node in pn[1:-1]:  # intermediate nodes only, not source/target
                degree = len(self.adj.get(node, [])) + len(self.adj_rev.get(node, []))
                if degree > 10:
                    sc /= math.log2(degree + 1)
            # Geometric mean normalization: prevents longer paths from being
            # automatically penalized by multiplicative score collapse.
            # A strong 4-hop path now competes fairly with a weak 1-hop path.
            if len(ae) > 1:
                sc = abs(sc) ** (1.0 / len(ae))
            found.append((pn, ae, sc))
        found.sort(key=lambda x: x[2], reverse=True)
        return found[:max_paths]

    def find_paths_multiseed(self, source, target, max_len=MAX_PATH_LEN, max_paths=MAX_PATHS):
        ap = self.find_paths_bidirectional(source, target, max_len, max_paths)
        if len(ap) >= max_paths: return sorted(ap, key=lambda x:x[2], reverse=True)[:max_paths]
        tid = self.entity2id.get(target)
        if tid is not None:
            for nb,_,_ in self.embedding_neighbors(tid, MULTI_SEED_K):
                ap.extend(self.find_paths_bidirectional(source, nb, max_len, 3))
        sid = self.entity2id.get(source)
        if sid is not None and len(ap) < max_paths:
            for nb,_,_ in self.embedding_neighbors(sid, MULTI_SEED_K):
                ap.extend(self.find_paths_bidirectional(nb, target, max_len, 3))
        ap.sort(key=lambda x: x[2], reverse=True)
        return ap[:max_paths]

class CincinnatusEngine:
    def __init__(self, data_dir=DATA_DIR):
        self.graph = CincinnatusGraph(data_dir)

    def load(self):
        print("="*60); print("CINCINNATUS — Loading Knowledge Graph"); print("="*60)
        self.graph.load()
        print("="*60)
        print(f"Ready. {len(self.graph.entity2id):,} entities, "
              f"{len(self.graph.rel2id)} relations, embeddings {self.graph.embeddings.shape}")
        print("="*60)

    def _best_of_n_extract(self, question, n=BEST_OF_N):
        prompt = (f'Extract the key scientific entities from this question. '
                  f'Return ONLY a JSON list of entity name strings, nothing else.\n\n'
                  f'Question: {question}\n\nEntities:')
        best = []; best_c = -1
        for attempt in range(n):
            ents = ollama_json(prompt, temperature=0.2+attempt*0.1)
            mc = sum(1 for t in ents if self.graph.find_entities([t]))
            if mc > best_c: best_c = mc; best = ents
            if mc >= len(ents) and mc >= 2: break
        return best

    def _graph_guided_explore(self, question, source_entity, max_depth=2, max_branches=10):
        """LLM-guided graph traversal (Amayuelas Agent pattern, Section 4.2.1).

        Instead of asking the LLM to guess entity names, we show it the ACTUAL
        neighbors from the graph and let it decide which branches are relevant
        to the question. This eliminates entity matching errors because every
        entity comes directly from the adjacency list.

        Algorithm (adapted from Amayuelas Algorithm 1, Appendix B):
          1. Get real neighbors of source entity from graph
          2. Group neighbors by relation type
          3. Ask LLM: "Which relations are relevant to this question?"
          4. For selected relations, show neighbor entities to LLM
          5. Ask LLM: "Which neighbors are relevant?"
          6. Record selected triples as provenance
          7. Optionally recurse on selected neighbors (depth 2)

        Reference: Amayuelas et al. (2025) "Grounding LLM Reasoning with
        Knowledge Graphs", Section 4.2.1 Agent, Algorithm 1.
        """
        found_triples = []
        visited = {source_entity}
        frontier = [source_entity]

        for depth in range(max_depth):
            next_frontier = []
            for entity in frontier:
                edges = self.graph.adj.get(entity, [])
                if not edges:
                    continue

                # Group neighbors by relation type
                rel_groups = defaultdict(list)
                for pred, target, src_db, conf in edges:
                    rel_groups[pred].append((target, src_db, conf))

                # Step 1: LLM prunes relations
                rel_list = [f"{r} ({len(targets)} neighbors)" for r, targets in rel_groups.items()]
                if not rel_list:
                    continue

                prune_prompt = (
                    f'Question: "{question}"\n'
                    f'Entity: {entity}\n'
                    f'This entity has the following relation types:\n'
                    f'{chr(10).join(f"  - {r}" for r in rel_list)}\n\n'
                    f'Which relations are relevant to answering the question? '
                    f'Return ONLY a JSON list of relation names. '
                    f'Select at most 5 most relevant relations.\n\n'
                    f'Relevant relations:'
                )
                selected_rels = ollama_json(prune_prompt, temperature=0.1)

                # Match LLM selections to actual relation names
                # Priority: exact > case-insensitive exact > substring (with length guard)
                matched_rels = []
                for sel in selected_rels:
                    sel_lower = sel.lower().strip()
                    # Exact match first
                    if sel_lower in [r.lower() for r in rel_groups]:
                        for actual_rel in rel_groups:
                            if sel_lower == actual_rel.lower() and actual_rel not in matched_rels:
                                matched_rels.append(actual_rel)
                                break
                    else:
                        # Substring match only if selection is >6 chars
                        if len(sel_lower) > 6:
                            for actual_rel in rel_groups:
                                if sel_lower in actual_rel.lower() or actual_rel.lower() in sel_lower:
                                    if actual_rel not in matched_rels:
                                        matched_rels.append(actual_rel)
                                        break
                # Fallback: if LLM matched nothing, use high-weight relations
                if not matched_rels:
                    causal = [r for r in rel_groups if r in self.graph.REL_WEIGHTS and self.graph.REL_WEIGHTS[r] >= 0.7]
                    matched_rels = causal[:5] if causal else list(rel_groups.keys())[:3]

                # Step 2: For each selected relation, LLM prunes entities
                for rel in matched_rels[:5]:
                    targets = rel_groups[rel]
                    # If too many neighbors, sample for LLM
                    if len(targets) > 30:
                        # Show first 30, sorted by confidence
                        targets_show = sorted(targets, key=lambda x: x[2], reverse=True)[:30]
                    else:
                        targets_show = targets

                    target_names = [t[0] for t in targets_show]
                    target_str = ", ".join(target_names[:20])

                    entity_prompt = (
                        f'Question: "{question}"\n'
                        f'Entity: {entity}\n'
                        f'Relation: {rel}\n'
                        f'Neighbors via this relation: {target_str}\n\n'
                        f'Which of these neighbors are most relevant to answering '
                        f'the question? Return ONLY a JSON list of entity names. '
                        f'Select at most {max_branches} most relevant.\n\n'
                        f'Relevant neighbors:'
                    )
                    selected_entities = ollama_json(entity_prompt, temperature=0.1)

                    # Match LLM selections to actual neighbor names
                    # Priority: exact match > case-insensitive exact > substring (short names only)
                    for sel in selected_entities[:max_branches]:
                        sel_clean = sel.strip()
                        sel_lower = sel_clean.lower()
                        best_match = None
                        # Pass 1: exact match
                        for target_name, src_db, conf in targets_show:
                            if sel_clean == target_name:
                                best_match = (target_name, src_db, conf)
                                break
                        # Pass 2: case-insensitive exact
                        if not best_match:
                            for target_name, src_db, conf in targets_show:
                                if sel_lower == target_name.lower():
                                    best_match = (target_name, src_db, conf)
                                    break
                        # Pass 3: substring only if selection is >5 chars (prevents "regulate" matching "upregulates")
                        if not best_match and len(sel_lower) > 5:
                            for target_name, src_db, conf in targets_show:
                                if sel_lower in target_name.lower() or target_name.lower() in sel_lower:
                                    best_match = (target_name, src_db, conf)
                                    break
                        if best_match:
                            target_name, src_db, conf = best_match
                            triple = (entity, rel, target_name, src_db, conf)
                            if triple not in found_triples:
                                found_triples.append(triple)
                            if target_name not in visited and depth < max_depth - 1:
                                next_frontier.append(target_name)
                                visited.add(target_name)

            frontier = next_frontier[:10]  # Cap frontier to avoid explosion

            # Ask LLM if we have enough to answer
            if found_triples and depth < max_depth - 1:
                triple_str = "\n".join(f"  {h} --[{r}]--> {t} [source: {s}]"
                                       for h,r,t,s,c in found_triples)
                end_prompt = (
                    f'Question: "{question}"\n'
                    f'Found triples:\n{triple_str}\n\n'
                    f'Is this sufficient to answer the question? '
                    f'Answer ONLY "Yes" or "No".'
                )
                end_check = ollama_generate(end_prompt, temperature=0.1, max_tokens=10)
                if "yes" in end_check.lower():
                    break

        return found_triples

    def answer(self, question):
        print(f"\n{'='*60}\nQ: {question}\n{'='*60}")

        # ── Step 1: Entity extraction (Best-of-N) ──
        print(f"\n>>> Step 1: Entity extraction (Mistral, best-of-{BEST_OF_N})...")
        t0 = time.time()
        entities = self._best_of_n_extract(question)
        print(f"  Extracted: {entities} ({time.time()-t0:.1f}s)")

        # ── Step 2: Entity matching ──
        print("\n>>> Step 2: Entity matching...")
        matched = {}
        for term in entities:
            ms = self.graph.find_entities([term])
            if ms:
                matched[term] = (ms[0][0], ms[0][1])
                print(f"  '{term}' -> {ms[0][0]} (ID: {ms[0][1]})")
            else:
                print(f"  '{term}' -> NO MATCH")
        # Fallback: if fewer than 2 matches, try splitting question words
        if len(matched) < 2:
            stops = {"what","how","does","the","and","with","from","that","this",
                     "which","where","when","connects","connect","between","about",
                     "into","through"}
            for w in question.lower().split():
                if len(w)>4 and w not in stops and w not in [t.lower() for t in matched]:
                    ex = self.graph.find_entities([w])
                    if ex:
                        matched[w] = (ex[0][0], ex[0][1])
                        print(f"  '{w}' (fallback) -> {ex[0][0]} (ID: {ex[0][1]})")

        # ── Step 2b: Alias resolution ──
        print("\n>>> Step 2b: Alias resolution...")
        resolved = {}
        for term, (name, eid) in matched.items():
            nd = len(self.graph.adj.get(name, []))
            if nd < 50:
                aliases = self.graph.resolve_aliases(name)
                if aliases and aliases[0][2] > nd:
                    cn, ci, ce = aliases[0]
                    print(f"  {name} ({nd} edges) -> {cn} ({ce} edges)")
                    resolved[term] = (cn, ci); continue
            resolved[term] = (name, eid)
        matched = resolved
        if len(matched) < 1: return "Could not match any entities."

        # ── Step 3: Graph-guided exploration (Amayuelas Agent pattern) ──
        # Instead of asking Mistral to guess entity names, we show it the ACTUAL
        # neighbors from the graph and let it decide which branches to follow.
        # This eliminates entity matching errors — every entity comes from the graph.
        print(f"\n>>> Step 3: Graph-guided exploration...")
        t0 = time.time()
        explored_triples = []
        # Explore from each resolved primary entity
        for term in entities:
            if term in matched:
                src_name, src_id = matched[term]
                print(f"  Exploring from: {src_name}...")
                triples = self._graph_guided_explore(question, src_name, max_depth=2, max_branches=5)
                explored_triples.extend(triples)
                print(f"    Found {len(triples)} relevant triples")
        print(f"  Total explored triples: {len(explored_triples)} ({time.time()-t0:.1f}s)")

        # Collect explored entities as additional search targets
        explored_entities = set()
        for h, r, t_ent, s, c in explored_triples:
            explored_entities.add(t_ent)
        # Add explored entities to matched dict for path search
        for ent_name in explored_entities:
            if ent_name not in [v[0] for v in matched.values()]:
                eid = self.graph.entity2id.get(ent_name, -1)
                if eid >= 0:
                    matched[f"_explored_{ent_name}"] = (ent_name, eid)

        # ── Step 4: Path search ──
        # Search paths from primary entities to explored targets only
        # (not between every pair of expanded entities)
        print("\n>>> Step 4: Path search (bidirectional + multi-seed)...")
        all_paths = []

        # Get primary entity names
        primary_resolved = [(matched[t][0], matched[t][1]) for t in entities if t in matched]

        # Search from each primary entity to each explored target
        for src_name, src_id in primary_resolved:
            for tgt_name in list(explored_entities)[:15]:  # Cap at 15 targets
                if tgt_name == src_name:
                    continue
                print(f"  Searching: {src_name} -> {tgt_name}...", end=" ", flush=True)
                t0 = time.time()
                paths = self.graph.find_paths_multiseed(src_name, tgt_name, MAX_PATH_LEN, MAX_PATHS)
                print(f"{len(paths)} paths ({time.time()-t0:.1f}s)")
                all_paths.extend(paths)

        # Also search between primary entities if more than one
        if len(primary_resolved) >= 2:
            for i in range(len(primary_resolved)):
                for j in range(i+1, len(primary_resolved)):
                    sn, si = primary_resolved[i]; tn, ti = primary_resolved[j]
                    print(f"  Searching: {sn} -> {tn}...", end=" ", flush=True)
                    t0 = time.time()
                    paths = self.graph.find_paths_multiseed(sn, tn, MAX_PATH_LEN, MAX_PATHS)
                    print(f"{len(paths)} paths ({time.time()-t0:.1f}s)")
                    all_paths.extend(paths)

        # Also add the explored triples directly as 1-hop paths
        for h, r, t_ent, s, c in explored_triples:
            edge = (h, r, t_ent, s, c)
            sc = max(c, 0.01) * self.graph.REL_WEIGHTS.get(r, self.graph.DEFAULT_REL_WEIGHT) * 0.9
            all_paths.append(([h, t_ent], [edge], sc))

        # ── Step 4: Embedding neighbors ──
        print("\n>>> Step 5: Embedding neighbors...")
        nb_ctx = []
        for term, (name, eid) in matched.items():
            nbs = self.graph.embedding_neighbors(eid, 10)
            if nbs:
                print(f"  {name}: {', '.join(f'{n[0]} ({n[2]:.3f})' for n in nbs[:5])}")
                nb_ctx.append((name, nbs[:5]))

        # ── Step 5: Explanation ──
        print(f"\n>>> Step 6: Explanation (Mistral)...")

        # Confidence threshold: filter out paths too weak to present as evidence.
        # A system that admits "insufficient evidence" is more trustworthy than
        # one that always gives an answer regardless of confidence.
        MIN_PATH_CONFIDENCE = 0.01
        if all_paths:
            all_paths.sort(key=lambda x: x[2], reverse=True)
            # Deduplicate: remove paths with identical edge sets
            unique_paths = []; seen_edges = set()
            for pn, pe, ps in all_paths:
                edge_key = tuple((h,p,t) for h,p,t,s,c in pe)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    unique_paths.append((pn, pe, ps))
            all_paths = unique_paths
            # Filter by confidence threshold
            strong_paths = [(pn,pe,ps) for pn,pe,ps in all_paths if ps >= MIN_PATH_CONFIDENCE]
            weak_paths = [(pn,pe,ps) for pn,pe,ps in all_paths if ps < MIN_PATH_CONFIDENCE]
            if strong_paths:
                all_paths = strong_paths
            else:
                # All paths are below threshold — flag as low confidence
                print(f"  WARNING: All {len(all_paths)} paths below confidence threshold ({MIN_PATH_CONFIDENCE})")
                print(f"  Highest path score: {all_paths[0][2]:.4f}")

        if all_paths:

            pt = ""
            for pi,(pn,pe,ps) in enumerate(all_paths[:MAX_EXPLAIN_PATHS]):
                pt += f"\nPath {pi+1} (confidence: {ps:.3f}):\n"
                for h,p,t,s,c in pe:
                    pt += f"  {h} --[{p}]--> {t} [source: {s}, confidence: {c}]\n"
            prompt = (f"You are a scientific reasoning assistant for Cincinnatus, a cross-domain "
                      f"knowledge graph with {len(self.graph.entity2id):,} entities from 56 databases "
                      f"spanning agriculture, ecology, chemistry, genomics, and biomedicine.\n\n"
                      f"A user asked: {question}\n\nThe system found these paths:\n{pt}\n"
                      f"RULES:\n- Only discuss what the graph paths show. Do NOT invent information.\n"
                      f"- Cite the source database for each claim in brackets.\n"
                      f"- If a path seems weak or indirect, say so.\n"
                      f"- Note which connections are direct vs indirect.\n"
                      f"- Keep it concise and scientific.\n\nExplanation:")
        elif nb_ctx:
            nt = ""
            for name, nbs in nb_ctx:
                nt += f"\n{name}: " + ", ".join(f"{n[0]} ({n[2]:.3f})" for n in nbs)
            prompt = (f"You are a scientific reasoning assistant for Cincinnatus ({len(self.graph.entity2id):,} entities).\n\n"
                      f"A user asked: {question}\n\nNo direct paths found. Embedding similarity suggests:\n{nt}\n\n"
                      f"Provide a brief explanation. Note these are similarity-based, not proven paths.\n\nExplanation:")
        else: return "No paths or similar entities found."

        t0 = time.time()
        explanation = ollama_generate(prompt, temperature=0.3, max_tokens=800)
        print(f"  Generated ({time.time()-t0:.1f}s)")

        ans = f"\n{'='*60}\nCINCINNATUS ANSWER\n{'='*60}\n\n{explanation}\n"
        if all_paths:
            # Evidence strength indicator
            top_score = all_paths[0][2] if all_paths else 0
            if top_score >= 0.3:
                strength = "STRONG"
            elif top_score >= 0.1:
                strength = "MODERATE"
            elif top_score >= MIN_PATH_CONFIDENCE:
                strength = "WEAK"
            else:
                strength = "INSUFFICIENT"
            ans += f"\n{'─'*60}\nEVIDENCE STRENGTH: {strength} (top path score: {top_score:.3f})\n"
            ans += f"PROVENANCE ({len(all_paths)} paths, ranked by confidence)\n{'─'*60}\n"
            for pi,(pn,pe,ps) in enumerate(all_paths[:MAX_EXPLAIN_PATHS]):
                ans += f"\nPath {pi+1} [score: {ps:.3f}]:\n"
                for h,p,t,s,c in pe:
                    ans += f"  {h} ──[{p}]──> {t}\n    Source: {s} | Confidence: {c}\n"
        ans += f"\n{'─'*60}\nEntities: {len(self.graph.entity2id):,} | Triples: {sum(len(v) for v in self.graph.adj.values()):,} | Sources: 56 databases\n"
        return ans

def main():
    parser = argparse.ArgumentParser(description="Cincinnatus Scientific Reasoning Engine")
    parser.add_argument("-q","--query", type=str, help="Question to answer")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    args = parser.parse_args()
    engine = CincinnatusEngine(data_dir=args.data_dir); engine.load()
    if args.query:
        print(engine.answer(args.query))
    else:
        print("\nCincinnatus Scientific Reasoning Engine\nType your question, or 'quit' to exit.\n")
        while True:
            try:
                q = input("Q: ").strip()
                if q.lower() in ("quit","exit","q"): break
                if q: print(engine.answer(q))
            except (KeyboardInterrupt, EOFError): print("\nGoodbye."); break

if __name__ == "__main__": main()
