# Cincinnatus

**A cross-domain scientific knowledge graph with released graph embeddings and graph-grounded reasoning — proof-of-concept release.**

Cincinnatus layers knowledge graphs together to support cross-domain discussion, inquiry, and hypothesis generation. It is a tool to help reveal connections that already exist in curated databases.

```
12,331,977 entities | 30,839,288 triples | 33 audited source labels | 41 unique relation values
```

> **Version 1 is a proof-of-concept release, not a finished platform.** It was released to show what is possible and to encourage contributions toward Version 2.

## What it does

Ask a natural language question. The engine traverses the graph using LLM-guided exploration, finds mechanistic paths through curated scientific databases, and returns every edge traced to its source and confidence score.

```
Q: What connects atrazine to endocrine disruption?

EVIDENCE STRENGTH: STRONG (top path score: 0.765)
PROVENANCE (315 paths, ranked by confidence)

Path 1 [score: 0.765]:
  CHEBI:15930 ──[inhibits]──> NCBI_GENE:1588
    Source: CTD-ChemGene | Confidence: 0.85

Path 2 [score: 0.765]:
  CHEBI:15930 ──[activates]──> CHEBI:51380
    Source: CTD-ChemGene | Confidence: 0.85

Path 3 [score: 0.765]:
  CHEBI:15930 ──[activates]──> NCBI_GENE:2494
    Source: CTD-ChemGene | Confidence: 0.85

Path 4 [score: 0.765]:
  CHEBI:15930 ──[downregulates]──> CHEBI:93785
    Source: CTD-ChemGene | Confidence: 0.85

Path 5 [score: 0.765]:
  CHEBI:15930 ──[upregulates]──> NCBI_GENE:1544
    Source: CTD-ChemGene | Confidence: 0.85
```

The engine found that atrazine inhibits androgen-related genes, activates both estrogen receptor subtypes (ERα and ERβ), downregulates CYP3A metabolism enzymes, and upregulates CYP19A1 (aromatase) — the enzyme responsible for estrogen biosynthesis. Every connection traces to CTD-ChemGene.

*These are associations present in curated databases, not proof of causation. Interpretation requires domain expertise.*

## Why it matters

Complex scientific questions — such as the behavior of prions in the environment — require combining knowledge from soil science, genetics, ecology, toxicology, and more. These databases exist in separate silos with different identifiers and formats. No freely available system connects agriculture, ecology, environmental chemistry, and biomedicine at this scale with provenance tracing.

Current AI tools generate plausible-sounding scientific answers with no sources and no traceability. Cincinnatus takes a different approach: the LLM explains what the graph contains, and every claim traces to a specific database with a confidence score. Scientists can verify each step independently.

## Graph statistics

The public release metadata describe integration of 56 source databases across 10 scientific domains. A direct audit of the released edge table found the following:

| Metric | Release metadata | Audited edge table |
|--------|-----------------|-------------------|
| Entities | 12,331,977 | 12,331,977 |
| Triples | 30,839,288 | 30,839,288 |
| Source labels | 56 databases (including 18 Gramene species databases) | 33 unique source labels |
| Relation types | 55 (in rel2id.json) | 41 unique values in edge table |
| Embedding dimensions | 128 | 128 |

The 33 audited source labels were assigned to 11 curated provenance domains. The largest edge-count groups were genomics/proteomics (6,427,093 edges), taxonomy (5,461,100), toxicology (5,019,291), biomedicine (4,945,637), biochemistry (3,541,286), and ecology (3,256,781). A structural proxy based on node sharing found 502,270 of 10,591,298 nodes (4.74%) appeared in more than one domain.

## Embedding results

All 12,331,977 entities were first encoded using [PubMedBERT](https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO) (768d, reduced to 128d via IncrementalPCA). These PubMedBERT embeddings served as initialization for all subsequent KGE (TransE, RotatE, ComplEx) and BuNN training.

BuNN training used Taylor-approximated sheaf diffusion over flat vector bundles. Due to GPU memory constraints (the initial PolyNSD architecture required ~287 GB for 70.6M edges but only 24 GB was available), training used subgraph sampling (500,000 nodes per batch, each node sampled approximately 16 times across the full 12.3M-node graph) with full-graph inference over all 12.3M nodes.

| Model | MRR | Hits@1 | Hits@10 |
|-------|-----|--------|---------|
| TransE (baseline) | 0.4654 | 0.400 | 0.599 |
| BuNN-Chebyshev | 0.3067 | 0.198 | 0.563 |
| **BuNN-Taylor** | **0.5705** | **0.460** | **0.786** |

BuNN-Taylor improves over TransE by 22.6% on MRR. To our knowledge, this is the first application of BuNN sheaf neural networks at this scale — the original BuNN paper (Bamberger et al., ICLR 2025; [arXiv:2405.15540](https://arxiv.org/abs/2405.15540)) evaluated on benchmark graphs orders of magnitude smaller.

However, whether BuNN produces more productive hypotheses than simpler graph methods like TransE — which achieved MRR = 0.4654 on this same graph with far less engineering effort — is an open question that Version 2 should address directly.

Both models were evaluated on the same held-out test set. TransE was trained using standard negative sampling; BuNN-Taylor used 500K-node subgraph sampling with full-graph inference. BuNN-Chebyshev replaced the Taylor heat kernel with Chebyshev spectral filters (PolyNSD-style) under identical conditions — the Taylor kernel was substantially superior, confirming the design choice in the original BuNN paper.

## Domains covered

The knowledge graph integrates databases across ten scientific domains:

- **Agriculture** — AGROVOC, Gramene (18 crop species: arabidopsis, barley, cannabis, coffee, cotton, grape, maize, medicago, pepper, potato, rapeseed, rice, sorghum, soybean, sunflower, tobacco, tomato, wheat), PHI-Base, Planteome, PlantReactome
- **Ecology** — GBIF, GloBI, ENVO, MGnify, SILVA
- **Environmental chemistry** — EPA CompTox, ECOTOX
- **Toxicology** — CTD, CTD-ChemGene, AOP-Wiki
- **Genomics** — Ensembl, Gene Ontology, Expression Atlas, STRING
- **Biomedicine** — PrimeKG, CancerMine, CIViC, CARD, ChEMBL
- **Biochemistry** — ChEBI, BRENDA, KEGG Reactions, HMDB, LOTUS, Rhea, PathwayCommons, UniProt, MIBiG
- **Taxonomy** — NCBI Taxonomy, ITIS
- **Nutrition & plant traits** — FoodDataCentral, TRY
- **Physical sciences** — Periodic Table, CODATA Constants

Database selection was driven by open-source availability and relevance to cross-domain scientific questions, beginning with prion-environment research and broadening to crop species, chemicals, and ecological interactions. Several domains remain thin, particularly soil science, crop genomics, and emerging contaminant research.

## Architecture

```
Question
  → Mistral 7B (Best-of-3 entity extraction via Ollama)
  → Inverted index entity matching + word-level fuzzy matching
  → Alias resolution (e.g., "Atrazine" → CHEBI:15930 with 17,379 edges)
  → LLM-guided graph traversal (Amayuelas Agent pattern)
      - Show LLM the actual graph neighbors
      - LLM prunes relations, then prunes entities
      - Recurse on selected neighbors (depth 2)
  → Bidirectional BFS path search from source to LLM-selected targets
  → Relation-weighted, confidence-scored path ranking
      - Geometric mean normalization (prevents path-length bias)
      - Hub penalty (downgrades high-degree generic nodes)
      - Evidence strength classification (STRONG / MODERATE / WEAK)
  → Mistral 7B (provenance-constrained explanation)
  → Answer with full provenance chain
```

The graph-guided traversal follows the Agent pattern described in Amayuelas et al. (2025), "Grounding LLM Reasoning with Knowledge Graphs." Instead of asking the LLM to guess entity names, we show it the actual neighbors from the graph's adjacency list and let it decide which branches are relevant to the question. This eliminates entity matching errors because every target comes directly from the graph.

Key design decisions:
- **Graph-guided LLM traversal** shows the LLM real neighbors from the graph and lets it prune irrelevant relations and entities at each depth.
- **Bidirectional BFS** searches from both source and target entities simultaneously, meeting in the middle.
- **Best-of-N entity extraction** runs the LLM 3 times and picks the extraction that matches the most graph entities.
- **Multi-seed path search** uses FAISS embedding neighbors as additional search entry points when direct paths are sparse.
- **Alias resolution** follows `has_name` and `synonym_of` edges to find canonical entity IDs (e.g., "atrazine" resolves to CHEBI:15930 which has 17,379 mechanistic connections).
- **Directional relation respect** prevents nonsensical reverse traversal of causal relations (e.g., "causes", "upregulates", "kills" are never traversed backward).
- **Relation weights** rank mechanistic edges (causes=1.0, upregulates=1.0) above metadata edges (has_name=0.1).
- **Evidence strength indicator** classifies results as STRONG (≥0.3), MODERATE (≥0.1), or WEAK based on top path confidence.

## Representative case studies

Release-edge extraction was tested on four compounds. Three showed meaningful connections; one did not.

| Compound | Name matches | Nearby links | Important links |
|----------|-------------|-------------|-----------------|
| Atrazine | 2 | 17,509 | 16,089 |
| Glyphosate | 2 | 5,901 | 4,783 |
| Resveratrol | 2 | 12,948 | 12,707 |
| Delta-9-tetrahydrocannabinol | 0 | 0 | 0 |

**The delta-9-tetrahydrocannabinol failure is instructive.** Cannabis compounds are in the graph — LOTUS added 1,113 *Cannabis sativa* entries, and a path can be traced from *Cannabis sativa* through THC to genes and diseases using three databases. However, a search for "delta-9-tetrahydrocannabinol" found no name matches. This is the same class of failure that occurred earlier in development when atrazine was missed because the search used a capitalized variant not linked to CHEBI:15930. Success depends on matching the exact name or alias present in the graph, and synonym coverage has gaps.

These case studies were derived from release-edge extraction workflows, not from a completed end-to-end reasoning-engine run. An attempt to reproduce the full local reasoning-engine workflow failed during embedding normalization due to memory pressure.

## Local usability benchmarks

On a local Windows-based environment:

| Operation | Time |
|-----------|------|
| Load entity2id.json (12.3M entries) | 29.6–32.2 s |
| Load rel2id.json (55 mappings) | 0.001–0.061 s |
| Load kuzu_edges.parquet (30.8M rows, 3 columns) | 35.0–35.8 s |
| Lookup: "atrazine" (32 matches) | 29.9–59.2 s |
| Lookup: "CYP19A1" (822 matches) | 49.8–53.2 s |
| Lookup: "soil" (3,680 matches) | 62.6–68.0 s |
| Lookup: "plant" (4,927 matches) | 41.7–42.3 s |

These are local release-usability observations, not optimized production benchmarks. The full reasoning engine remains hardware-sensitive — it failed on the test machine during embedding normalization due to memory pressure.

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com/) with Mistral 7B: `ollama pull mistral`
- ~50 GB RAM (for loading the full graph and embeddings)
- GPU recommended but not required (FAISS and Ollama benefit from GPU)

### Data files (not included in this repo due to size)

Place these in `~/agronomic-ai/data/`:

| File | Size | Description |
|------|------|-------------|
| `entity2id.json` | ~403 MB | Entity name → integer ID mapping |
| `rel2id.json` | < 1 MB | Relation name → integer ID mapping |
| `kuzu_edges.parquet` | ~261 MB | All 30.8M triples with source and confidence |
| `sheaf_embeddings.npy` | ~6.3 GB | BuNN-Taylor embeddings (12.3M × 128) |

Data files available on Zenodo: [https://doi.org/10.5281/zenodo.19195636](https://doi.org/10.5281/zenodo.19195636)

## Usage

### Single question
```bash
python cincinnatus_engine.py -q "What connects atrazine to endocrine disruption?"
```

### Interactive mode
```bash
python cincinnatus_engine.py
```

## Limitations

This project was built by a plant scientist, not a computer scientist or mathematician. The author welcomes input from those communities.

- **v1 is a proof-of-concept, not a finished platform.** The reasoning engine uses LLM-guided graph traversal, not learned reasoning. More sophisticated approaches are planned for v2.
- **Entity matching depends on exact name or alias overlap.** The delta-9-tetrahydrocannabinol case study shows that entities present in the graph under canonical identifiers may not be found if the search term doesn't match any name or alias. Embedding-based FAISS search provides a fallback but is not a complete solution.
- **The full reasoning engine is hardware-sensitive.** It failed on the test machine during embedding normalization due to memory pressure. The case studies reported here used release-edge extraction workflows, not end-to-end engine runs.
- **Audited counts differ from release metadata.** The release describes 56 source databases and 55 relation types. Direct audit of the edge table found 33 unique source labels and 41 unique relation values. The discrepancy reflects how Gramene species databases and relation mappings are counted.
- **Confidence scores are not probabilities.** They reflect source database reliability rankings (0.75–1.0) assigned during integration and are useful for ranking but should not be interpreted as statistical confidence measures.
- **Validation was structural, not biological.** File consistency, mapping coverage, and reference integrity were verified. Semantic correctness and biological accuracy of individual edges were not assessed.
- **LLM explanation quality depends on Mistral 7B.** Larger models may produce better explanations. The graph paths and provenance are independent of the LLM.
- **Load time is approximately 3 minutes** on first startup due to the size of the graph and embeddings.
- **The graph is a snapshot.** Source databases are updated independently. The integrated graph reflects the state of sources at the time of construction.
- **LLM-guided traversal introduces variability.** Results may differ slightly between runs because Mistral selects which graph branches to explore. The underlying graph and paths are stable; the selection of which paths to surface is not.
- **Several domains remain thin**, particularly soil science, crop genomics, and emerging contaminant research.

## Roadmap (v2)

Three areas will drive Version 2 development:

**1. Expanded graph coverage**
- Additional databases to fill identified coverage gaps (soil science, crop genomics, emerging contaminants)
- Expanded alias and synonym coverage to reduce entity resolution failures
- Literature extraction overlays (PubMed, OpenAlex) to capture knowledge in text but not yet in structured databases

**2. Richer embeddings and reasoning**
- Domain-specific initialization strategies
- Better handling of heterophilic edges through sheaf neural network architectures
- Provenance-weighted confidence scoring for multihop reasoning
- Direct comparison of whether BuNN embeddings produce more productive hypotheses than simpler methods like TransE
- Contrastive learning for negative statements and path-based reasoning

**3. Infrastructure**
- Evaluation framework with benchmark questions
- Kùzu native graph database backend (faster traversal)
- Access to larger GPU resources for expanded training
- Web interface

**What databases, features, or domains would help your research?** Open an issue or contact the author.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where help is especially valued:
- Adding new source databases
- Improving entity resolution and synonym coverage
- Testing with domain-specific queries
- Algorithmic improvements from computer science / mathematics community
- Documentation and tutorials

## Citation

If you use Cincinnatus in your research, please cite:

```bibtex
@software{feiss2026cincinnatus,
  author = {Feiss IV, Richard A.},
  title = {Cincinnatus: Graph-Grounded AI for Science},
  year = {2026},
  url = {https://github.com/RFeissIV/cincinnatus}
}
```

## Acknowledgments

- **Minnesota Center for Prion Research and Outreach (MNPRO)**, University of Minnesota — computing resources
- **Amazon Web Services** — GPU rental (~$70 USD total)
- **PubMedBERT**: pritamdeka/S-PubMedBert-MS-MARCO — semantic initialization for all 12.3M entity embeddings
- **Bundle Neural Networks**: Bamberger, Barbero, Dong & Bronstein (ICLR 2025; [arXiv:2405.15540](https://arxiv.org/abs/2405.15540))
- **Sheaf neural network survey**: arXiv:2502.15476v1 — Open Problem 7 (scaling beyond 1M nodes)
- **Graph-grounded LLM reasoning**: Amayuelas et al. (2025), "Grounding LLM Reasoning with Knowledge Graphs"
- **Source databases**: CTD, ChEBI, KEGG, PrimeKG, AGROVOC, ECOTOX, PathwayCommons, and all other integrated databases retain their original licenses and attribution requirements
- AI coding assistants (Claude, GPT) were used extensively for code development, debugging, and architecture refinement. The author directed the design, selected databases, defined the scientific scope, and validated results.
- Hardware: AWS EC2 g6e.2xlarge instance, Lenovo ThinkPad X1, 4TB external hard drive

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

Source databases retain their original licenses. Users are responsible for complying with the terms of individual data sources.

## Author

**Richard A. Feiss IV, Ph.D.**
Postdoctoral Researcher, Minnesota Center for Prion Research and Outreach (MNPRO)
University of Minnesota

*"Built from a plant science background to connect siloed research domains. This is v1 — a proof of concept. Input and contributions welcome for v2."*

## Name

Named after the Roman farmer-general who brought order to chaos.
