# Cincinnatus

**Graph-grounded AI for science — 12.3M-entity cross-domain knowledge graph with BuNN embeddings and provenance-traceable paths.**

Cincinnatus layers knowledge graphs together to increase cross-domain discussion, inquiry, and hypothesis generation. It is a tool to help reveal information that was already there.

```
12,331,977 entities | 30,839,288 triples | 56 source databases | 10 scientific domains | 55 relation types
```

## What it does

Ask a natural language question. The engine traverses the graph using LLM-guided exploration, finds mechanistic paths through 56 scientific databases, and returns every edge traced to its source and confidence score.

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

When asking complex questions — such as the behavior of prions in the environment — so many different variables have to be taken into account, from soil type to prion species to plant genotype to susceptible hosts, herbicides, pesticides, microplastics, and everything in between. The siloing of information across separate databases prevents us from seeing the full picture.

Current AI tools generate plausible-sounding scientific answers with no sources and no traceability. Cincinnatus takes a different approach: the LLM explains what the graph proves, and every claim traces to a specific database with a confidence score. Scientists can verify each step independently.

## Graph statistics

| Metric | Value |
|--------|-------|
| Entities | 12,331,977 |
| Triples | 30,839,288 |
| Source databases | 56 (40 unique sources; 18 Gramene species databases) |
| Scientific domains | 10 |
| Relation types | 55 |
| Embedding dimensions | 128 |

## Embedding results

Trained using Bundle Neural Networks (BuNN) with Taylor-approximated sheaf diffusion. Due to GPU memory constraints, training used subgraph sampling (500,000 nodes per batch, each node sampled approximately 16 times across the full 12.3M-node graph).

| Model | MRR | Hits@1 | Hits@10 |
|-------|-----|--------|---------|
| TransE (baseline) | 0.4654 | 0.400 | 0.599 |
| BuNN-Chebyshev | 0.3067 | 0.198 | 0.563 |
| **BuNN-Taylor** | **0.5705** | **0.460** | **0.786** |

BuNN-Taylor improves over TransE by 22.6% on MRR. To our knowledge, this is the first application of BuNN sheaf neural networks at this scale — the original BuNN paper (Gebhart & Schrater, 2025; arXiv:2502.15476v1) evaluated on benchmark graphs orders of magnitude smaller.
Both models were evaluated on the same held-out test set. TransE was trained using standard negative sampling; BuNN-Taylor used 500K-node subgraph sampling with approximately 16 passes over the full graph.

## Domains covered

The knowledge graph integrates 56 source databases across ten scientific domains:

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

Database selection was driven by open-source availability and relevance to cross-domain scientific questions, beginning with prion-environment research and broadening to crop species, chemicals, and ecological interactions.

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

*Data files will be made available on Zenodo/Figshare. Contact the author for early access.*

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

This project was built by a plant scientist, not a computer scientist or mathematician. The author is neither, and welcomes input from those communities.

- **v1 is a foundation.** The reasoning engine uses LLM-guided graph traversal, not learned reasoning. More sophisticated approaches (neural path scoring, learned traversal policies) are planned for v2.
- **Entity matching depends on name overlap.** Entities not present in the graph by name or synonym will not be found. FAISS embedding search provides a fallback.
- **LLM explanation quality depends on Mistral 7B.** Larger models may produce better explanations. The graph paths and provenance are independent of the LLM.
- **Load time is approximately 3 minutes** on first startup due to the size of the graph and embeddings.
- **The graph is a snapshot.** Source databases are updated independently. The integrated graph reflects the state of sources at the time of construction.
- **Confidence scores** were developed during the graph integration process and reflect edge reliability based on source database quality and crosswalk confidence. Users should verify important findings against the original source databases.
- **LLM-guided traversal introduces variability.** Results may differ slightly between runs because Mistral selects which graph branches to explore. The underlying graph and paths are stable; the selection of which paths to surface is not.

## Roadmap (v2)

We are actively developing v2 based on community feedback. Planned features:

- [ ] Additional databases to fill identified coverage gaps
- [ ] Literature extraction overlays (PubMed, OpenAlex)
- [ ] Neural path scoring (learned edge weights)
- [ ] Evaluation framework with benchmark questions
- [ ] Kùzu native graph database backend (faster traversal)
- [ ] Access to larger GPU resources for expanded training
- [ ] Web interface

**What databases, features, or domains would help your research?** Open an issue or contact the author. As a non-computer scientist and non-mathematician, input from those communities is especially valued.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where help is especially valued:
- Adding new source databases
- Improving entity resolution
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
- **Amazon Web Services** — GPU rental
- **Bundle Neural Networks**: Gebhart & Schrater (2025), arXiv:2502.15476v1
- **Graph-grounded LLM reasoning**: Amayuelas et al. (2025), "Grounding LLM Reasoning with Knowledge Graphs"
- **Knowledge graph embedding injection**: Coppolillo (2025), arXiv:2505.07554v1 (methodology reference for v2)
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

*"Built from a plant science background to connect siloed research domains. This is v1 — input and contributions welcome for v2."*

## Name

Named after the Roman farmer-general who brought order to chaos.
