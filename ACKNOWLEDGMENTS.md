# Acknowledgments & Influences

This work stands on the shoulders of giants. While all errors and limitations are mine alone, the intellectual foundations come from researchers, educators, and practitioners who generously share their knowledge.

---

## Direct Influences

### Category Theory & Applied Mathematics

**David Spivak** (MIT) — His book *Category Theory for Scientists* (2014) and subsequent work on applied category theory provided the mathematical foundation for the conjecture engine. The idea that category theory could be a "general theory of systems" directly inspired the oracle architecture.

**Urs Schreiber** (NYU Abu Dhabi, nLab) — The nLab wiki and his work on higher category theory, homotopy type theory, and mathematical physics shaped my understanding of how categorical structures capture natural patterns. His "Know Time" podcast discussions on the deep connections between physics and category theory influenced the project's philosophical framework.

**Bruno Gavranović** (Symbolica AI) — His research on categorical deep learning and compositional approaches to AI demonstrated that category theory isn't just abstract mathematics — it's a practical tool for building ML systems. His work on string diagrams and functorial learning directly influenced the strategy design.

**Paul Lessard** — His contributions to the MLST (Machine Learning Street Talk) discussions on category theory and its applications to modern AI provided accessible bridges between abstract mathematics and practical implementation.

### Systems Thinking & AI Architecture

**Eric Daimler** (*Building Better Systems* podcast) — His discussions on systems-level thinking, emergence, and how to build robust AI architectures shaped my approach to combining multiple inference strategies. The emphasis on compositional systems rather than monolithic models directly influenced the oracle's voting mechanism.

**MLST Podcast Community** — The Machine Learning Street Talk discussions on category theory, large language models, and emergent intelligence provided crucial context for understanding how modern AI systems (like ESM-2) learn structured patterns from unstructured data.

### Protein Science & AlphaFold

**Demis Hassabis** (Google DeepMind) — His Davos 2026 conjecture that "any natural pattern can be efficiently modeled by classical learning algorithms" is the explicit hypothesis this work tests. AlphaFold's success demonstrated that evolutionary patterns are learnable — KOMPOSOS-III extends this to functional interactions.

**The AlphaFold Team** (Jumper et al., 2021) — While this work uses ESM-2 rather than AlphaFold directly, the AlphaFold breakthrough proved that protein structure is a learnable pattern. The question KOMPOSOS-III asks is: *What else can we learn from protein sequences beyond structure?*

**The ESM-2 Team** (Lin et al., 2023, Meta AI) — ESM-2's demonstration that protein language models capture functional information from sequence alone made this entire project possible. Their decision to open-source the model enabled independent research like this.

---

## Intellectual Lineage

This project sits at the intersection of three traditions:

1. **Applied Category Theory** (Spivak, Fong, Spivak, Gavranović) — The mathematics of composition and universal properties
2. **Protein Language Models** (ESM-2, ProtBERT, AlphaFold) — Neural networks learning evolutionary constraints
3. **Knowledge Discovery Systems** (DeepMind, OpenAI, Anthropic) — AI systems that generate novel hypotheses

The core insight — that category-theoretic inference strategies could systematically explore embedding spaces to discover novel biology — emerged from synthesizing these three traditions.

---

## Technical Foundations

**PyTorch & fair-esm** — Meta AI's ESM-2 implementation made biological embeddings accessible
**sentence-transformers** — Hugging Face's library enabled the text embedding baseline
**STRING Database** — Szklarczyk et al.'s comprehensive PPI database provided training data
**SQLite** — D. Richard Hipp's elegant database system enabled efficient knowledge graph storage

---

## Development Tools

**Claude Code (Anthropic)** — Used as a development assistant for debugging, documentation, and code review. All core algorithms, mathematical frameworks, and scientific design decisions are original work. Claude helped write cleaner code faster, not think the thoughts.

**Python Scientific Stack** — NumPy, SciPy, Pandas — the foundational tools that make computational science accessible.

---

## A Note on Attribution

I started coding seriously 6 months ago (July 2025). Everything you see here — from the categorical inference strategies to the ESM-2 integration — was learned through reading papers, watching lectures, and building.

The ideas aren't mine. The synthesis is.

If you recognize your work reflected here and I've failed to cite it properly, please let me know (jhawk314@gmail.com). Attribution matters.

---

## References & Further Reading

### Category Theory
- Spivak, D. I. (2014). *Category Theory for Scientists*. MIT Press.
- Fong, B., & Spivak, D. I. (2019). *An Invitation to Applied Category Theory*. Cambridge University Press.
- Milewski, B. (2018). *Category Theory for Programmers*. (Free online book)
- nLab: https://ncatlab.org (The comprehensive wiki for higher category theory)

### Protein Language Models
- Lin, Z., et al. (2023). "Evolutionary-scale prediction of atomic-level protein structure with a language model." *Science*, 379(6637), 1123-1130.
- Jumper, J., et al. (2021). "Highly accurate protein structure prediction with AlphaFold." *Nature*, 596, 583-589.
- Rives, A., et al. (2021). "Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences." *PNAS*, 118(15).

### Applied Category Theory in AI
- Gavranović, B. (2023). "Compositional Deep Learning." arXiv:2301.02851
- Shiebler, D., Gavranović, B., & Wilson, P. (2021). "Category Theory in Machine Learning." arXiv:2106.07032

### Podcasts & Lectures
- *Machine Learning Street Talk* (MLST) — Category theory episodes
- *Building Better Systems* with Eric Daimler — Systems thinking in AI
- *Lex Fridman Podcast #241* — Demis Hassabis on AlphaFold and AI for science

---

## License

While the code is Apache 2.0 licensed (see LICENSE), the intellectual debts acknowledged here are immeasurable. If this work contributes anything useful, credit belongs to the community that made it possible.

---

**James Ray Hawkins**
January 2026
jhawk314@gmail.com
