#!/usr/bin/env python3
"""
KOMPOSOS-III Command Line Interface
====================================

A CLI for interacting with KOMPOSOS-III's categorical knowledge system.

Commands:
    init        Initialize a new corpus directory
    load        Load data from corpus into the store
    query       Query the knowledge graph (paths, equivalences, gaps)
    report      Generate detailed markdown reports
    oracle      Run Oracle predictions between two concepts
    homotopy    Analyze path homotopy (are paths equivalent?)
    predict     Make a single prediction about a relationship
    stress-test Run quality stress tests on the Oracle system
    stats       Show store statistics
    embed       Compute embeddings for all objects

Usage:
    python cli.py init [--corpus PATH]
    python cli.py load [--corpus PATH] [--db PATH]
    python cli.py query evolution "Newton" "Dirac"
    python cli.py query equivalence "WaveMechanics" "MatrixMechanics"
    python cli.py query gaps --threshold 0.3
    python cli.py report evolution "Newton" "Dirac" --output report.md
    python cli.py report full --output full_report.md
    python cli.py oracle "Planck" "Feynman"
    python cli.py homotopy "Planck" "Feynman"
    python cli.py predict "Newton" "Einstein"
    python cli.py stress-test
    python cli.py stats
    python cli.py embed
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data import (
    create_store, create_memory_store, KomposOSStore,
    EmbeddingsEngine, StoreEmbedder,
    CorpusLoader, CustomDataLoader,
    KomposOSConfig, get_config, init_corpus, verify_corpus,
    StoredObject, StoredMorphism, EquivalenceClass
)

# Import the enhanced Oracle (optional - gracefully degrades if not available)
try:
    from oracle import CategoricalOracle, OracleResult
    ORACLE_AVAILABLE = True
except ImportError:
    ORACLE_AVAILABLE = False
    CategoricalOracle = None

# Import path homotopy checking
try:
    from hott import check_path_homotopy, HomotopyResult
    HOMOTOPY_AVAILABLE = True
except ImportError:
    HOMOTOPY_AVAILABLE = False
    check_path_homotopy = None

# Import geometric homotopy (Thurston-aware)
try:
    from hott import check_geometric_homotopy, GeometricHomotopyResult
    GEOMETRIC_HOMOTOPY_AVAILABLE = True
except ImportError:
    GEOMETRIC_HOMOTOPY_AVAILABLE = False
    check_geometric_homotopy = None

# Import geometry (Ollivier-Ricci curvature and Ricci flow)
try:
    from geometry import (
        OllivierRicciCurvature, compute_graph_curvature, CurvatureResult,
        DiscreteRicciFlow, run_ricci_flow, DecompositionResult
    )
    GEOMETRY_AVAILABLE = True
except ImportError:
    GEOMETRY_AVAILABLE = False
    OllivierRicciCurvature = None
    DiscreteRicciFlow = None


# =============================================================================
# Report Generator - Rich Human-Readable Reports
# =============================================================================

class ReportGenerator:
    """
    Generates rich, human-readable markdown reports from KOMPOSOS-III analysis.

    Inspired by KOMPOSOS-jf's research lab reports, these reports feature:
    - Detailed narrative prose explaining findings
    - Scientific analysis sections with deep interpretation
    - Cross-domain bridges with explanations
    - Oracle/Predictive Intelligence module
    - Sheaf coherence analysis
    - Yoneda-based structural analogies
    - Future research directions

    Report types:
    - Evolution report: traces how A became B with rich analysis
    - Equivalence report: analyzes equivalences with HoTT interpretation
    - Gap report: identifies missing connections with predictions
    - Full report: comprehensive analysis with scientific narrative
    """

    def __init__(self, store: KomposOSStore, embeddings: Optional[EmbeddingsEngine] = None):
        self.store = store
        self.embeddings = embeddings
        self.generated_at = datetime.now()

    def _header(self, title: str, subtitle: str = "") -> str:
        """Generate report header with metadata."""
        stats = self.store.get_statistics()
        header = f"""# {title}

**Generated:** {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}
**Knowledge Base:** {stats['objects']} objects, {stats['morphisms']} morphisms, {stats['equivalences']} equivalences
"""
        if subtitle:
            header += f"**Focus:** {subtitle}\n"

        header += """
---

"""
        return header

    def _section(self, title: str, level: int = 2) -> str:
        """Generate section header."""
        return f"\n{'#' * level} {title}\n\n"

    def _table(self, headers: list, rows: list) -> str:
        """Generate markdown table."""
        if not rows:
            return "*No data available*\n"

        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
        return "\n".join(lines) + "\n"

    def _code_block(self, content: str, lang: str = "") -> str:
        """Generate code block."""
        return f"```{lang}\n{content}\n```\n"

    def _prose(self, text: str) -> str:
        """Format prose paragraph."""
        return text.strip() + "\n\n"

    def _get_object_context(self, obj_name: str) -> dict:
        """Get rich context about an object for prose generation."""
        obj = self.store.get_object(obj_name)
        if not obj:
            return {"name": obj_name, "exists": False}

        # Get incoming and outgoing morphisms
        morphisms = self.store.list_morphisms(limit=10000)
        incoming = [m for m in morphisms if m.target_name == obj_name]
        outgoing = [m for m in morphisms if m.source_name == obj_name]

        return {
            "name": obj_name,
            "exists": True,
            "type": obj.type_name,
            "metadata": obj.metadata,
            "incoming_count": len(incoming),
            "outgoing_count": len(outgoing),
            "incoming": incoming[:5],
            "outgoing": outgoing[:5],
            "era": obj.metadata.get("era", obj.metadata.get("category", "unknown")),
            "birth": obj.metadata.get("birth", ""),
            "death": obj.metadata.get("death", ""),
        }

    def _analyze_path_significance(self, paths: list) -> dict:
        """Analyze paths to extract meaningful patterns."""
        if not paths:
            return {"found": False}

        analysis = {
            "found": True,
            "count": len(paths),
            "min_length": min(p.length for p in paths),
            "max_length": max(p.length for p in paths),
            "convergent": len(paths) > 1,
            "relation_types": set(),
            "intermediaries": set(),
            "eras_traversed": [],
        }

        for path in paths:
            for mor_id in path.morphism_ids:
                mor = self.store.get_morphism(mor_id)
                if mor:
                    analysis["relation_types"].add(mor.name)
                    if mor.source_name not in [paths[0].source_name if hasattr(paths[0], 'source_name') else None]:
                        analysis["intermediaries"].add(mor.source_name)
                    if mor.target_name not in [paths[0].target_name if hasattr(paths[0], 'target_name') else None]:
                        analysis["intermediaries"].add(mor.target_name)

        return analysis

    def _generate_scientific_narrative(self, source_ctx: dict, target_ctx: dict, path_analysis: dict) -> str:
        """Generate a scientific narrative about the evolutionary relationship."""
        narrative = ""

        if not path_analysis["found"]:
            narrative += f"The categorical analysis reveals no direct evolutionary pathway from {source_ctx['name']} to {target_ctx['name']} within the current knowledge graph. "
            narrative += "This absence is itself significant: it may indicate either (a) a genuine historical discontinuity, "
            narrative += "(b) missing intermediate concepts in our data, or (c) an indirect relationship mediated through concepts not yet captured. "
            narrative += "From a categorical perspective, the lack of composed morphisms suggests these objects lie in disconnected components "
            narrative += "or require higher-dimensional paths (2-morphisms) to bridge.\n\n"
            return narrative

        # Analyze the nature of evolution
        if path_analysis["convergent"]:
            narrative += f"The discovery of {path_analysis['count']} distinct evolutionary pathways from {source_ctx['name']} to {target_ctx['name']} "
            narrative += "reveals a pattern of **convergent evolution** in the conceptual landscape. "
            narrative += "Just as biological evolution can produce similar structures through independent lineages, "
            narrative += "intellectual history shows multiple routes leading to the same theoretical destination. "
            narrative += f"These paths range from {path_analysis['min_length']} to {path_analysis['max_length']} steps in length, "
            narrative += "with shorter paths representing more direct intellectual inheritance and longer paths capturing "
            narrative += "the full richness of intermediate developments.\n\n"
        else:
            narrative += f"A single evolutionary pathway connects {source_ctx['name']} to {target_ctx['name']}, "
            narrative += f"spanning {path_analysis['min_length']} morphisms in the category of conceptual evolution. "
            narrative += "This unique path suggests a canonical historical development, though the categorical framework "
            narrative += "reminds us that uniqueness up to isomorphism may still permit equivalent reformulations.\n\n"

        # Discuss relation types found
        if path_analysis["relation_types"]:
            rel_list = list(path_analysis["relation_types"])
            if len(rel_list) == 1:
                narrative += f"The pathway is characterized exclusively by '{rel_list[0]}' relationships, "
                narrative += "indicating a homogeneous mode of knowledge transmission. "
            else:
                narrative += f"The evolutionary paths traverse multiple relationship types: {', '.join(rel_list)}. "
                narrative += "This heterogeneity reflects the complex nature of intellectual development, "
                narrative += "where influence, creation, reformulation, and extension interweave to produce new knowledge.\n\n"

        # Discuss intermediaries
        if path_analysis["intermediaries"]:
            intermediaries = list(path_analysis["intermediaries"])[:5]
            narrative += f"Key intermediary figures and concepts include: {', '.join(intermediaries)}. "
            narrative += "These nodes serve as categorical 'waypoints'—objects through which the composed morphisms factor. "
            narrative += "From a Kan extension perspective, these intermediaries provide the structure needed "
            narrative += "to extend knowledge from source to target domains.\n\n"

        return narrative

    def _generate_layman_summary(self, source: str, target: str, path_analysis: dict, paths: list, predictions: list) -> str:
        """
        Generate a plain-English summary that anyone can understand.

        This method creates an accessible explanation of the findings,
        avoiding technical jargon while preserving the key insights.
        Works for any domain (physics, philosophy, biology, etc.)
        """
        summary = ""

        # Opening - what we were looking for
        summary += "### What We Were Looking For\n\n"
        summary += f"We wanted to understand: **How did {source} lead to {target}?** "
        summary += "In other words, what's the 'story' of how one concept or person influenced another over time?\n\n"

        # Main finding in simple terms
        summary += "### What We Found\n\n"

        if not path_analysis["found"]:
            summary += f"**We couldn't find a direct connection** between {source} and {target} in our data. "
            summary += "This doesn't mean they're unrelated - it might mean:\n"
            summary += "- We're missing some information\n"
            summary += "- The connection is indirect or subtle\n"
            summary += "- This could be a genuinely new discovery opportunity\n\n"
            return summary

        # Found paths - explain them simply
        if path_analysis["count"] == 1:
            summary += f"We found **one clear path** showing how {source} led to {target}. "
            summary += f"This path has **{path_analysis['min_length']} steps** (like a chain with {path_analysis['min_length']} links).\n\n"
        else:
            summary += f"We found **{path_analysis['count']} different paths** showing how {source} led to {target}. "
            summary += f"The shortest path has **{path_analysis['min_length']} steps**, and the longest has **{path_analysis['max_length']} steps**.\n\n"

            summary += "**Why multiple paths matter:** Think of it like getting from your house to a friend's house - "
            summary += "there might be several routes (through downtown, via the highway, through the park). "
            summary += "Having multiple paths means the connection is robust - it's not just one fragile chain of events.\n\n"

        # Explain the spine (essential intermediaries)
        if path_analysis.get("intermediaries"):
            intermediaries = list(path_analysis["intermediaries"])[:5]
            summary += "### The Key Players In Between\n\n"
            summary += f"Every path from {source} to {target} goes through certain important figures/concepts:\n\n"
            for inter in intermediaries:
                summary += f"- **{inter}**\n"
            summary += "\n"
            summary += "These are like 'required stops' on the journey - you can't get from start to finish without going through them. "
            summary += "They represent the essential links in the chain of influence.\n\n"

        # Explain the types of relationships
        if path_analysis.get("relation_types"):
            rel_types = list(path_analysis["relation_types"])
            summary += "### How They're Connected\n\n"
            summary += "The connections we found include these types of relationships:\n\n"

            # Translate relation types to plain English
            rel_explanations = {
                "influenced": "one person/idea shaped another",
                "created": "someone invented or developed something",
                "superseded": "a newer idea replaced an older one",
                "superseded_by": "something was replaced by a newer approach",
                "unified": "someone combined multiple ideas into one",
                "unified_by": "multiple ideas were combined by someone",
                "extended": "someone built upon earlier work",
                "reformulated": "someone expressed the same idea in a new way",
                "collaborated": "people worked together",
                "independently_developed": "people arrived at similar ideas separately",
            }

            for rel in rel_types:
                explanation = rel_explanations.get(rel, f"a '{rel}' relationship")
                summary += f"- **{rel}**: {explanation}\n"
            summary += "\n"

        # Explain what the paths actually show (with example)
        if paths:
            summary += "### Reading the Paths\n\n"
            summary += "Here's how to read the first (shortest) path:\n\n"

            # Get first path details
            first_path = paths[0]
            chain = []
            for mor_id in first_path.morphism_ids:
                mor = self.store.get_morphism(mor_id)
                if mor:
                    if not chain:
                        chain.append(mor.source_name)
                    chain.append(f"--({mor.name})-->")
                    chain.append(mor.target_name)

            if len(chain) >= 3:
                # Show simplified version
                nodes = [chain[i] for i in range(0, len(chain), 2)]
                summary += f"**{' -> '.join(nodes)}**\n\n"
                summary += "This means: "

                # Build narrative
                narrative_parts = []
                for i in range(0, len(chain)-2, 2):
                    from_node = chain[i]
                    relation = chain[i+1].replace("--", "").replace("-->", "").replace("(", "").replace(")", "")
                    to_node = chain[i+2]
                    narrative_parts.append(f"{from_node} *{relation}* {to_node}")

                summary += ", then ".join(narrative_parts) + ".\n\n"

        # If multiple paths - explain what that means
        if path_analysis["count"] > 1:
            summary += "### Why Are There Multiple Paths?\n\n"
            summary += "The different paths represent different 'stories' about how the connection happened:\n\n"

            # Identify what makes paths different
            if paths:
                path_summaries = []
                for i, path in enumerate(paths[:4], 1):
                    nodes = []
                    for mor_id in path.morphism_ids:
                        mor = self.store.get_morphism(mor_id)
                        if mor:
                            if not nodes:
                                nodes.append(mor.source_name)
                            nodes.append(mor.target_name)

                    # Find what's unique about this path
                    path_summaries.append((i, nodes, path.length))

                for i, nodes, length in path_summaries:
                    summary += f"- **Path {i}** ({length} steps): Goes through {' -> '.join(nodes[:4])}{'...' if len(nodes) > 4 else ''}\n"
                summary += "\n"

            summary += "Each path tells a different but valid part of the story. "
            summary += "Some emphasize personal mentorship, others go through specific theories or institutions.\n\n"

        # Predictions in plain terms
        if predictions:
            summary += "### What the System Predicts\n\n"
            summary += "Based on the patterns it found, the system makes these predictions about connections that **might** exist but aren't in our data yet:\n\n"

            for pred in predictions[:3]:
                conf = pred.get('confidence', 0)
                conf_word = "very likely" if conf > 0.7 else "likely" if conf > 0.5 else "possible"
                summary += f"- **{pred.get('prediction', 'Unknown')}** ({conf_word}, {conf:.0%} confidence)\n"
            summary += "\n"
            summary += "These are educated guesses based on patterns - they'd need to be verified by checking historical records.\n\n"

        # Bottom line
        summary += "### The Bottom Line\n\n"
        if path_analysis["found"]:
            if path_analysis["count"] > 1:
                summary += f"**{source} definitely influenced {target}**, and we have {path_analysis['count']} different pieces of evidence for this. "
                summary += "The connection is robust because it shows up through multiple independent routes.\n"
            else:
                summary += f"**{source} influenced {target}** through a clear chain of {path_analysis['min_length']} connections. "
                summary += "While there's only one path, it provides a clear historical narrative.\n"

        return summary

    def _generate_oracle_predictions(self, source: str, target: str, paths: list) -> list:
        """Generate predictions using enhanced CategoricalOracle.

        Uses 8 inference strategies:
        1. KanExtensionStrategy - Categorical Kan extensions
        2. SemanticSimilarityStrategy - Embedding-based similarity
        3. TemporalReasoningStrategy - Temporal metadata analysis
        4. TypeHeuristicStrategy - Type-constrained inference
        5. YonedaPatternStrategy - Morphism pattern matching
        6. CompositionStrategy - Path composition
        7. FibrationLiftStrategy - Cartesian lift predictions
        8. StructuralHoleStrategy - Triangle closure

        Falls back to basic heuristics if Oracle not available.
        """
        # Try enhanced Oracle first
        if ORACLE_AVAILABLE and self.embeddings and self.embeddings.is_available:
            try:
                oracle = CategoricalOracle(self.store, self.embeddings)
                result = oracle.predict(source, target)

                # Convert to report format
                predictions = []
                for pred in result.predictions:
                    predictions.append({
                        "type": pred.strategy_name,
                        "prediction": pred.description,
                        "confidence": pred.confidence,
                        "reason": pred.reasoning,
                    })

                # Add metadata about the Oracle run
                if predictions:
                    predictions[0]["_oracle_metadata"] = {
                        "total_candidates": result.total_candidates,
                        "coherence_score": result.coherence_result.coherence_score,
                        "computation_time_ms": result.computation_time_ms,
                        "strategy_contributions": result.strategy_contributions,
                    }

                return predictions[:20]  # Oracle already limits, but cap for safety

            except Exception as e:
                print(f"Warning: CategoricalOracle failed, falling back to basic heuristics: {e}")

        # Fallback to basic heuristics if Oracle not available
        return self._generate_basic_predictions(source, target, paths)

    def _generate_basic_predictions(self, source: str, target: str, paths: list) -> list:
        """Basic prediction heuristics (fallback when Oracle unavailable)."""
        predictions = []

        # Get all objects for comparison
        all_objects = self.store.list_objects(limit=500)
        morphisms = self.store.list_morphisms(limit=10000)

        # Build morphism index
        outgoing = {}
        incoming = {}
        for mor in morphisms:
            if mor.source_name not in outgoing:
                outgoing[mor.source_name] = []
            outgoing[mor.source_name].append(mor)
            if mor.target_name not in incoming:
                incoming[mor.target_name] = []
            incoming[mor.target_name].append(mor)

        # Prediction 1: Objects with similar outgoing patterns should have similar roles
        source_out = set(m.target_name for m in outgoing.get(source, []))
        for obj in all_objects:
            if obj.name == source:
                continue
            obj_out = set(m.target_name for m in outgoing.get(obj.name, []))
            overlap = len(source_out & obj_out)
            if overlap >= 2:
                predictions.append({
                    "type": "yoneda_analogy",
                    "prediction": f"{obj.name} may have played a role analogous to {source}",
                    "confidence": min(0.95, 0.5 + overlap * 0.1),
                    "reason": f"Shares {overlap} outgoing relationships with {source} (Yoneda lemma: objects defined by their morphisms)"
                })

        # Prediction 2: Missing inverse relationships
        for mor in morphisms:
            if mor.source_name == target and mor.target_name != source:
                # Target influences something else - does source also?
                if mor.target_name not in source_out:
                    predictions.append({
                        "type": "missing_morphism",
                        "prediction": f"{source} may have also {mor.name} {mor.target_name}",
                        "confidence": 0.65,
                        "reason": f"Both {source} and {target} are in evolutionary relationship; {target} has '{mor.name}' to {mor.target_name}"
                    })

        # Prediction 3: Structural gaps suggesting missing intermediaries
        if paths:
            path = paths[0]  # Analyze first path
            for i, mor_id in enumerate(path.morphism_ids[:-1]):
                mor1 = self.store.get_morphism(mor_id)
                mor2 = self.store.get_morphism(path.morphism_ids[i+1])
                if mor1 and mor2:
                    # Large temporal gap?
                    year1 = mor1.metadata.get("year", 0)
                    year2 = mor2.metadata.get("year", 0)
                    if year1 and year2 and abs(year2 - year1) > 50:
                        predictions.append({
                            "type": "temporal_gap",
                            "prediction": f"Missing intermediary between {mor1.target_name} and {mor2.source_name}",
                            "confidence": 0.70,
                            "reason": f"Large temporal gap ({abs(year2-year1)} years) suggests undocumented developments"
                        })

        return predictions[:10]  # Limit to top 10 predictions

    def _generate_yoneda_analysis(self, focus_objects: list) -> str:
        """Generate Yoneda lemma-based structural analysis."""
        analysis = ""

        morphisms = self.store.list_morphisms(limit=10000)

        # Build hom-set representations
        hom_out = {}  # Hom(A, -)
        hom_in = {}   # Hom(-, A)

        for mor in morphisms:
            if mor.source_name not in hom_out:
                hom_out[mor.source_name] = set()
            hom_out[mor.source_name].add((mor.name, mor.target_name))

            if mor.target_name not in hom_in:
                hom_in[mor.target_name] = set()
            hom_in[mor.target_name].add((mor.name, mor.source_name))

        analysis += "### Yoneda Lemma Application\n\n"
        analysis += "The Yoneda lemma tells us that an object is completely determined by its relationships to all other objects. "
        analysis += "By examining the 'representable presheaf' Hom(A, -) for each object A, we can identify structural analogies:\n\n"

        for obj_name in focus_objects[:5]:
            out_rels = hom_out.get(obj_name, set())
            in_rels = hom_in.get(obj_name, set())

            analysis += f"**{obj_name}**:\n"
            analysis += f"- Outgoing morphisms (Hom({obj_name}, -)): {len(out_rels)} relationships\n"
            analysis += f"- Incoming morphisms (Hom(-, {obj_name})): {len(in_rels)} relationships\n"

            # Find structurally similar objects
            similar = []
            for other_name, other_out in hom_out.items():
                if other_name != obj_name:
                    # Compare outgoing relationship types (not targets)
                    out_types = set(r[0] for r in out_rels)
                    other_types = set(r[0] for r in other_out)
                    similarity = len(out_types & other_types) / max(len(out_types | other_types), 1)
                    if similarity > 0.5:
                        similar.append((other_name, similarity))

            if similar:
                similar.sort(key=lambda x: -x[1])
                analysis += f"- Structurally similar to: {', '.join(f'{n} ({s:.0%})' for n, s in similar[:3])}\n"
            analysis += "\n"

        return analysis

    def _compute_coherence_score(self, objects: list) -> dict:
        """Compute sheaf-like coherence across the knowledge graph."""
        coherence = {
            "total_objects": len(objects),
            "connected": 0,
            "isolated": 0,
            "score": 0.0,
            "inconsistencies": []
        }

        morphisms = self.store.list_morphisms(limit=10000)

        # Build connectivity map
        connected_objects = set()
        for mor in morphisms:
            connected_objects.add(mor.source_name)
            connected_objects.add(mor.target_name)

        for obj in objects:
            if obj.name in connected_objects:
                coherence["connected"] += 1
            else:
                coherence["isolated"] += 1
                coherence["inconsistencies"].append(f"{obj.name} is isolated (no morphisms)")

        # Check for type consistency
        type_relations = {}  # {(type1, type2): [relation_types]}
        for mor in morphisms:
            src_obj = self.store.get_object(mor.source_name)
            tgt_obj = self.store.get_object(mor.target_name)
            if src_obj and tgt_obj:
                key = (src_obj.type_name, tgt_obj.type_name)
                if key not in type_relations:
                    type_relations[key] = set()
                type_relations[key].add(mor.name)

        # Check for unusual patterns
        for (type1, type2), rels in type_relations.items():
            if len(rels) > 5:
                coherence["inconsistencies"].append(
                    f"{type1} -> {type2}: {len(rels)} different relation types may indicate inconsistent categorization"
                )

        coherence["score"] = coherence["connected"] / max(coherence["total_objects"], 1)

        return coherence

    def evolution_report(self, source: str, target: str, max_paths: int = 10) -> str:
        """
        Generate a rich evolution report: how did source become target?

        This is the core KOMPOSOS-III use case: tracing the phylogenetics of concepts
        with deep scientific analysis and predictive intelligence.
        """
        report = self._header(
            f"KOMPOSOS-III Evolution Analysis: {source} to {target}",
            f"Tracing the conceptual phylogeny from {source} to {target}"
        )

        # =====================================================================
        # Executive Summary
        # =====================================================================
        report += self._section("Executive Summary")

        # Get context and paths
        source_ctx = self._get_object_context(source)
        target_ctx = self._get_object_context(target)
        paths = self.store.find_paths(source, target, max_length=8)
        path_analysis = self._analyze_path_significance(paths)

        # Generate summary statistics
        stats = self.store.get_statistics()
        report += f"This evolution analysis examined **{stats['objects']} objects** and **{stats['morphisms']} morphisms** "
        report += f"to trace how the concept of **{source}** evolved into **{target}**.\n\n"

        if path_analysis["found"]:
            report += f"**Key Findings:**\n"
            report += f"- **{path_analysis['count']} evolutionary pathway(s)** discovered\n"
            report += f"- **Path lengths:** {path_analysis['min_length']} to {path_analysis['max_length']} steps\n"
            report += f"- **Convergent evolution:** {'Yes' if path_analysis['convergent'] else 'No'}\n"
            report += f"- **Relationship types involved:** {', '.join(path_analysis['relation_types'])}\n\n"
        else:
            report += "**Finding:** No direct evolutionary pathway found in the current knowledge graph.\n\n"

        # =====================================================================
        # Scientific Analysis
        # =====================================================================
        report += self._section("Scientific Analysis")
        report += self._generate_scientific_narrative(source_ctx, target_ctx, path_analysis)

        # Add context about the objects
        if source_ctx["exists"]:
            era_info = f" ({source_ctx['era']} era)" if source_ctx.get('era') and source_ctx['era'] != 'unknown' else ""
            birth_info = f", born {source_ctx['birth']}" if source_ctx.get('birth') else ""
            report += f"**{source}** is categorized as a *{source_ctx['type']}*{era_info}{birth_info}. "
            report += f"Within the knowledge graph, it has {source_ctx['outgoing_count']} outgoing morphisms "
            report += f"(concepts it influenced or created) and {source_ctx['incoming_count']} incoming morphisms "
            report += "(concepts that influenced it).\n\n"

        if target_ctx["exists"]:
            era_info = f" ({target_ctx['era']} era)" if target_ctx.get('era') and target_ctx['era'] != 'unknown' else ""
            birth_info = f", born {target_ctx['birth']}" if target_ctx.get('birth') else ""
            report += f"**{target}** is categorized as a *{target_ctx['type']}*{era_info}{birth_info}. "
            report += f"It has {target_ctx['outgoing_count']} outgoing and {target_ctx['incoming_count']} incoming morphisms.\n\n"

        # =====================================================================
        # Evolutionary Paths Detail
        # =====================================================================
        report += self._section("Evolutionary Pathways")

        if not paths:
            report += "No paths were found connecting these concepts. This represents a **categorical gap** "
            report += "in the knowledge graph. See the Oracle Predictions section for hypotheses about "
            report += "potential missing connections.\n\n"
        else:
            report += f"The system discovered {len(paths)} distinct evolutionary pathway(s):\n\n"

            for i, path in enumerate(paths[:max_paths], 1):
                report += f"### Pathway {i}: Length {path.length}\n\n"

                # Collect morphism details
                morphism_details = []
                for mor_id in path.morphism_ids:
                    mor = self.store.get_morphism(mor_id)
                    if mor:
                        morphism_details.append({
                            "from": mor.source_name,
                            "to": mor.target_name,
                            "type": mor.name,
                            "year": mor.metadata.get("year", ""),
                            "confidence": mor.confidence,
                            "notes": mor.metadata.get("notes", "")
                        })

                # Visual representation
                if morphism_details:
                    path_str = morphism_details[0]["from"]
                    for m in morphism_details:
                        year_str = f" ({m['year']})" if m['year'] else ""
                        path_str += f"\n    |--[{m['type']}{year_str}]-->\n{m['to']}"
                    report += self._code_block(path_str)

                # Narrative for this path
                if len(morphism_details) >= 2:
                    report += f"This pathway traces the evolution through {len(morphism_details)} stages. "
                    first_rel = morphism_details[0]
                    last_rel = morphism_details[-1]
                    report += f"Beginning with {first_rel['from']}'s {first_rel['type']} relationship to {first_rel['to']}, "
                    report += f"the chain of influence ultimately leads to {last_rel['to']}.\n\n"

                # Detailed table
                report += self._table(
                    ["Step", "From", "Relationship", "To", "Year", "Confidence"],
                    [[j+1, m["from"], m["type"], m["to"], m["year"] or "-", f"{m['confidence']:.2f}"]
                     for j, m in enumerate(morphism_details)]
                )
                report += "\n"

        # =====================================================================
        # Equivalence Analysis
        # =====================================================================
        report += self._section("Equivalence Analysis (HoTT)")

        report += "In Homotopy Type Theory, equivalences represent 'sameness' at a deep structural level. "
        report += "Two concepts that are equivalent can be freely substituted in any context—"
        report += "they are, in a precise mathematical sense, 'the same thing.'\n\n"

        # Check for relevant equivalences
        equivalences = self.store.list_equivalences()
        relevant_equivs = []

        for equiv in equivalences:
            if source in equiv.member_names or target in equiv.member_names:
                relevant_equivs.append(equiv)
            # Also check if any intermediaries are in equivalence classes
            if paths:
                for path in paths:
                    for mor_id in path.morphism_ids:
                        mor = self.store.get_morphism(mor_id)
                        if mor and (mor.source_name in equiv.member_names or mor.target_name in equiv.member_names):
                            if equiv not in relevant_equivs:
                                relevant_equivs.append(equiv)

        if relevant_equivs:
            report += f"Found **{len(relevant_equivs)} relevant equivalence class(es)**:\n\n"
            for equiv in relevant_equivs:
                report += f"### {equiv.name}\n\n"
                report += f"**Members:** {', '.join(equiv.member_names)}\n\n"
                report += f"**Type:** {equiv.equivalence_type}\n\n"
                report += f"**Witness:** {equiv.witness}\n\n"
                report += f"**Confidence:** {equiv.confidence:.2f}\n\n"

                # Explain significance
                if source in equiv.member_names and target in equiv.member_names:
                    report += f"> **Significant Finding:** Both {source} and {target} belong to this equivalence class! "
                    report += "This suggests they are, at some level of abstraction, 'the same concept.'\n\n"
                elif source in equiv.member_names:
                    report += f"> {source} is equivalent to {[m for m in equiv.member_names if m != source]}. "
                    report += "Consider whether evolution could proceed through equivalent formulations.\n\n"
        else:
            report += "No equivalence classes directly involve the source or target concepts.\n\n"

        # =====================================================================
        # Oracle Module: Predictive Intelligence
        # =====================================================================
        report += self._section("Oracle Module: Predictive Intelligence")

        report += "The Categorical Oracle uses discovered structures to **predict** what should exist, "
        report += "enabling hypothesis-driven research rather than just organizing discovered knowledge.\n\n"

        report += "### Oracle Architecture\n\n"
        report += "The Oracle employs **8 rigorous inference strategies** backed by categorical mathematics:\n\n"
        report += "1. **Kan Extension Strategy** - Computes colimits to infer missing morphisms from surrounding structure\n"
        report += "2. **Semantic Similarity Strategy** - Uses embedding space to find semantically related concepts\n"
        report += "3. **Temporal Reasoning Strategy** - Analyzes birth/death dates for causal plausibility\n"
        report += "4. **Type Heuristic Strategy** - Applies domain-specific relationship constraints\n"
        report += "5. **Yoneda Pattern Strategy** - Identifies objects with isomorphic morphism patterns\n"
        report += "6. **Composition Strategy** - Predicts transitive closure (A→B→C implies A→C)\n"
        report += "7. **Fibration Lift Strategy** - Uses fibered category structure for Cartesian lifts\n"
        report += "8. **Structural Hole Strategy** - Finds triangles that should close\n\n"

        predictions = self._generate_oracle_predictions(source, target, paths)
        oracle_metadata = predictions[0].get("_oracle_metadata", {}) if predictions else {}

        if oracle_metadata:
            report += "### Oracle Execution Summary\n\n"
            report += f"- **Total candidates generated:** {oracle_metadata.get('total_candidates', 'N/A')}\n"
            report += f"- **Coherence score:** {oracle_metadata.get('coherence_score', 0):.2%}\n"
            report += f"- **Computation time:** {oracle_metadata.get('computation_time_ms', 0):.1f}ms\n\n"

            if oracle_metadata.get('strategy_contributions'):
                report += "**Strategy Contributions:**\n\n"
                for strategy, count in oracle_metadata['strategy_contributions'].items():
                    report += f"- {strategy}: {count} predictions\n"
                report += "\n"

        if predictions:
            report += "### Generated Hypotheses\n\n"
            report += self._table(
                ["#", "Type", "Prediction", "Confidence", "Reasoning"],
                [[i+1, p["type"][:30], p["prediction"][:50], f"{p['confidence']:.0%}", p["reason"][:40] + "..."]
                 for i, p in enumerate(predictions)]
            )
            report += "\n"

            # Detailed predictions with full interpretation
            report += "### Detailed Prediction Analysis\n\n"
            for i, pred in enumerate(predictions[:10], 1):
                report += f"#### Prediction {i}: {pred['prediction']}\n\n"
                report += f"**Strategies:** {pred['type']}\n\n"
                report += f"**Confidence:** {pred['confidence']:.0%}\n\n"
                report += f"**Full Reasoning:**\n\n"
                report += f"> {pred['reason']}\n\n"

                # Interpret what this means
                conf = pred['confidence']
                if conf >= 0.8:
                    report += f"*Interpretation:* This prediction has **high confidence** ({conf:.0%}). "
                    report += "Multiple independent strategies converged on this hypothesis, suggesting strong structural support. "
                    report += "This relationship likely exists and should be verifiable through historical research.\n\n"
                elif conf >= 0.6:
                    report += f"*Interpretation:* This prediction has **moderate confidence** ({conf:.0%}). "
                    report += "The categorical structure provides reasonable support for this hypothesis. "
                    report += "Further investigation is warranted to confirm or refute.\n\n"
                else:
                    report += f"*Interpretation:* This prediction has **lower confidence** ({conf:.0%}). "
                    report += "While structurally plausible, more evidence would strengthen this hypothesis.\n\n"
        else:
            report += "### No Predictions Generated\n\n"
            report += "The Oracle could not generate predictions for this query. This may indicate:\n\n"
            report += "- Insufficient surrounding structure to infer relationships\n"
            report += "- Objects are too far apart in the categorical structure\n"
            report += "- Missing embeddings (run `python cli.py embed` first)\n\n"

        report += "### Suggested Verification Searches\n\n"
        report += "To verify these predictions, investigate:\n\n"
        report += f"- `{source} {target} historical connection`\n"
        report += f"- `{source} influence on {target}`\n"
        report += f"- `{source} AND {target} relationship`\n"
        if path_analysis.get("intermediaries"):
            for intermediary in list(path_analysis["intermediaries"])[:3]:
                report += f"- `{intermediary} relationship to {source}`\n"
                report += f"- `{intermediary} contribution to {target}`\n"
        report += "\n"

        # =====================================================================
        # Sheaf Coherence Analysis (NEW - like KOMPOSOS-jf)
        # =====================================================================
        report += self._section("Sheaf Coherence Analysis")

        report += "In sheaf theory, data is **coherent** if local pieces 'glue together' consistently. "
        report += "The sheaf condition requires that information from different sources agrees on overlaps. "
        report += "We apply this principle to validate our categorical findings.\n\n"

        # Get coherence from Oracle if available
        if ORACLE_AVAILABLE and self.embeddings and self.embeddings.is_available:
            try:
                from oracle import CategoricalOracle
                oracle = CategoricalOracle(self.store, self.embeddings)
                result = oracle.predict(source, target)

                report += f"**Coherence Score:** {result.coherence_result.coherence_score:.2%}\n\n"
                report += f"**Minimum Similarity:** {result.coherence_result.min_similarity:.2%}\n\n"

                if result.coherence_result.is_coherent:
                    report += "✅ **Status: COHERENT** - Predictions from different strategies agree.\n\n"
                    report += "The categorical structure satisfies the sheaf condition: local data glues consistently "
                    report += "to form a coherent global picture. This increases confidence in our findings.\n\n"
                else:
                    report += "⚠️ **Status: INCONSISTENCIES DETECTED** - Some predictions conflict.\n\n"
                    report += "The sheaf condition is not fully satisfied. This may indicate:\n"
                    report += "- Genuine historical disagreements in how this evolution occurred\n"
                    report += "- Incomplete data requiring further investigation\n"
                    report += "- Alternative interpretations of the same conceptual development\n\n"

                if result.coherence_result.contradictions:
                    report += "### Detected Contradictions\n\n"
                    for pred1, pred2, reason in result.coherence_result.contradictions[:5]:
                        report += f"- **Conflict:** '{pred1.predicted_relation}' vs '{pred2.predicted_relation}'\n"
                        report += f"  - Reason: {reason}\n"
                        report += f"  - Resolution: Investigate which relationship is historically accurate\n\n"
            except Exception as e:
                report += f"*Coherence analysis unavailable: {e}*\n\n"
        else:
            report += "*Sheaf coherence analysis requires embeddings. Run `python cli.py embed` first.*\n\n"

        # =====================================================================
        # Yoneda-Based Structural Analysis
        # =====================================================================
        report += self._section("Structural Analysis (Yoneda Lemma)")
        report += self._generate_yoneda_analysis([source, target] + list(path_analysis.get("intermediaries", []))[:3])

        # =====================================================================
        # Categorical 2-Structure and Path Homotopy Analysis
        # =====================================================================
        report += self._section("Categorical 2-Structure")

        report += "Beyond morphisms (1-morphisms), KOMPOSOS-III can represent **2-morphisms**—"
        report += "transformations between transformations. These capture:\n\n"
        report += "- **Natural transformations** between different evolutionary pathways\n"
        report += "- **Homotopies** showing when two paths are 'essentially the same'\n"
        report += "- **Higher coherence** conditions for complex conceptual relationships\n\n"

        report += self._code_block(f"""{source}_to_{target} (2-Category Structure)
  Objects: {source}, {target}, intermediaries...
  1-Morphisms (Paths):
    - {'Multiple pathways' if path_analysis.get('convergent') else 'Single pathway'}
  2-Morphisms (Path equivalences):
    - structural_correspondence: Path1 => Path2 (if paths are equivalent)
""")

        if path_analysis.get("convergent"):
            report += f"\nThe existence of {path_analysis['count']} paths suggests a rich 2-categorical structure. "
            report += "A natural question is: are these paths **homotopic** (equivalent as paths)? "
            report += "If so, they represent the same 'proof' that " + source + " evolved into " + target + ".\n\n"

            # Perform actual homotopy analysis
            if HOMOTOPY_AVAILABLE and paths:
                report += "### Path Homotopy Analysis\n\n"

                # Extract node sequences from paths
                path_sequences = []
                for path in paths[:6]:  # Limit to first 6 paths for performance
                    sequence = [source]
                    for mor_id in path.morphism_ids:
                        mor = self.store.get_morphism(mor_id)
                        if mor and mor.target_name not in sequence:
                            sequence.append(mor.target_name)
                    path_sequences.append(sequence)

                # Run homotopy checker
                homotopy_result = check_path_homotopy(path_sequences, self.store)

                # Report shared spine
                if homotopy_result.shared_spine:
                    report += f"**Shared Spine** (nodes in ALL paths):\n\n"
                    report += f"```\n{' → '.join(homotopy_result.shared_spine)}\n```\n\n"
                    report += f"The spine represents the **essential intermediaries** through which every "
                    report += f"evolutionary pathway must pass. It is the 'irreducible core' of the relationship.\n\n"

                # Report homotopy classes
                report += f"**Homotopy Classes**: {homotopy_result.num_classes}\n\n"

                if homotopy_result.all_homotopic:
                    report += "**Result**: All paths are **HOMOTOPIC** (equivalent as proofs)\n\n"
                    report += "> In HoTT terms, these paths represent the 'same proof' that "
                    report += f"{source} evolved into {target}. The variations (different intermediaries) "
                    report += "are 'contractible detours' that don't change the essential structure.\n\n"
                    report += "> Categorically: There exists a 2-morphism (natural transformation) "
                    report += "between any pair of paths, making them 'essentially the same'.\n\n"
                else:
                    report += f"**Result**: Paths fall into **{homotopy_result.num_classes} distinct** homotopy classes\n\n"

                    for i, cls in enumerate(homotopy_result.homotopy_classes, 1):
                        path_nums = sorted([n+1 for n in cls])
                        report += f"- **Class {i}**: Paths {path_nums}\n"
                    report += "\n"

                    report += "> In HoTT terms, these paths represent **genuinely different 'proofs'** "
                    report += f"of the evolutionary relationship from {source} to {target}. "
                    report += "The differences are NOT mere 'detours' but represent distinct intellectual "
                    report += "lineages that cannot be continuously deformed into each other.\n\n"
                    report += "> Categorically: There is NO 2-morphism between paths in different "
                    report += "classes. They represent independent pieces of evidence.\n\n"

        # =====================================================================
        # Cross-Domain Analysis (NEW - like KOMPOSOS-jf)
        # =====================================================================
        report += self._section("Cross-Domain Analysis")

        # Analyze type transitions in the paths
        type_transitions = []
        if paths:
            for path in paths[:3]:
                for mor_id in path.morphism_ids:
                    mor = self.store.get_morphism(mor_id)
                    if mor:
                        src_obj = self.store.get_object(mor.source_name)
                        tgt_obj = self.store.get_object(mor.target_name)
                        if src_obj and tgt_obj and src_obj.type_name != tgt_obj.type_name:
                            type_transitions.append({
                                "from_type": src_obj.type_name,
                                "to_type": tgt_obj.type_name,
                                "from_obj": mor.source_name,
                                "to_obj": mor.target_name,
                                "relation": mor.name
                            })

        if type_transitions:
            report += "The evolutionary pathway traverses **multiple conceptual domains**, revealing cross-domain bridges:\n\n"
            report += "### Discovered Cross-Domain Bridges\n\n"

            for i, bridge in enumerate(type_transitions[:5], 1):
                report += f"**Bridge {i}:** {bridge['from_obj']} ({bridge['from_type']}) → {bridge['to_obj']} ({bridge['to_type']})\n\n"
                report += f"- **Relationship:** {bridge['relation']}\n"
                report += f"- **Domain transition:** {bridge['from_type']} → {bridge['to_type']}\n"
                report += f"- **Interpretation:** This morphism represents a **categorical functor** between the category of "
                report += f"{bridge['from_type']}s and the category of {bridge['to_type']}s. "
                report += "Such functors preserve structure while translating between domains.\n\n"

            report += "### Cross-Domain Significance\n\n"
            report += "Cross-domain bridges are categorically significant because they represent **functorial mappings** "
            report += "between different conceptual categories. In the language of category theory, these bridges show how "
            report += "structure in one domain (e.g., physics) maps to structure in another (e.g., mathematics). "
            report += "The existence of such bridges suggests deep structural connections that transcend disciplinary boundaries.\n\n"
        else:
            report += "The evolutionary pathway remains within a single conceptual domain. "
            report += "This suggests a more traditional mode of intellectual inheritance within a field.\n\n"

        # =====================================================================
        # Plain English Summary (Layman's Explanation)
        # =====================================================================
        report += self._section("Plain English Summary")
        report += "*This section explains the findings in everyday language, without technical jargon.*\n\n"
        report += self._generate_layman_summary(source, target, path_analysis, paths, predictions)

        # =====================================================================
        # Conclusions and Future Directions
        # =====================================================================
        report += self._section("Conclusions")

        # Calculate confidence score with more factors
        confidence = 30
        if path_analysis["found"]:
            confidence = 60
            confidence += min(20, len(paths) * 5)  # Bonus for multiple paths
            confidence += 10 if path_analysis.get('convergent') else 0  # Bonus for convergence
            confidence += min(10, len(predictions) * 2)  # Bonus for predictions
            confidence = min(95, confidence)

        if path_analysis["found"]:
            report += f"**Hypothesis Status:** SUPPORTED\n\n"
        else:
            report += f"**Hypothesis Status:** INCONCLUSIVE (no direct pathway found)\n\n"

        report += f"**Overall Confidence Score:** {confidence}%\n\n"

        # Detailed interpretation
        report += "### Interpretation of Findings\n\n"
        if path_analysis["found"]:
            if path_analysis.get("convergent"):
                report += f"The discovery of {path_analysis['count']} independent evolutionary pathways from **{source}** "
                report += f"to **{target}** provides **strong categorical evidence** for the claimed evolutionary relationship. "
                report += "In category theory, the existence of multiple paths that compose to the same endpoint suggests "
                report += "a robust structural connection—one that does not depend on any single historical narrative. "
                report += "This is analogous to having multiple independent proofs of the same theorem.\n\n"
            else:
                report += f"A single canonical pathway connects **{source}** to **{target}**, suggesting a well-defined "
                report += "historical lineage. While unique, this path provides clear evidence of conceptual evolution. "
                report += "The categorical framework guarantees that this path represents a valid composition of morphisms.\n\n"

            if predictions:
                report += f"The Oracle generated **{len(predictions)} hypothesis(es)**, indicating that the categorical "
                report += "structure contains sufficient information to make meaningful predictions. "
                avg_conf = sum(p['confidence'] for p in predictions) / len(predictions)
                report += f"The average prediction confidence of **{avg_conf:.0%}** suggests "
                report += f"{'strong' if avg_conf > 0.7 else 'moderate' if avg_conf > 0.5 else 'tentative'} structural support.\n\n"
        else:
            report += f"No direct pathway was found from **{source}** to **{target}** in the current knowledge graph. "
            report += "This does not necessarily mean no relationship exists—it may indicate:\n\n"
            report += "1. **Missing data:** Intermediate concepts not yet captured in the graph\n"
            report += "2. **Indirect relationship:** The connection may require higher-dimensional structure\n"
            report += "3. **Novel discovery opportunity:** A genuine gap in documented intellectual history\n\n"

        report += "### Evidence Summary\n\n"
        report += self._table(
            ["Evidence Type", "Finding", "Contribution to Confidence"],
            [
                ["Path Analysis", f"{path_analysis['count'] if path_analysis['found'] else 0} pathways", f"+{min(20, (path_analysis['count'] if path_analysis['found'] else 0) * 5)}%"],
                ["Convergent Evolution", "Yes" if path_analysis.get('convergent') else "No", f"+{10 if path_analysis.get('convergent') else 0}%"],
                ["Equivalences", f"{len(relevant_equivs)} classes", f"+{min(10, len(relevant_equivs) * 5)}%"],
                ["Oracle Predictions", f"{len(predictions)} hypotheses", f"+{min(10, len(predictions) * 2)}%"],
            ]
        )
        report += "\n"

        report += "### Future Research Directions\n\n"
        report += "Based on this analysis, the following research directions are recommended:\n\n"
        report += "1. **Verify Oracle Predictions:** The Oracle has generated testable hypotheses. "
        report += "Use the suggested search queries to find evidence for or against each prediction.\n\n"
        report += "2. **Investigate Equivalences:** The HoTT framework suggests searching for undiscovered equivalence classes. "
        report += "Are there concepts in this evolutionary chain that are secretly 'the same'?\n\n"
        report += "3. **Fill Gaps:** Apply Kan extensions to infer missing morphisms. The categorical structure may "
        report += "contain enough information to compute what relationships *should* exist.\n\n"
        report += "4. **Higher Structure:** If multiple paths exist, investigate whether they are homotopic (equivalent as paths). "
        report += "This would reveal deeper structural insights about the nature of conceptual evolution.\n\n"
        report += "5. **Cross-Domain Exploration:** If cross-domain bridges were found, explore whether similar "
        report += "functorial mappings exist for related concepts.\n\n"

        report += "---\n\n"
        report += "*Report generated by KOMPOSOS-III Categorical AI System*\n"
        report += "*'Phylogenetics of concepts'—tracing how ideas evolve*\n"
        report += f"*Full workflow: Path Analysis → Equivalence Detection → Oracle Predictions → Yoneda Analysis → Conclusions*\n"

        return report

    def gap_report(self, threshold: float = 0.3) -> str:
        """
        Generate a rich report on semantic gaps in the knowledge graph.

        Gaps are pairs of objects that should plausibly be connected but aren't.
        This is where Kan extensions and cubical type theory become essential.
        """
        report = self._header(
            "KOMPOSOS-III Gap Analysis Report",
            f"Identifying missing connections (threshold: {threshold})"
        )

        # =====================================================================
        # Executive Summary
        # =====================================================================
        report += self._section("Executive Summary")

        stats = self.store.get_statistics()
        objects = self.store.list_objects(limit=1000)
        morphisms = self.store.list_morphisms(limit=10000)

        # Calculate graph density
        max_edges = len(objects) * (len(objects) - 1)
        density = len(morphisms) / max_edges if max_edges > 0 else 0

        report += f"This gap analysis examined **{stats['objects']} objects** connected by **{stats['morphisms']} morphisms** "
        report += f"to identify **structural holes** in the knowledge graph.\n\n"

        report += f"**Graph Statistics:**\n"
        report += f"- **Density:** {density:.4f} ({density*100:.2f}% of possible connections exist)\n"
        report += f"- **Average degree:** {2*len(morphisms)/len(objects):.2f} morphisms per object\n"
        report += f"- **Sparsity:** {'High' if density < 0.1 else 'Medium' if density < 0.3 else 'Low'}\n\n"

        # =====================================================================
        # Scientific Analysis
        # =====================================================================
        report += self._section("Scientific Analysis")

        report += "From a categorical perspective, a 'gap' represents a **missing morphism**—a relationship "
        report += "that structural or semantic considerations suggest should exist, but which is not yet documented. "
        report += "These gaps are candidates for:\n\n"

        report += "1. **Kan Extension:** Using the existing categorical structure to *compute* what the missing "
        report += "morphism should be, based on universal properties.\n\n"

        report += "2. **Cubical Filling (hcomp/hfill):** In cubical type theory, open boxes can be 'filled' "
        report += "with appropriate terms. Similarly, gaps in our knowledge graph can be filled with "
        report += "inferred relationships.\n\n"

        report += "3. **Research Opportunities:** Each gap represents a potential discovery—a relationship "
        report += "that may exist but hasn't been explicitly documented.\n\n"

        # =====================================================================
        # Semantic Gap Analysis
        # =====================================================================
        report += self._section("Semantic Gap Analysis")

        if not self.embeddings or not self.embeddings.is_available:
            report += "**Note:** Embeddings engine not available. Proceeding with structural analysis only.\n\n"
            report += "For full semantic gap analysis, run with embeddings enabled:\n"
            report += self._code_block("python cli.py embed  # First compute embeddings\npython cli.py report gaps --threshold 0.3", "bash")

            # Fall back to structural analysis
            gaps = []
        else:
            # Get existing connections
            connected_pairs = set()
            for mor in morphisms:
                connected_pairs.add((mor.source_name, mor.target_name))
                connected_pairs.add((mor.target_name, mor.source_name))

            # Find semantic gaps
            gaps = []
            report += f"Computing semantic similarities for {len(objects)} objects...\n\n"

            for i, obj1 in enumerate(objects):
                for obj2 in objects[i+1:]:
                    if (obj1.name, obj2.name) not in connected_pairs:
                        sim = self.embeddings.similarity(obj1.name, obj2.name)
                        if sim > threshold:  # High similarity but no connection = gap
                            gaps.append({
                                "obj1": obj1.name,
                                "obj2": obj2.name,
                                "type1": obj1.type_name,
                                "type2": obj2.type_name,
                                "similarity": sim,
                                "gap_significance": sim  # Higher similarity = more significant gap
                            })

            gaps.sort(key=lambda x: -x["gap_significance"])

            report += f"Found **{len(gaps)} semantic gaps** (similarity > {threshold} but no morphism):\n\n"

        # =====================================================================
        # Gap Details
        # =====================================================================
        if gaps:
            report += self._section("Significant Gaps")

            report += self._table(
                ["Rank", "Object 1", "Type", "Object 2", "Type", "Similarity", "Significance"],
                [[i+1, g["obj1"], g["type1"], g["obj2"], g["type2"],
                  f"{g['similarity']:.3f}", "HIGH" if g["similarity"] > 0.7 else "MEDIUM"]
                 for i, g in enumerate(gaps[:30])]
            )

            # Detailed analysis of top gaps
            report += "\n### Analysis of Top Gaps\n\n"

            for i, gap in enumerate(gaps[:5], 1):
                report += f"**Gap {i}: {gap['obj1']} ↔ {gap['obj2']}**\n\n"
                report += f"- **Semantic similarity:** {gap['similarity']:.4f}\n"
                report += f"- **Types:** {gap['type1']} and {gap['type2']}\n"
                report += f"- **Significance:** {'High' if gap['similarity'] > 0.7 else 'Medium'}\n\n"

                # Generate hypothesis about the gap
                if gap['type1'] == gap['type2']:
                    report += f"> **Hypothesis:** As both are {gap['type1']}s with high semantic similarity, "
                    report += f"there may be a direct influence or collaboration relationship.\n\n"
                else:
                    report += f"> **Hypothesis:** The {gap['type1']}-{gap['type2']} relationship suggests "
                    report += f"a cross-type connection, possibly 'created', 'formalized', or 'inspired'.\n\n"
        else:
            report += "No significant semantic gaps detected at the current threshold.\n\n"

        # =====================================================================
        # Structural Gap Analysis
        # =====================================================================
        report += self._section("Structural Gap Analysis")

        report += "Beyond semantic similarity, we can identify structural gaps using categorical principles:\n\n"

        # Find objects with low connectivity
        degree_count = {}
        for mor in morphisms:
            degree_count[mor.source_name] = degree_count.get(mor.source_name, 0) + 1
            degree_count[mor.target_name] = degree_count.get(mor.target_name, 0) + 1

        isolated = [obj for obj in objects if obj.name not in degree_count]
        low_degree = [obj for obj in objects if degree_count.get(obj.name, 0) <= 1]

        if isolated:
            report += f"### Isolated Objects ({len(isolated)})\n\n"
            report += "These objects have **no morphisms** connecting them to the graph:\n\n"
            report += self._table(
                ["Name", "Type", "Status"],
                [[obj.name, obj.type_name, "ISOLATED"] for obj in isolated[:20]]
            )
            report += "\nIsolated objects represent **categorical vacuums**—they exist but have no structural role.\n\n"

        if low_degree:
            report += f"### Under-Connected Objects ({len(low_degree)})\n\n"
            report += "These objects have **minimal connections** (degree ≤ 1):\n\n"
            report += self._table(
                ["Name", "Type", "Degree"],
                [[obj.name, obj.type_name, degree_count.get(obj.name, 0)] for obj in low_degree[:20]]
            )

        # =====================================================================
        # Coherence Analysis
        # =====================================================================
        report += self._section("Coherence Analysis (Sheaf Condition)")

        coherence = self._compute_coherence_score(objects)

        report += "In sheaf theory, data is **coherent** if local pieces glue together consistently. "
        report += "We apply this principle to check if our knowledge graph forms a coherent whole:\n\n"

        report += f"**Coherence Score:** {coherence['score']:.2%}\n\n"
        report += f"- **Connected objects:** {coherence['connected']}\n"
        report += f"- **Isolated objects:** {coherence['isolated']}\n\n"

        if coherence['inconsistencies']:
            report += "### Detected Inconsistencies\n\n"
            for inc in coherence['inconsistencies'][:10]:
                report += f"- {inc}\n"
            report += "\n"

        # =====================================================================
        # Kan Extension Candidates
        # =====================================================================
        report += self._section("Kan Extension Candidates")

        report += "**Kan extensions** are the 'best approximation' of extending a functor along another functor. "
        report += "In our context, they identify morphisms that *should* exist based on the surrounding structure.\n\n"

        # Find triangles that should close
        kan_candidates = []
        obj_names = [obj.name for obj in objects]

        mor_dict = {}  # (source, target) -> morphism
        for mor in morphisms:
            mor_dict[(mor.source_name, mor.target_name)] = mor

        for obj in objects[:50]:  # Limit for performance
            # Find A -> B -> C where A -> C is missing
            outgoing = [m for m in morphisms if m.source_name == obj.name]
            for mor1 in outgoing:
                second_hop = [m for m in morphisms if m.source_name == mor1.target_name]
                for mor2 in second_hop:
                    if (obj.name, mor2.target_name) not in mor_dict and obj.name != mor2.target_name:
                        kan_candidates.append({
                            "from": obj.name,
                            "to": mor2.target_name,
                            "via": mor1.target_name,
                            "rel1": mor1.name,
                            "rel2": mor2.name,
                            "suggested_rel": f"composed_{mor1.name}_{mor2.name}"
                        })

        if kan_candidates:
            report += f"Found **{len(kan_candidates)} Kan extension candidates**:\n\n"
            report += self._table(
                ["From", "Via", "To", "Path", "Suggested Relation"],
                [[k["from"], k["via"], k["to"], f"{k['rel1']} -> {k['rel2']}", k["suggested_rel"][:30]]
                 for k in kan_candidates[:20]]
            )
            report += "\nThese represent morphisms that categorical structure suggests should exist.\n\n"
        else:
            report += "No obvious Kan extension candidates found. The graph may be categorically complete.\n\n"

        # =====================================================================
        # Conclusions
        # =====================================================================
        report += self._section("Conclusions")

        total_gaps = len(gaps) + len(isolated) + len(kan_candidates)

        if total_gaps > 20:
            status = "SIGNIFICANT GAPS DETECTED"
            confidence = 85
        elif total_gaps > 5:
            status = "MODERATE GAPS DETECTED"
            confidence = 70
        else:
            status = "WELL-CONNECTED"
            confidence = 90

        report += f"**Graph Status:** {status}\n\n"
        report += f"**Completeness Confidence:** {confidence}%\n\n"

        report += "### Summary Statistics\n\n"
        report += f"- **Semantic gaps:** {len(gaps)}\n"
        report += f"- **Isolated objects:** {len(isolated)}\n"
        report += f"- **Kan extension candidates:** {len(kan_candidates)}\n"
        report += f"- **Coherence score:** {coherence['score']:.2%}\n\n"

        report += "### Recommended Actions\n\n"
        report += "1. **Priority 1:** Connect isolated objects to the main graph\n"
        report += "2. **Priority 2:** Investigate high-similarity gaps for missing relationships\n"
        report += "3. **Priority 3:** Implement Kan extensions to infer composed morphisms\n"
        report += "4. **Priority 4:** Apply cubical hcomp/hfill to close structural holes\n\n"

        report += "---\n\n"
        report += "*Report generated by KOMPOSOS-III Categorical AI System*\n"
        report += "*Gap analysis enables discovery of hidden connections*\n"

        return report

    def equivalence_report(self) -> str:
        """
        Generate a rich report on equivalences in the knowledge graph.

        Equivalences implement HoTT's univalence axiom: equivalent things ARE equal.
        This is where the type-theoretic foundation becomes essential.
        """
        report = self._header(
            "KOMPOSOS-III Equivalence Analysis Report",
            "Implementing the Univalence Axiom"
        )

        # =====================================================================
        # Executive Summary
        # =====================================================================
        report += self._section("Executive Summary")

        equivalences = self.store.list_equivalences()
        stats = self.store.get_statistics()

        report += f"This report analyzes **{len(equivalences)} equivalence classes** across "
        report += f"**{stats['objects']} objects** in the KOMPOSOS-III knowledge graph.\n\n"

        report += "**Key Metrics:**\n"
        report += f"- **Equivalence classes:** {len(equivalences)}\n"
        report += f"- **Total equivalent pairs:** {sum(len(e.member_names)*(len(e.member_names)-1)//2 for e in equivalences)}\n"
        report += f"- **Coverage:** {sum(len(e.member_names) for e in equivalences)} objects in equivalence relations\n\n"

        # =====================================================================
        # Theoretical Foundation
        # =====================================================================
        report += self._section("Theoretical Foundation")

        report += "### The Univalence Axiom\n\n"
        report += "In Homotopy Type Theory (HoTT), the **univalence axiom** states:\n\n"
        report += "> **(A ≃ B) ≃ (A = B)**\n\n"
        report += "This revolutionary principle asserts that **equivalent types are equal**. "
        report += "Two mathematical structures that are isomorphic are, for all practical purposes, "
        report += "*the same structure*.\n\n"

        report += "KOMPOSOS-III implements this principle for conceptual evolution:\n\n"
        report += "- **Wave mechanics ≃ Matrix mechanics:** Schrödinger's and Heisenberg's formulations "
        report += "are equivalent descriptions of quantum mechanics (proven by von Neumann, 1932)\n"
        report += "- **Lagrangian ≃ Hamiltonian:** Different formulations of classical mechanics "
        report += "that describe the same physical systems\n"
        report += "- **Set-theoretic ≃ Category-theoretic:** Different foundations that support equivalent mathematics\n\n"

        report += "### Implications for Evolutionary Analysis\n\n"
        report += "When tracing how concept A evolved into concept B, we can freely substitute "
        report += "equivalent concepts along the path. If A evolved into B, and B ≃ B', "
        report += "then A also evolved into B' in a categorical sense.\n\n"

        # =====================================================================
        # Equivalence Classes Detail
        # =====================================================================
        report += self._section("Equivalence Classes")

        if not equivalences:
            report += "No equivalence classes have been defined in the current knowledge graph.\n\n"
            report += "### How to Add Equivalences\n\n"
            report += "Equivalences can be added programmatically:\n\n"
            report += self._code_block("""from data import EquivalenceClass

store.add_equivalence(EquivalenceClass(
    name="QM_Formulations",
    member_names=["WaveMechanics", "MatrixMechanics"],
    equivalence_type="mathematical",
    witness="vonNeumann_1932",
    confidence=1.0,
    metadata={
        "year": 1932,
        "significance": "Unified quantum mechanics",
        "proof": "Hilbert space isomorphism"
    }
))""", "python")
            report += "\n"
        else:
            for equiv in equivalences:
                report += f"### {equiv.name}\n\n"

                # Summary table
                report += self._table(
                    ["Property", "Value"],
                    [
                        ["Type", equiv.equivalence_type],
                        ["Members", len(equiv.member_names)],
                        ["Witness", equiv.witness],
                        ["Confidence", f"{equiv.confidence:.2%}"]
                    ]
                )

                # Member details
                report += "\n**Equivalent Concepts:**\n\n"
                for member in equiv.member_names:
                    obj = self.store.get_object(member)
                    if obj:
                        report += f"- **{member}** ({obj.type_name}): "
                        if obj.metadata:
                            meta_str = ", ".join(f"{k}={v}" for k, v in list(obj.metadata.items())[:3])
                            report += meta_str
                        report += "\n"
                    else:
                        report += f"- **{member}**: (not in store)\n"

                # Explain the equivalence
                report += f"\n**Interpretation:**\n\n"
                if equiv.equivalence_type == "mathematical":
                    report += f"This is a **mathematical equivalence**—the members are provably isomorphic "
                    report += f"mathematical structures. The witness '{equiv.witness}' provides the proof.\n\n"
                elif equiv.equivalence_type == "conceptual":
                    report += f"This is a **conceptual equivalence**—the members represent the same "
                    report += f"underlying idea expressed in different forms or traditions.\n\n"
                elif equiv.equivalence_type == "historical":
                    report += f"This is a **historical equivalence**—the members were recognized "
                    report += f"as equivalent at some point in intellectual history.\n\n"
                else:
                    report += f"Equivalence type: {equiv.equivalence_type}\n\n"

                # Metadata
                if equiv.metadata:
                    report += "**Additional Context:**\n"
                    for k, v in equiv.metadata.items():
                        report += f"- {k}: {v}\n"
                    report += "\n"

        # =====================================================================
        # Equivalence Graph Structure
        # =====================================================================
        report += self._section("Equivalence Graph Structure")

        report += "Equivalences form their own categorical structure:\n\n"

        report += self._code_block(f"""Equivalence_Category
  Objects: {', '.join(e.name for e in equivalences) if equivalences else '(none)'}
  Morphisms:
    - reflexivity: A ≃ A (every object is equivalent to itself)
    - symmetry: A ≃ B implies B ≃ A
    - transitivity: A ≃ B and B ≃ C implies A ≃ C
  Properties:
    - Forms an equivalence relation (groupoid structure)
    - Respects morphisms (functorial)
""")

        # Check for transitive closures
        if len(equivalences) > 1:
            report += "\n### Potential Transitive Equivalences\n\n"
            report += "If A ≃ B and B ≃ C, then A ≃ C. Checking for implicit equivalences...\n\n"

            # Build member-to-class mapping
            member_to_class = {}
            for equiv in equivalences:
                for member in equiv.member_names:
                    if member not in member_to_class:
                        member_to_class[member] = []
                    member_to_class[member].append(equiv.name)

            # Find objects in multiple classes
            multi_class = {m: classes for m, classes in member_to_class.items() if len(classes) > 1}

            if multi_class:
                report += "**Objects appearing in multiple equivalence classes:**\n\n"
                for member, classes in multi_class.items():
                    report += f"- **{member}** appears in: {', '.join(classes)}\n"
                report += "\nThis suggests these equivalence classes may be transitively related.\n\n"
            else:
                report += "No objects appear in multiple equivalence classes—equivalences are independent.\n\n"

        # =====================================================================
        # HoTT Path Space Analysis
        # =====================================================================
        report += self._section("Path Space Analysis (HoTT)")

        report += "In HoTT, equivalences correspond to **paths** in the type universe. "
        report += "The path space between two types A and B is the type of all ways A can be "
        report += "continuously deformed into B.\n\n"

        report += "For our equivalence classes:\n\n"

        for equiv in equivalences:
            n = len(equiv.member_names)
            paths = n * (n - 1) // 2  # Number of pairwise equivalences
            report += f"- **{equiv.name}:** {n} members generate {paths} path(s) "
            report += f"in the type universe\n"

        report += "\nThe **fundamental groupoid** of our type universe has:\n"
        report += f"- **Objects:** All types (concepts) in KOMPOSOS-III\n"
        report += f"- **Morphisms:** Paths (equivalences) between types\n"
        report += f"- **Composition:** Transitive closure of equivalences\n"
        report += f"- **Identity:** Reflexivity (every type equivalent to itself)\n"
        report += f"- **Inverse:** Symmetry (equivalences are bidirectional)\n\n"

        # =====================================================================
        # Oracle: Predicted Equivalences
        # =====================================================================
        report += self._section("Oracle: Predicted Equivalences")

        report += "Based on the categorical structure, the Oracle predicts these potential equivalences:\n\n"

        # Use structural similarity to predict equivalences
        predictions = []
        morphisms = self.store.list_morphisms(limit=10000)
        objects = self.store.list_objects(limit=500)

        # Build signature for each object (set of relation types)
        signatures = {}
        for obj in objects:
            out_rels = set(m.name for m in morphisms if m.source_name == obj.name)
            in_rels = set(m.name for m in morphisms if m.target_name == obj.name)
            signatures[obj.name] = (frozenset(out_rels), frozenset(in_rels), obj.type_name)

        # Find objects with identical signatures
        sig_groups = {}
        for name, sig in signatures.items():
            if sig not in sig_groups:
                sig_groups[sig] = []
            sig_groups[sig].append(name)

        for sig, members in sig_groups.items():
            if len(members) > 1:
                # Check not already in an equivalence class together
                already_equiv = False
                for equiv in equivalences:
                    if all(m in equiv.member_names for m in members):
                        already_equiv = True
                        break

                if not already_equiv:
                    predictions.append({
                        "members": members,
                        "reason": f"Same structural signature (out: {len(sig[0])}, in: {len(sig[1])}, type: {sig[2]})",
                        "confidence": 0.6 + 0.1 * len(sig[0]) + 0.1 * len(sig[1])
                    })

        if predictions:
            report += self._table(
                ["Predicted Equivalence", "Confidence", "Reason"],
                [[" ≃ ".join(p["members"][:3]), f"{min(p['confidence'], 0.95):.0%}", p["reason"][:50]]
                 for p in sorted(predictions, key=lambda x: -x["confidence"])[:10]]
            )
            report += "\nThese predictions are based on **Yoneda-style structural analysis**—objects with "
            report += "identical relationship patterns may be categorically equivalent.\n\n"
        else:
            report += "No additional equivalences predicted based on current structural analysis.\n\n"

        # =====================================================================
        # Conclusions
        # =====================================================================
        report += self._section("Conclusions")

        if equivalences:
            report += f"**Equivalence Status:** {len(equivalences)} equivalence classes defined\n\n"
            report += f"**Univalence Implementation:** Active\n\n"
        else:
            report += "**Equivalence Status:** No equivalences defined\n\n"
            report += "**Recommendation:** Consider adding equivalences for mathematically equivalent formulations\n\n"

        report += "### The Power of Equivalences\n\n"
        report += "Equivalences enable:\n\n"
        report += "1. **Path flexibility:** Multiple routes through equivalent concepts\n"
        report += "2. **Conceptual unification:** Recognizing when different terms mean the same thing\n"
        report += "3. **Historical insight:** Understanding when discoveries were actually rediscoveries\n"
        report += "4. **Inference:** Transferring knowledge between equivalent domains\n\n"

        report += "---\n\n"
        report += "*Report generated by KOMPOSOS-III Categorical AI System*\n"
        report += "*Implementing the univalence axiom: equivalent concepts are equal*\n"

        return report

    def full_report(self) -> str:
        """
        Generate a comprehensive analysis report on the entire knowledge graph.

        This is the master report combining all analytical perspectives.
        """
        report = self._header(
            "KOMPOSOS-III Comprehensive Analysis Report",
            "Categorical Game-Theoretic Type-Theoretic AI"
        )

        stats = self.store.get_statistics()
        objects = self.store.list_objects(limit=1000)
        morphisms = self.store.list_morphisms(limit=10000)
        equivalences = self.store.list_equivalences()

        # =====================================================================
        # Executive Summary
        # =====================================================================
        report += self._section("Executive Summary")

        report += "This comprehensive analysis examines the KOMPOSOS-III knowledge graph through "
        report += "the lens of **Category Theory**, **Homotopy Type Theory**, **Cubical Type Theory**, "
        report += "and **Game Theory**—the four pillars of our categorical AI system.\n\n"

        report += "### Key Findings\n\n"
        report += self._table(
            ["Metric", "Value", "Interpretation"],
            [
                ["Objects (Concepts)", stats['objects'], "Nodes in the categorical structure"],
                ["Morphisms (Relations)", stats['morphisms'], "Arrows connecting concepts"],
                ["Equivalence Classes", stats['equivalences'], "HoTT univalence implementation"],
                ["Stored Paths", stats['paths'], "Computed evolutionary trajectories"],
                ["2-Morphisms", stats['higher_morphisms'], "Higher categorical structure"],
            ]
        )

        # Calculate key metrics
        max_edges = len(objects) * (len(objects) - 1)
        density = len(morphisms) / max_edges if max_edges > 0 else 0
        avg_degree = 2 * len(morphisms) / len(objects) if objects else 0

        report += f"\n**Graph Density:** {density:.4f} ({density*100:.2f}%)\n"
        report += f"**Average Connectivity:** {avg_degree:.2f} morphisms per object\n\n"

        # =====================================================================
        # Scientific Overview
        # =====================================================================
        report += self._section("Scientific Overview")

        report += "### What is KOMPOSOS-III?\n\n"
        report += "KOMPOSOS-III implements **'phylogenetics of concepts'**—tracing how ideas evolve, "
        report += "transform, and relate across time and domains. Unlike traditional knowledge graphs "
        report += "that merely store relationships, KOMPOSOS-III provides:\n\n"

        report += "1. **Categorical Semantics:** Objects and morphisms form a category with composition, "
        report += "identity, and associativity. This isn't just data—it's mathematical structure.\n\n"

        report += "2. **HoTT Equivalences:** The univalence axiom lets us identify equivalent concepts, "
        report += "enabling flexible path finding and conceptual unification.\n\n"

        report += "3. **Cubical Operations:** hcomp and hfill operations allow 'filling gaps'—"
        report += "inferring missing connections from surrounding structure.\n\n"

        report += "4. **Game-Theoretic Optimization:** Nash equilibria guide optimal path selection "
        report += "when multiple evolutionary routes exist.\n\n"

        # Domain analysis
        report += "### Knowledge Domain Analysis\n\n"

        if stats.get("object_types"):
            report += "The knowledge graph spans these conceptual domains:\n\n"
            report += self._table(
                ["Domain/Type", "Count", "Percentage", "Role"],
                [[t, c, f"{100*c/stats['objects']:.1f}%",
                  "Primary" if c > stats['objects']*0.2 else "Secondary" if c > stats['objects']*0.1 else "Supporting"]
                 for t, c in sorted(stats["object_types"].items(), key=lambda x: -x[1])]
            )

            # Narrative about the domains
            top_types = sorted(stats["object_types"].items(), key=lambda x: -x[1])[:3]
            report += f"\nThe knowledge graph is primarily composed of **{top_types[0][0]}** objects "
            report += f"({top_types[0][1]} instances, {100*top_types[0][1]/stats['objects']:.1f}%), "
            if len(top_types) > 1:
                report += f"followed by **{top_types[1][0]}** ({top_types[1][1]} instances). "
            report += "This composition reflects the domain focus of the corpus.\n\n"

        # =====================================================================
        # Categorical Structure Analysis
        # =====================================================================
        report += self._section("Categorical Structure Analysis")

        report += "### Morphism Distribution\n\n"

        if stats.get("morphism_types"):
            report += "The relationships in our category have these types:\n\n"
            report += self._table(
                ["Relation Type", "Count", "Percentage", "Categorical Role"],
                [[t, c, f"{100*c/stats['morphisms']:.1f}%",
                  "Primary structure" if c > stats['morphisms']*0.15 else "Secondary" if c > stats['morphisms']*0.05 else "Fine structure"]
                 for t, c in sorted(stats["morphism_types"].items(), key=lambda x: -x[1])]
            )

            # Analyze morphism patterns
            top_mor = sorted(stats["morphism_types"].items(), key=lambda x: -x[1])
            report += f"\nThe dominant relationship type is **'{top_mor[0][0]}'** with {top_mor[0][1]} instances. "
            report += "This morphism type defines the primary categorical structure—the main way "
            report += "concepts relate to each other in this knowledge domain.\n\n"

        report += "### Categorical Properties\n\n"
        report += self._code_block(f"""KOMPOSOS_Category
  Objects: {stats['objects']} concepts
  Morphisms: {stats['morphisms']} relationships

  Properties:
    - Identity: Every object has id_A : A -> A
    - Composition: f: A->B, g: B->C implies g∘f: A->C
    - Associativity: (h∘g)∘f = h∘(g∘f)

  Additional Structure:
    - Equivalences: {stats['equivalences']} (implementing univalence)
    - 2-Morphisms: {stats['higher_morphisms']} (higher structure)
    - Enrichment: Over Set (discrete) or Vect (embeddings)
""")

        # =====================================================================
        # Graph Connectivity Analysis
        # =====================================================================
        report += self._section("Graph Connectivity Analysis")

        try:
            import networkx as nx
            G = self.store.export_to_networkx()

            report += f"**Network Metrics:**\n"
            report += f"- Nodes: {G.number_of_nodes()}\n"
            report += f"- Edges: {G.number_of_edges()}\n"
            report += f"- Density: {nx.density(G):.4f}\n\n"

            if G.number_of_nodes() > 0:
                # Connected components
                wccs = list(nx.weakly_connected_components(G))
                report += f"**Connected Components:** {len(wccs)}\n\n"

                if len(wccs) == 1:
                    report += "The graph is **fully connected**—every concept can reach every other concept "
                    report += "through some chain of morphisms. This is ideal for evolutionary analysis.\n\n"
                else:
                    report += f"The graph has **{len(wccs)} disconnected components**. This may indicate:\n"
                    report += "- Separate knowledge domains not yet linked\n"
                    report += "- Missing bridging concepts\n"
                    report += "- Opportunities for cross-domain discovery\n\n"

                    report += "**Component Sizes:**\n"
                    for i, wcc in enumerate(sorted(wccs, key=len, reverse=True)[:5], 1):
                        sample = list(wcc)[:3]
                        report += f"- Component {i}: {len(wcc)} nodes (e.g., {', '.join(sample)}...)\n"
                    report += "\n"

                # Hub analysis
                report += "### Hub Analysis (Most Connected Concepts)\n\n"
                degrees = sorted(G.degree(), key=lambda x: -x[1])[:15]
                report += self._table(
                    ["Concept", "Connections", "Role"],
                    [[n, d, "Major Hub" if d > avg_degree * 2 else "Hub" if d > avg_degree else "Standard"]
                     for n, d in degrees]
                )

                report += f"\nThese highly-connected nodes serve as **categorical limits**—universal objects "
                report += "through which many morphisms factor. They are essential for path finding.\n\n"

        except ImportError:
            report += "NetworkX not available for detailed graph analysis.\n\n"

        # =====================================================================
        # Equivalence Summary
        # =====================================================================
        report += self._section("Equivalence Summary (HoTT)")

        if equivalences:
            report += f"**{len(equivalences)} equivalence classes** implement the univalence axiom:\n\n"

            for equiv in equivalences:
                report += f"### {equiv.name}\n\n"
                report += f"- **Members:** {', '.join(equiv.member_names)}\n"
                report += f"- **Type:** {equiv.equivalence_type}\n"
                report += f"- **Witness:** {equiv.witness}\n"
                report += f"- **Confidence:** {equiv.confidence:.2%}\n\n"

                # Interpret the equivalence
                if len(equiv.member_names) == 2:
                    report += f"> This equivalence asserts that **{equiv.member_names[0]}** and "
                    report += f"**{equiv.member_names[1]}** are, at the appropriate level of abstraction, "
                    report += f"*the same thing*. Any path through one can be transformed into a path through the other.\n\n"
                else:
                    report += f"> This equivalence class groups {len(equiv.member_names)} concepts that are "
                    report += f"mutually equivalent—forming a single point in the quotient category.\n\n"
        else:
            report += "No equivalence classes defined. Consider adding equivalences for:\n"
            report += "- Mathematically equivalent formulations\n"
            report += "- Concepts that are 'the same' under different names\n"
            report += "- Historical rediscoveries\n\n"

        # =====================================================================
        # Coherence Analysis
        # =====================================================================
        report += self._section("Coherence Analysis (Sheaf Condition)")

        coherence = self._compute_coherence_score(objects)

        report += f"**Overall Coherence Score:** {coherence['score']:.2%}\n\n"
        report += f"- Connected objects: {coherence['connected']}\n"
        report += f"- Isolated objects: {coherence['isolated']}\n\n"

        if coherence['score'] > 0.9:
            report += "The knowledge graph exhibits **high coherence**—data glues together consistently "
            report += "across different parts of the structure. This satisfies the sheaf condition.\n\n"
        elif coherence['score'] > 0.7:
            report += "The knowledge graph has **good coherence** with some isolated elements. "
            report += "Consider connecting isolated objects to improve structural integrity.\n\n"
        else:
            report += "The knowledge graph has **low coherence**—many objects are disconnected. "
            report += "This may impede evolutionary analysis and path finding.\n\n"

        if coherence['inconsistencies']:
            report += "### Detected Issues\n\n"
            for inc in coherence['inconsistencies'][:5]:
                report += f"- {inc}\n"
            report += "\n"

        # =====================================================================
        # The Four Pillars Status
        # =====================================================================
        report += self._section("The Four Pillars: Implementation Status")

        report += self._table(
            ["Pillar", "Status", "Implementation", "Coverage"],
            [
                ["Category Theory", "✅ Active", "Objects, morphisms, paths, composition", f"{stats['morphisms']} morphisms"],
                ["Homotopy Type Theory", "✅ Active", "Equivalence classes, univalence", f"{stats['equivalences']} equivalences"],
                ["Cubical Type Theory", "🔄 Partial", "Path types, (hcomp/hfill pending)", "Structure ready"],
                ["Game Theory", "🔄 Partial", "Open games defined, (Nash pending)", "Structure ready"],
            ]
        )

        report += "\n### Pillar Integration\n\n"
        report += "The four pillars work together:\n\n"
        report += "1. **Category Theory** provides the structural foundation—objects and morphisms\n"
        report += "2. **HoTT** adds equivalences, enabling flexible path finding\n"
        report += "3. **Cubical TT** will enable gap-filling via hcomp/hfill operations\n"
        report += "4. **Game Theory** will optimize path selection via Nash equilibrium\n\n"

        # =====================================================================
        # Sample Data
        # =====================================================================
        report += self._section("Sample Data")

        report += "### Recent Objects\n\n"
        sample_objects = objects[:15]
        if sample_objects:
            report += self._table(
                ["Name", "Type", "Era/Domain", "Connectivity"],
                [[o.name, o.type_name,
                  o.metadata.get("era", o.metadata.get("category", "-")),
                  len([m for m in morphisms if m.source_name == o.name or m.target_name == o.name])]
                 for o in sample_objects]
            )

        report += "\n### Recent Morphisms\n\n"
        sample_morphisms = morphisms[:15]
        if sample_morphisms:
            report += self._table(
                ["From", "Relation", "To", "Year", "Confidence"],
                [[m.source_name, m.name, m.target_name,
                  m.metadata.get("year", "-"), f"{m.confidence:.2f}"]
                 for m in sample_morphisms]
            )

        # =====================================================================
        # Embeddings Analysis
        # =====================================================================
        if self.embeddings and self.embeddings.is_available:
            report += self._section("Semantic Embedding Analysis")

            cache_stats = self.embeddings.get_cache_stats()
            report += f"**Embedding Model:** {self.embeddings.model_name}\n"
            report += f"**Dimension:** {self.embeddings.dimension}\n"
            report += f"**Cached Embeddings:** {cache_stats.get('sqlite_cache', 0)}\n\n"

            report += "Semantic embeddings enable:\n"
            report += "- Similarity-based gap detection\n"
            report += "- Semantic clustering of concepts\n"
            report += "- Embedding-space path optimization\n"
            report += "- Cross-domain analogy discovery\n\n"

        # =====================================================================
        # Future Directions
        # =====================================================================
        report += self._section("Conclusions and Future Directions")

        # Calculate overall health score
        health_score = (
            (0.3 if coherence['score'] > 0.8 else 0.15 if coherence['score'] > 0.5 else 0) +
            (0.2 if stats['equivalences'] > 0 else 0) +
            (0.2 if density > 0.01 else 0.1 if density > 0.001 else 0) +
            (0.15 if stats['morphisms'] > stats['objects'] else 0.075) +
            (0.15 if avg_degree > 2 else 0.075)
        )

        report += f"**Knowledge Graph Health Score:** {health_score:.0%}\n\n"

        if health_score > 0.8:
            report += "The knowledge graph is in **excellent condition** for categorical analysis.\n\n"
        elif health_score > 0.6:
            report += "The knowledge graph is in **good condition** with room for improvement.\n\n"
        else:
            report += "The knowledge graph needs **enrichment** for optimal categorical analysis.\n\n"

        report += "### Recommended Actions\n\n"
        report += "1. **Enrich Morphisms:** Add more relationships to increase connectivity\n"
        report += "2. **Define Equivalences:** Identify concepts that are 'the same'\n"
        report += "3. **Connect Isolates:** Link disconnected objects to the main graph\n"
        report += "4. **Compute Embeddings:** Enable semantic analysis with `python cli.py embed`\n"
        report += "5. **Run Evolution Queries:** Test path finding with `python cli.py query evolution A B`\n\n"

        report += "### Research Opportunities\n\n"
        report += "Based on this analysis, promising research directions include:\n\n"

        # Suggest specific queries based on the data
        if objects and len(objects) >= 2:
            obj1, obj2 = objects[0], objects[-1]
            report += f"- **Evolution Analysis:** How did {obj1.name} influence {obj2.name}?\n"

        if stats.get("object_types") and len(stats["object_types"]) >= 2:
            types = list(stats["object_types"].keys())
            report += f"- **Cross-Type Bridges:** How do {types[0]}s relate to {types[1]}s?\n"

        report += "- **Gap Filling:** What connections are missing but should exist?\n"
        report += "- **Equivalence Discovery:** What concepts are secretly 'the same'?\n\n"

        report += "---\n\n"
        report += "*Report generated by KOMPOSOS-III Categorical AI System*\n"
        report += "*'Phylogenetics of concepts'—tracing how ideas evolve*\n"
        report += f"*Analysis completed: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}*\n"

        return report


# =============================================================================
# CLI Commands
# =============================================================================

def cmd_init(args):
    """Initialize a new corpus directory."""
    corpus_path = Path(args.corpus) if args.corpus else Path.cwd() / "corpus"

    print(f"Initializing corpus at: {corpus_path}")
    init_corpus(corpus_path)
    print("\nCorpus initialized successfully!")
    print("\nNext steps:")
    print("  1. Add data to the corpus subdirectories")
    print("  2. Run: python cli.py load")
    print("  3. Run: python cli.py report full --output analysis.md")


def cmd_load(args):
    """Load data from corpus into the store."""
    corpus_path = Path(args.corpus) if args.corpus else Path.cwd() / "corpus"
    db_path = Path(args.db) if args.db else Path(__file__).parent / "data" / "store.db"

    print(f"Loading from corpus: {corpus_path}")
    print(f"Database: {db_path}")

    # Verify corpus
    verification = verify_corpus(corpus_path)
    existing = [k for k, v in verification.items() if v]

    if not existing:
        print("\nError: Corpus directory not found or empty.")
        print("Run 'python cli.py init' first.")
        return

    print(f"\nFound data in: {', '.join(existing)}")

    # Create store
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = create_store(db_path)

    # Load data
    loader = CorpusLoader(corpus_path)
    stats = loader.load_all(store)

    print("\nLoading complete!")
    print("\nStatistics:")
    for k, v in stats.items():
        if v > 0:
            print(f"  {k}: {v}")

    total = store.get_statistics()
    print(f"\nTotal in store: {total['objects']} objects, {total['morphisms']} morphisms")


def cmd_query(args):
    """Query the knowledge graph."""
    db_path = Path(args.db) if args.db else Path(__file__).parent / "data" / "store.db"

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        print("Run 'python cli.py load' first.")
        return

    store = create_store(db_path)

    if args.query_type == "evolution":
        if len(args.args) < 2:
            print("Usage: python cli.py query evolution SOURCE TARGET")
            return

        source, target = args.args[0], args.args[1]
        paths = store.find_paths(source, target)

        print(f"\nEvolution paths from '{source}' to '{target}':")
        print(f"Found {len(paths)} path(s)\n")

        for i, path in enumerate(paths[:10], 1):
            print(f"Path {i} (length {path.length}):")
            for mor_id in path.morphism_ids:
                mor = store.get_morphism(mor_id)
                if mor:
                    year = mor.metadata.get("year", "")
                    year_str = f" ({year})" if year else ""
                    print(f"  {mor.source_name} ─[{mor.name}{year_str}]→ {mor.target_name}")
            print()

    elif args.query_type == "equivalence":
        if len(args.args) < 2:
            print("Usage: python cli.py query equivalence OBJ1 OBJ2")
            return

        obj1, obj2 = args.args[0], args.args[1]
        result = store.are_equivalent(obj1, obj2)

        if result:
            print(f"\n'{obj1}' ≃ '{obj2}' (equivalent)")
            print(f"  Class: {result.name}")
            print(f"  Type: {result.equivalence_type}")
            print(f"  Witness: {result.witness}")
        else:
            print(f"\n'{obj1}' ≄ '{obj2}' (not equivalent)")

    elif args.query_type == "gaps":
        engine = EmbeddingsEngine()
        if not engine.is_available:
            print("Error: Embeddings not available for gap analysis")
            return

        embedder = StoreEmbedder(store, engine)
        threshold = float(args.threshold) if args.threshold else 0.3

        gaps = embedder.find_gaps(threshold)

        print(f"\nSemantic gaps (similarity < {threshold}):")
        print(f"Found {len(gaps)} gap(s)\n")

        for obj1, obj2, sim in gaps[:20]:
            print(f"  {obj1.name} ↔ {obj2.name}: {sim:.4f}")

    else:
        print(f"Unknown query type: {args.query_type}")
        print("Available: evolution, equivalence, gaps")


def cmd_report(args):
    """Generate markdown reports."""
    db_path = Path(args.db) if args.db else Path(__file__).parent / "data" / "store.db"

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        print("Run 'python cli.py load' first.")
        return

    store = create_store(db_path)

    # Initialize embeddings if available
    try:
        engine = EmbeddingsEngine()
    except Exception:
        engine = None

    generator = ReportGenerator(store, engine)

    # Generate report based on type
    if args.report_type == "evolution":
        if len(args.args) < 2:
            print("Usage: python cli.py report evolution SOURCE TARGET [--output FILE]")
            return

        source, target = args.args[0], args.args[1]
        report = generator.evolution_report(source, target)

    elif args.report_type == "gaps":
        threshold = float(args.threshold) if args.threshold else 0.3
        report = generator.gap_report(threshold)

    elif args.report_type == "equivalence":
        report = generator.equivalence_report()

    elif args.report_type == "full":
        report = generator.full_report()

    else:
        print(f"Unknown report type: {args.report_type}")
        print("Available: evolution, gaps, equivalence, full")
        return

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report, encoding='utf-8')
        print(f"Report saved to: {output_path}")
    else:
        print(report)


def cmd_stats(args):
    """Show store statistics."""
    db_path = Path(args.db) if args.db else Path(__file__).parent / "data" / "store.db"

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    store = create_store(db_path)
    stats = store.get_statistics()

    print("\nKOMPOSOS-III Store Statistics")
    print("=" * 40)
    print(f"Objects:           {stats['objects']}")
    print(f"Morphisms:         {stats['morphisms']}")
    print(f"Stored Paths:      {stats['paths']}")
    print(f"Equivalences:      {stats['equivalences']}")
    print(f"Higher Morphisms:  {stats['higher_morphisms']}")

    if stats.get("object_types"):
        print("\nObject Types:")
        for t, c in sorted(stats["object_types"].items(), key=lambda x: -x[1]):
            print(f"  {t}: {c}")

    if stats.get("morphism_types"):
        print("\nMorphism Types:")
        for t, c in sorted(stats["morphism_types"].items(), key=lambda x: -x[1]):
            print(f"  {t}: {c}")


def cmd_embed(args):
    """Compute embeddings for all objects."""
    db_path = Path(args.db) if args.db else Path(__file__).parent / "data" / "store.db"

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    store = create_store(db_path)

    print("Initializing embeddings engine...")
    engine = EmbeddingsEngine()

    if not engine.is_available:
        print("Error: Sentence Transformers not available")
        print("Install with: pip install sentence-transformers")
        return

    print(f"Model: {engine.model_name} ({engine.dimension}d)")

    embedder = StoreEmbedder(store, engine)

    print("\nComputing embeddings...")
    count = embedder.embed_all_objects(show_progress=True)

    print(f"\nEmbedded {count} objects")


def cmd_oracle(args):
    """Run Oracle predictions between two concepts."""
    db_path = Path(args.db) if args.db else Path(__file__).parent / "data" / "store.db"

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    if not ORACLE_AVAILABLE:
        print("Error: Oracle module not available")
        return

    if len(args.args) < 2:
        print("Usage: python cli.py oracle <source> <target>")
        return

    source, target = args.args[0], args.args[1]

    store = create_store(db_path)

    # Initialize embeddings
    embeddings = None
    try:
        embeddings = EmbeddingsEngine()
        if embeddings.is_available:
            embedder = StoreEmbedder(store, embeddings)
            embedder.embed_all_objects(show_progress=False)
    except Exception:
        pass

    print(f"Running Oracle: {source} -> {target}")
    print("=" * 60)

    oracle = CategoricalOracle(store, embeddings)
    result = oracle.predict(source, target)

    print(f"\nPredictions: {len(result.predictions)}")
    print(f"Coherence Score: {result.coherence_result.coherence_score:.1%}")
    print(f"Coherent: {'Yes' if result.coherence_result.is_coherent else 'No'}")
    print()

    for i, pred in enumerate(result.predictions, 1):
        print(f"[{i}] {pred.description}")
        print(f"    Strategies: {pred.strategy_name}")
        print(f"    Confidence: {pred.confidence:.1%}")
        print(f"    Reasoning: {pred.reasoning[:100]}...")
        print()

    # Save report if --report flag
    if getattr(args, 'report', False):
        generator = ReportGenerator(store, embeddings)
        report = generator.evolution_report(source, target)
        output_path = Path(f"oracle_{source}_{target}.md")
        output_path.write_text(report, encoding='utf-8')
        print(f"Report saved to: {output_path}")


def cmd_homotopy(args):
    """Analyze path homotopy between two concepts."""
    db_path = Path(args.db) if args.db else Path(__file__).parent / "data" / "store.db"

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    if not HOMOTOPY_AVAILABLE:
        print("Error: Homotopy module not available")
        return

    if len(args.args) < 2:
        print("Usage: python cli.py homotopy <source> <target>")
        return

    source, target = args.args[0], args.args[1]

    store = create_store(db_path)

    print(f"Analyzing path homotopy: {source} -> {target}")
    print("=" * 60)

    # Find paths
    paths = store.find_paths(source, target, max_length=8)

    if not paths:
        print(f"No paths found from {source} to {target}")
        return

    print(f"Found {len(paths)} paths")
    print()

    # Extract node sequences
    path_sequences = []
    for path in paths[:10]:  # Limit to 10 paths
        sequence = [source]
        for mor_id in path.morphism_ids:
            mor = store.get_morphism(mor_id)
            if mor and mor.target_name not in sequence:
                sequence.append(mor.target_name)
        path_sequences.append(sequence)

    # Run homotopy analysis
    result = check_path_homotopy(path_sequences, store)

    print(result.analysis)

    # Save report if --report flag
    if getattr(args, 'report', False):
        try:
            engine = EmbeddingsEngine()
        except:
            engine = None
        generator = ReportGenerator(store, engine)
        report = generator.evolution_report(source, target)
        output_path = Path(f"homotopy_{source}_{target}.md")
        output_path.write_text(report, encoding='utf-8')
        print(f"\nReport saved to: {output_path}")


def cmd_geo_homotopy(args):
    """Analyze geometric (Thurston-aware) path homotopy between two concepts."""
    db_path = Path(args.db) if args.db else Path(__file__).parent / "data" / "store.db"

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    if not GEOMETRIC_HOMOTOPY_AVAILABLE:
        print("Error: Geometric homotopy module not available")
        return

    if not GEOMETRY_AVAILABLE:
        print("Error: Geometry module not available (required for curvature)")
        return

    if len(args.args) < 2:
        print("Usage: python cli.py geo-homotopy <source> <target>")
        return

    source, target = args.args[0], args.args[1]

    store = create_store(db_path)

    print(f"Analyzing geometric homotopy: {source} -> {target}")
    print("(Thurston-aware path equivalence using Ricci curvature)")
    print("=" * 60)

    # Find paths
    paths = store.find_paths(source, target, max_length=8)

    if not paths:
        print(f"No paths found from {source} to {target}")
        return

    print(f"Found {len(paths)} paths")
    print()

    # Extract node sequences
    path_sequences = []
    for path in paths[:10]:  # Limit to 10 paths
        sequence = [source]
        for mor_id in path.morphism_ids:
            mor = store.get_morphism(mor_id)
            if mor and mor.target_name not in sequence:
                sequence.append(mor.target_name)
        path_sequences.append(sequence)

    # Create Ricci curvature computer
    ricci = OllivierRicciCurvature(store)

    # Run geometric homotopy analysis
    result = check_geometric_homotopy(path_sequences, ricci=ricci)

    print(result.analysis)


def cmd_stress_test(args):
    """Run quality stress tests on the Oracle system."""
    print("Running KOMPOSOS-III Stress Tests")
    print("=" * 60)
    print()

    try:
        from evaluation.stress_test import run_all_stress_tests
        results = run_all_stress_tests()
        print("\nStress tests completed.")
    except ImportError as e:
        print(f"Error: Could not import stress test module: {e}")
    except Exception as e:
        print(f"Error running stress tests: {e}")


def cmd_ricci_flow(args):
    """Run discrete Ricci flow for geometric decomposition."""
    db_path = Path(args.db) if args.db else Path(__file__).parent / "data" / "store.db"

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    if not GEOMETRY_AVAILABLE:
        print("Error: Geometry module not available")
        print("Check that geometry/flow.py exists and scipy is installed")
        return

    store = create_store(db_path)
    stats = store.get_statistics()

    max_steps = int(args.steps) if hasattr(args, 'steps') and args.steps else 30
    dt = float(args.dt) if hasattr(args, 'dt') and args.dt else 0.2

    print(f"Running Discrete Ricci Flow")
    print(f"Graph: {stats['objects']} objects, {stats['morphisms']} morphisms")
    print(f"Parameters: max_steps={max_steps}, dt={dt}")
    print("=" * 60)
    print()

    # Run Ricci flow
    result = run_ricci_flow(store, max_steps=max_steps, dt=dt)

    # Print analysis
    print()
    print(result.analysis)

    # Output to file if requested
    if hasattr(args, 'output') and args.output:
        output_path = Path(args.output)
        output_path.write_text(result.analysis, encoding='utf-8')
        print(f"\nAnalysis saved to: {output_path}")

    # Print summary
    print()
    print("=" * 60)
    print("Decomposition Summary")
    print("=" * 60)
    print(f"  Geometric Regions: {result.num_regions}")
    print(f"  Converged: {'Yes' if result.converged else 'No'}")
    print(f"  Flow Steps: {result.num_steps}")
    print(f"  Boundary Edges: {len(result.boundary_edges)}")
    print()

    for region in result.regions:
        print(f"  {region.name}: {region.size} nodes, {region.geometry_type.value}")


def cmd_curvature(args):
    """Compute Ollivier-Ricci curvature for the knowledge graph."""
    db_path = Path(args.db) if args.db else Path(__file__).parent / "data" / "store.db"

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    if not GEOMETRY_AVAILABLE:
        print("Error: Geometry module not available")
        print("Check that geometry/ricci.py exists and scipy is installed")
        return

    store = create_store(db_path)
    stats = store.get_statistics()

    print(f"Computing Ollivier-Ricci curvature for knowledge graph")
    print(f"Graph: {stats['objects']} objects, {stats['morphisms']} morphisms")
    print("=" * 60)
    print()

    # Compute curvature
    alpha = float(args.alpha) if hasattr(args, 'alpha') and args.alpha else 0.5
    result = compute_graph_curvature(store, alpha=alpha)

    # Print analysis
    print(result.analysis)

    # Output to file if requested
    if hasattr(args, 'output') and args.output:
        output_path = Path(args.output)
        output_path.write_text(result.analysis, encoding='utf-8')
        print(f"\nAnalysis saved to: {output_path}")

    # Print summary statistics
    print()
    print("=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"  Total Edges Analyzed: {result.statistics['num_edges']}")
    print(f"  Mean Curvature: {result.statistics['mean']:.4f}")
    print(f"  Std Deviation: {result.statistics['std']:.4f}")
    print(f"  Range: [{result.statistics['min']:.4f}, {result.statistics['max']:.4f}]")
    print()
    print(f"  Spherical Edges (clusters): {result.num_spherical}")
    print(f"  Hyperbolic Edges (hierarchies): {result.num_hyperbolic}")
    print(f"  Euclidean Edges (chains): {result.num_euclidean}")


def cmd_predict(args):
    """Make a single prediction about a relationship."""
    db_path = Path(args.db) if args.db else Path(__file__).parent / "data" / "store.db"

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    if len(args.args) < 2:
        print("Usage: python cli.py predict <source> <target> [relation_type]")
        return

    source = args.args[0]
    target = args.args[1]
    relation_type = args.args[2] if len(args.args) > 2 else None

    store = create_store(db_path)

    # Check if relationship already exists
    existing = [m for m in store.get_morphisms_from(source) if m.target_name == target]
    if existing:
        print(f"Existing relationship(s) found:")
        for mor in existing:
            print(f"  {mor.source_name} --[{mor.name}]--> {mor.target_name}")
        print()

    # Find paths
    paths = store.find_paths(source, target, max_length=6)

    if paths:
        print(f"Found {len(paths)} indirect path(s):")
        for i, path in enumerate(paths[:3], 1):
            print(f"  Path {i} (length {path.length}): {' -> '.join([source] + [store.get_morphism(m).target_name for m in path.morphism_ids])}")
        print()

    # Run Oracle if available
    if ORACLE_AVAILABLE:
        embeddings = None
        try:
            embeddings = EmbeddingsEngine()
            if embeddings.is_available:
                embedder = StoreEmbedder(store, embeddings)
                embedder.embed_all_objects(show_progress=False)
        except Exception:
            pass

        oracle = CategoricalOracle(store, embeddings)
        result = oracle.predict(source, target)

        if result.predictions:
            print("Oracle Predictions:")
            for pred in result.predictions:
                print(f"  [{pred.confidence:.0%}] {pred.description}")
        else:
            print("No Oracle predictions generated.")
    else:
        print("Oracle module not available for predictions.")


# =============================================================================
# ASK Command - Plain English Interface
# =============================================================================

def cmd_ask(args):
    """
    Simple interface - extracts concepts from text and generates full report.

    Usage:
        python cli.py ask "Newton Schrodinger"
        python cli.py ask "How did Planck influence Feynman?"
    """
    if not args.question:
        print("Usage: python cli.py ask \"concept1 concept2\"")
        print("\nExamples:")
        print("  python cli.py ask \"Newton Schrodinger\"")
        print("  python cli.py ask \"Planck Feynman\"")
        return

    question = " ".join(args.question)

    # Get database
    db_path = Path(args.db) if args.db else get_config().db_path
    store = create_store(db_path)

    stats = store.get_statistics()
    if stats['objects'] == 0:
        print("No data loaded. Run: python cli.py load --corpus <path>")
        return

    # Extract concepts from question
    words = question.replace("?", "").replace(".", "").replace(",", "").split()
    all_objects = [obj.name for obj in store.list_objects(limit=1000)]

    found = []
    for word in words:
        for obj_name in all_objects:
            if word.lower() == obj_name.lower() or word.lower() in obj_name.lower():
                if obj_name not in found:
                    found.append(obj_name)

    if len(found) < 2:
        print(f"Found concepts: {found if found else 'none'}")
        print(f"\nAvailable concepts:")
        for name in sorted(all_objects)[:20]:
            print(f"  - {name}")
        if len(all_objects) > 20:
            print(f"  ... and {len(all_objects) - 20} more")
        return

    source, target = found[0], found[1]
    print(f"Generating report: {source} -> {target}")

    # Use the existing ReportGenerator
    try:
        engine = EmbeddingsEngine()
    except:
        engine = None

    generator = ReportGenerator(store, engine)
    report = generator.evolution_report(source, target)

    # Output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"report_{source}_{target}.md")

    output_path.write_text(report, encoding='utf-8')
    print(f"Report saved to: {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="KOMPOSOS-III: Categorical Game-Theoretic Type-Theoretic AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py init --corpus ./my_corpus
  python cli.py load --corpus ./my_corpus
  python cli.py query evolution "Newton" "Dirac"
  python cli.py report full --output report.md
  python cli.py report evolution "Newton" "Dirac" --output evolution.md
  python cli.py oracle "Planck" "Feynman"
  python cli.py homotopy "Planck" "Feynman"
  python cli.py predict "Newton" "Einstein"
  python cli.py stress-test
  python cli.py stats
  python cli.py embed
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize a new corpus directory")
    init_parser.add_argument("--corpus", help="Path to corpus directory")

    # load command
    load_parser = subparsers.add_parser("load", help="Load data from corpus")
    load_parser.add_argument("--corpus", help="Path to corpus directory")
    load_parser.add_argument("--db", help="Path to database file")

    # query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge graph")
    query_parser.add_argument("query_type", choices=["evolution", "equivalence", "gaps"])
    query_parser.add_argument("args", nargs="*", help="Query arguments")
    query_parser.add_argument("--db", help="Path to database file")
    query_parser.add_argument("--threshold", help="Similarity threshold for gaps")

    # report command
    report_parser = subparsers.add_parser("report", help="Generate markdown reports")
    report_parser.add_argument("report_type", choices=["evolution", "gaps", "equivalence", "full"])
    report_parser.add_argument("args", nargs="*", help="Report arguments")
    report_parser.add_argument("--db", help="Path to database file")
    report_parser.add_argument("--output", "-o", help="Output file path")
    report_parser.add_argument("--threshold", help="Similarity threshold for gaps")

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show store statistics")
    stats_parser.add_argument("--db", help="Path to database file")

    # embed command
    embed_parser = subparsers.add_parser("embed", help="Compute embeddings for all objects")
    embed_parser.add_argument("--db", help="Path to database file")

    # oracle command (NEW)
    oracle_parser = subparsers.add_parser("oracle", help="Run Oracle predictions between concepts")
    oracle_parser.add_argument("args", nargs="*", help="<source> <target>")
    oracle_parser.add_argument("--db", help="Path to database file")
    oracle_parser.add_argument("--report", "-r", action="store_true", help="Save full MD report")

    # homotopy command (NEW)
    homotopy_parser = subparsers.add_parser("homotopy", help="Analyze path homotopy between concepts")
    homotopy_parser.add_argument("args", nargs="*", help="<source> <target>")
    homotopy_parser.add_argument("--db", help="Path to database file")
    homotopy_parser.add_argument("--report", "-r", action="store_true", help="Save full MD report")

    # geo-homotopy command (NEW - Thurston-aware geometric homotopy)
    geo_homotopy_parser = subparsers.add_parser("geo-homotopy", help="Analyze geometric (Thurston-aware) path homotopy")
    geo_homotopy_parser.add_argument("args", nargs="*", help="<source> <target>")
    geo_homotopy_parser.add_argument("--db", help="Path to database file")

    # predict command (NEW)
    predict_parser = subparsers.add_parser("predict", help="Predict relationship between concepts")
    predict_parser.add_argument("args", nargs="*", help="<source> <target> [relation_type]")
    predict_parser.add_argument("--db", help="Path to database file")

    # stress-test command (NEW)
    stress_parser = subparsers.add_parser("stress-test", help="Run quality stress tests")

    # curvature command (NEW - Geometry Layer)
    curvature_parser = subparsers.add_parser("curvature", help="Compute Ollivier-Ricci curvature")
    curvature_parser.add_argument("--db", help="Path to database file")
    curvature_parser.add_argument("--alpha", help="Laziness parameter (0-1, default 0.5)")
    curvature_parser.add_argument("--output", "-o", help="Output file for analysis")

    # ricci-flow command (NEW - Geometry Layer)
    flow_parser = subparsers.add_parser("ricci-flow", help="Run discrete Ricci flow for decomposition")
    flow_parser.add_argument("--db", help="Path to database file")
    flow_parser.add_argument("--steps", help="Maximum flow steps (default 30)")
    flow_parser.add_argument("--dt", help="Time step size (default 0.2)")
    flow_parser.add_argument("--output", "-o", help="Output file for analysis")

    # ask command (SIMPLE - Plain English interface)
    ask_parser = subparsers.add_parser("ask", help="Ask a question in plain English (outputs MD report)")
    ask_parser.add_argument("question", nargs="*", help="Your question or two concepts")
    ask_parser.add_argument("--db", help="Path to database file")
    ask_parser.add_argument("--output", "-o", help="Output file (default: auto-generated)")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "load":
        cmd_load(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "embed":
        cmd_embed(args)
    elif args.command == "oracle":
        cmd_oracle(args)
    elif args.command == "homotopy":
        cmd_homotopy(args)
    elif args.command == "geo-homotopy":
        cmd_geo_homotopy(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "stress-test":
        cmd_stress_test(args)
    elif args.command == "curvature":
        cmd_curvature(args)
    elif args.command == "ricci-flow":
        cmd_ricci_flow(args)
    elif args.command == "ask":
        cmd_ask(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
