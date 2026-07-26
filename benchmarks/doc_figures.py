"""Render the documentation's visualization gallery from cached Qasper vectors.

Every figure in ``docs/source/_static/viz/`` is produced here, from *real*
embeddings of real documents: a slice of the Qasper dev split (NLP papers with
their full text and native section structure) encoded by the models in
``eval_dual.MODEL_POOL``.

Nothing is embedded at run time. The dual-embedding study already wrote every
chunk/section vector for these papers to ``benchmarks/.cache/hier_embed/``, so
this script is a pure cache read -- it refuses to run if a vector is missing
rather than quietly firing thousands of requests at Ollama (pass
``--allow-embed`` to override, as ``eval_dual`` does).

Usage::

    ./.venv/Scripts/python.exe -m benchmarks.doc_figures
    ./.venv/Scripts/python.exe -m benchmarks.doc_figures --papers 60 --model egemma

Document vectors are the unit-normalised mean of a paper's chunk vectors --
the same centroid the hierarchical doc level builds -- because the study cached
chunk and section spans, not whole-document spans.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from benchmarks.config import DATA_DIR  # noqa: E402
from benchmarks.eval_dual import MODEL_POOL, ModelSpec, PrefixedEncoder, embed_model, load_units  # noqa: E402
from benchmarks.eval_hierarchical import _unit  # noqa: E402
from benchmarks.qasper_data import load_qasper  # noqa: E402

logger = logging.getLogger("benchmarks.doc_figures")

OUT_DIR = _ROOT / "docs" / "source" / "_static" / "viz"

# The gallery model, and the pool used for the cross-model comparison. nomic is
# omitted from the comparison: the study culled it, so showing it here would
# advertise an encoder we no longer recommend.
GALLERY_MODEL = "egemma"
COMPARE_MODELS = ("egemma", "openai", "qwen3", "arctic")

N_PAPERS = 60
N_MAP_PAPERS = 6  # papers whose sections make up the coloured embedding map
N_MATRIX_DOCS = 14  # heatmap stays under plot_similarity_matrix's 20-doc annotation limit
N_GRAPH_DOCS = 18  # full titles need room; a force-directed core packs tightly
N_TOPIC_CLUSTERS = 4
# Silhouette prefers k=2 on this corpus, but scores are low at every k and the
# map shows finer structure than two blobs; see fig_clusters.
N_CLUSTER_K = 5
N_COMPARE_QUERIES = 3
N_COMPARE_DOCS = 30

FIG_DPI = 130  # 150 (the library default) is ~2x the bytes for no visible gain in the docs theme


# ---------------------------------------------------------------------------
# Corpus + cached vectors
# ---------------------------------------------------------------------------


@dataclass
class Gallery:
    """Everything the figures need, for one embedding model."""

    model_key: str
    doc_ids: List[str]  # papers, in deterministic corpus order
    doc_vectors: np.ndarray  # (D, dim) unit-normalised chunk centroids
    chunk_vectors: Dict[str, np.ndarray]  # doc id -> (C, dim) unit-normalised, in document order
    section_vectors: Dict[str, np.ndarray]  # doc id -> (S, dim) unit-normalised, in document order
    chunk_headings: Dict[str, List[str]]  # doc id -> the section heading each chunk falls under
    query_ids: List[str]
    query_vectors: np.ndarray  # (Q, dim) unit-normalised, embedded with the model's *query* prefix
    query_texts: Dict[str, str]
    query_doc: Dict[str, str]  # query id -> the paper its answer is grounded in
    titles: Dict[str, str]


def _titles(split: str) -> Dict[str, str]:
    """Paper id -> title, read straight from the Qasper JSON.

    ``load_qasper`` returns only rendered body text; the title lives in the raw
    record and makes far better axis labels than an arXiv id.
    """
    matches = sorted((DATA_DIR / "qasper").glob(f"*{split}*.json"))
    if not matches:
        raise FileNotFoundError(f"No Qasper {split!r} JSON under {DATA_DIR / 'qasper'}")
    data = json.loads(matches[0].read_text(encoding="utf-8"))
    return {pid: (rec.get("title") or pid).strip() for pid, rec in data.items()}


def _short(title: str, width: int) -> str:
    """A single-line label of at most ``width`` chars, cut on a word boundary."""
    flat = re.sub(r"\s+", " ", title).strip()
    if len(flat) <= width:
        return flat
    cut = flat[: width - 1]
    space = cut.rfind(" ")
    if space > width // 2:
        cut = cut[:space]
    return cut + "…"


def _group(vectors: np.ndarray, owners: Sequence[str]) -> Dict[str, np.ndarray]:
    """Split row-blocks of ``vectors`` by ``owners``, preserving document order."""
    out: Dict[str, List[np.ndarray]] = {}
    for row, owner in zip(vectors, owners, strict=True):
        out.setdefault(owner, []).append(row)
    return {k: _unit(np.vstack(v)) for k, v in out.items()}


def _group_list(values: Sequence[str], owners: Sequence[str]) -> Dict[str, List[str]]:
    """``_group`` for plain lists -- keeps per-chunk labels aligned with per-chunk vectors."""
    out: Dict[str, List[str]] = {}
    for value, owner in zip(values, owners, strict=True):
        out.setdefault(owner, []).append(value)
    return out


def _section_headings(corpus: Dict[str, str]) -> Dict[str, str]:
    """Section qrel id -> its heading text, over every document in the corpus.

    ``load_units`` records which section each chunk falls in, but only as an
    opaque id; the heading itself is what makes a readable chunk label.
    """
    from benchmarks.eval_hierarchical import _detect_sections
    from benchmarks.superdocs import section_qrel_id

    headings: Dict[str, str] = {}
    for doc_id, text in corpus.items():
        for sec in _detect_sections(text):
            if sec.heading is not None:
                headings[section_qrel_id(doc_id, sec.index)] = sec.heading
    return headings


def build_gallery(model_key: str, n_papers: int, split: str, allow_embed: bool) -> Gallery:
    """Load ``n_papers`` Qasper papers and pull their cached vectors for one model."""
    spec: ModelSpec = MODEL_POOL[model_key]
    bench = load_qasper(split=split, max_papers=n_papers)
    units = load_units(bench, None)

    if not allow_embed:
        doc_enc = PrefixedEncoder(spec, spec.doc_prefix)
        misses = doc_enc.count_misses(units.chunk_texts)[1] + doc_enc.count_misses(units.section_texts)[1]
        if misses:
            raise SystemExit(
                f"[{model_key}] {misses} span(s) are not in benchmarks/.cache/hier_embed/. "
                "Re-run the dual-embedding embed step, or pass --allow-embed to encode them now."
            )

    vectors, stats = embed_model(spec, units)
    assert vectors is not None  # dry_run is never set here
    logger.info("[%s] %s", model_key, stats)

    chunks = _group(vectors.chunks, units.chunk_doc)
    sections = _group(vectors.sections, units.section_doc)
    doc_ids = list(bench.corpus)
    # Document vector = unit-normalised mean of the paper's chunk vectors.
    doc_vectors = _unit(np.vstack([chunks[d].mean(axis=0) for d in doc_ids]))

    headings = _section_headings(bench.corpus)
    chunk_headings = _group_list([headings.get(s, "") for s in units.chunk_section], units.chunk_doc)

    return Gallery(
        model_key=model_key,
        doc_ids=doc_ids,
        doc_vectors=doc_vectors,
        chunk_vectors=chunks,
        section_vectors=sections,
        chunk_headings=chunk_headings,
        query_ids=list(units.query_ids),
        query_vectors=_unit(vectors.queries),
        query_texts=dict(zip(units.query_ids, units.query_texts, strict=True)),
        query_doc={q: next(iter(bench.doc_qrels[q])) for q in units.query_ids},
        titles=_titles(split),
    )


# ---------------------------------------------------------------------------
# Threshold helpers. Ribbon/edge thresholds are derived from the data rather
# than hard-coded, so a different paper slice still yields a legible figure.
# ---------------------------------------------------------------------------


def _threshold_for(values: np.ndarray, target: int, floor: float = 0.0) -> float:
    """The cutoff that keeps roughly ``target`` of ``values``, never below ``floor``."""
    flat = np.sort(np.asarray(values, dtype=float).ravel())[::-1]
    if flat.size == 0:
        return floor
    cut = float(flat[min(target, flat.size) - 1])
    return round(max(cut, floor), 2)


def _sim(a: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
    """Cosine similarity of unit-normalised rows."""
    return a @ (a if b is None else b).T


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _rel(path: Path) -> str:
    """Repo-relative path for logging; absolute for the scratch panel files."""
    return str(path.relative_to(_ROOT)) if path.is_relative_to(_ROOT) else str(path)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=FIG_DPI, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    logger.info("wrote %s (%.0f KB)", _rel(path), path.stat().st_size / 1024)


def _doc_matrix(g: Gallery, doc_ids: Sequence[str], labels: Sequence[str]):
    """A ``DocumentSimilarityMatrix`` over a subset of papers."""
    from localvectordb.core import DocumentSimilarityMatrix

    idx = [g.doc_ids.index(d) for d in doc_ids]
    emb = g.doc_vectors[idx]
    return DocumentSimilarityMatrix(matrix=_sim(emb), doc_ids=list(labels), embeddings=emb)


def _topic_order(g: Gallery, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """K-means over every paper; returns (labels, distance-to-own-centroid)."""
    from localvectordb.visualization import cluster_embeddings

    res = cluster_embeddings(g.doc_vectors, n_clusters=k)
    assert res.centroids is not None
    dist = np.linalg.norm(g.doc_vectors - res.centroids[res.labels], axis=1)
    return res.labels, dist


def fig_embedding_map(g: Gallery, out: Path) -> None:
    """Sections of a handful of papers, coloured by their parent paper.

    The interesting claim is structural, not decorative: sections of one paper
    land near each other without ever being told they belong together.
    """
    from localvectordb.visualization import plot_embedding_map, reduce_dimensions

    picked = sorted(g.doc_ids, key=lambda d: -len(g.section_vectors.get(d, [])))[:N_MAP_PAPERS]
    picked = sorted(picked, key=g.doc_ids.index)

    rows, labels, ids = [], [], []
    for d in picked:
        secs = g.section_vectors[d]
        rows.append(secs)
        labels.extend([_short(g.titles.get(d, d), 34)] * len(secs))
        ids.extend(str(i) for i in range(len(secs)))

    emb = np.vstack(rows)
    projection = reduce_dimensions(emb, method="tsne", doc_ids=ids, perplexity=12, random_state=42, init="pca")
    fig = plot_embedding_map(
        projection,
        color_by=labels,
        title=f"Section embedding map -- {N_MAP_PAPERS} Qasper papers ({g.model_key}, t-SNE)",
        figsize=(11, 8),
    )
    _save(fig, out / "viz_embedding_map.png")


def fig_clusters(g: Gallery, out: Path) -> None:
    """Every paper as a point, labelled by title and coloured by cluster.

    ``find_optimal_clusters`` picks k=2 here, and silhouette scores are low at
    every k (peak 0.076) -- a single-domain corpus is a gradient, not disjoint
    topics. Taking the silhouette optimum at face value hides real structure the
    map plainly shows, so the figure uses a larger k and says so.
    """
    from localvectordb.visualization import cluster_embeddings, find_optimal_clusters, plot_clusters, reduce_dimensions

    auto_k = find_optimal_clusters(g.doc_vectors)
    clusters = cluster_embeddings(g.doc_vectors, n_clusters=N_CLUSTER_K)
    logger.info("clusters: find_optimal_clusters -> k=%d; figure uses k=%d", auto_k, N_CLUSTER_K)

    ids = [_short(g.titles.get(d, d), 24) for d in g.doc_ids]
    projection = reduce_dimensions(
        g.doc_vectors, method="tsne", doc_ids=ids, perplexity=10, random_state=42, init="pca"
    )
    fig = plot_clusters(
        projection,
        clusters,
        title=f"{len(g.doc_ids)} Qasper papers, k={N_CLUSTER_K} -- {g.model_key}, t-SNE",
        figsize=(15, 11),
    )
    # Titles are annotated to the right of their point, so the rightmost ones sit
    # flush against the frame without a little extra horizontal room.
    fig.axes[0].margins(x=0.14)
    _save(fig, out / "viz_clusters.png")


def _matrix_docs(g: Gallery, n: int) -> List[str]:
    """``n`` papers sampled evenly from the topic clusters, grouped by cluster.

    Picking the papers nearest each cluster centre gives the heatmap visible
    block structure instead of the uniform wash a random NLP-paper sample
    produces. Clusters are drawn round-robin so a small cluster cannot short the
    total; the result is exactly ``n`` papers, still grouped by cluster.
    """
    labels, dist = _topic_order(g, N_TOPIC_CLUSTERS)
    ranked = [
        list(np.flatnonzero(labels == c)[np.argsort(dist[np.flatnonzero(labels == c)])])
        for c in range(N_TOPIC_CLUSTERS)
    ]
    taken: List[List[int]] = [[] for _ in ranked]
    while sum(map(len, taken)) < min(n, len(g.doc_ids)):
        progressed = False
        for c, pool in enumerate(ranked):
            if sum(map(len, taken)) >= n:
                break
            if len(taken[c]) < len(pool):
                taken[c].append(pool[len(taken[c])])
                progressed = True
        if not progressed:  # every cluster exhausted
            break
    return [g.doc_ids[i] for group in taken for i in group]


def fig_similarity_matrix(g: Gallery, out: Path) -> None:
    """Annotated pairwise heatmap over a topic-stratified subset."""
    from localvectordb.visualization import plot_similarity_matrix

    docs = _matrix_docs(g, N_MATRIX_DOCS)
    matrix = _doc_matrix(g, docs, [_short(g.titles.get(d, d), 40) for d in docs])
    fig = plot_similarity_matrix(
        matrix,
        title=f"Pairwise document similarity -- {len(docs)} Qasper papers ({g.model_key})",
    )
    _save(fig, out / "viz_similarity_matrix.png")


def fig_similarity_graph(g: Gallery, out: Path) -> None:
    """Papers as nodes, strongest similarities as edges."""
    from localvectordb.visualization import build_similarity_graph, plot_similarity_graph

    docs = _matrix_docs(g, N_GRAPH_DOCS)
    matrix = _doc_matrix(g, docs, [_short(g.titles.get(d, d), 42) for d in docs])

    # Keep roughly the top 18% of pairs so the graph reads as clusters, not a mesh.
    upper = matrix.matrix[np.triu_indices(len(docs), k=1)]
    threshold = _threshold_for(upper, max(8, int(0.18 * upper.size)))
    n_edges = len(build_similarity_graph(matrix, threshold=threshold)["edges"])
    logger.info("similarity graph: threshold=%.2f -> %d edges over %d nodes", threshold, n_edges, len(docs))

    fig = plot_similarity_graph(
        matrix,
        threshold=threshold,
        title=f"Document similarity graph -- {len(docs)} papers, edges at cosine >= {threshold:.2f} ({g.model_key})",
        figsize=(13, 10),
    )
    fig.axes[0].margins(0.12)
    _save(fig, out / "viz_similarity_graph.png")


def _chunk_matrix(g: Gallery, doc_1: str, doc_2: str, labels: bool = False):
    """Chunk-level similarity matrix; ``labels`` puts short titles on the tracks.

    ``doc_id_1``/``doc_id_2`` are what plot_synteny and plot_chord print as the
    track/figure captions, and a bare arXiv id is not a caption anyone can read.
    """
    from localvectordb.core import ChunkSimilarityMatrix

    a, b = g.chunk_vectors[doc_1], g.chunk_vectors[doc_2]

    def name(d: str) -> str:
        return _short(g.titles.get(d, d), 46) if labels else d

    return ChunkSimilarityMatrix(
        matrix=_sim(a, b),
        doc_id_1=name(doc_1),
        doc_id_2=name(doc_2),
        chunk_indices_1=list(range(len(a))),
        chunk_indices_2=list(range(len(b))),
    )


def fig_synteny(g: Gallery, out: Path) -> None:
    """The most similar pair of substantial papers, aligned chunk by chunk."""
    from localvectordb.visualization import plot_synteny

    sizes = {d: len(g.chunk_vectors[d]) for d in g.doc_ids}
    eligible = [d for d in g.doc_ids if 10 <= sizes[d] <= 22]
    idx = [g.doc_ids.index(d) for d in eligible]
    sims = _sim(g.doc_vectors[idx])
    np.fill_diagonal(sims, -np.inf)
    i, j = np.unravel_index(np.argmax(sims), sims.shape)
    doc_1, doc_2 = eligible[int(i)], eligible[int(j)]
    logger.info("synteny pair: %s <-> %s (cosine %.3f)", doc_1, doc_2, sims[i, j])

    cm = _chunk_matrix(g, doc_1, doc_2, labels=True)
    # Two papers on the same topic are similar almost everywhere, so a threshold
    # that admits half the pairs draws an unreadable wash. Keep only the top
    # ~12% -- the ribbons that survive are the genuine section-to-section matches.
    threshold = _threshold_for(cm.matrix, max(8, int(0.12 * cm.matrix.size)), floor=0.5)
    fig = plot_synteny(
        cm,
        similarity_threshold=threshold,
        orientation="horizontal",
        labels_1=[_short(h, 26) for h in g.chunk_headings[doc_1]],
        labels_2=[_short(h, 26) for h in g.chunk_headings[doc_2]],
        title=f"Chunk-level synteny -- ribbons at cosine >= {threshold:.2f} ({g.model_key})",
    )
    _save(fig, out / "viz_synteny.png")


def fig_chord(g: Gallery, out: Path) -> None:
    """One paper's internal self-similarity as a Circos-style chord diagram."""
    from localvectordb.visualization import plot_chord

    doc = max(g.doc_ids, key=lambda d: (len(g.chunk_vectors[d]) <= 30, len(g.chunk_vectors[d])))
    cm = _chunk_matrix(g, doc, doc)
    n = cm.matrix.shape[0]

    # Match plot_chord's own filter (upper triangle, >= min_chunk_distance apart)
    # so the target chord count is what actually gets drawn.
    min_distance = 3
    i, j = np.triu_indices(n, k=min_distance)
    threshold = _threshold_for(cm.matrix[i, j], 25, floor=0.5)
    logger.info("chord: %s, %d chunks, threshold=%.2f", doc, n, threshold)

    fig = plot_chord(
        cm,
        similarity_threshold=threshold,
        min_chunk_distance=min_distance,
        # Arcs carry the section each chunk sits in, so a chord reads as
        # "the evaluation section talks about the same thing as the intro"
        # rather than "chunk 17 resembles chunk 3".
        labels=[f"{i}  {_short(h, 30)}" for i, h in enumerate(g.chunk_headings[doc])],
        title=(
            f"Self-similarity: {_short(g.titles.get(doc, doc), 56)}\n"
            f"{n} chunks, chords at cosine >= {threshold:.2f} ({g.model_key})"
        ),
    )
    _save(fig, out / "viz_chord.png")


# ---------------------------------------------------------------------------
# Cross-model comparison
# ---------------------------------------------------------------------------


def _pick_queries(g: Gallery, n: int) -> List[str]:
    """``n`` *topically specific* questions, one per cluster.

    Qasper is full of generic questions -- "What baselines are used for
    comparison?" applies to every paper in the corpus, so it embeds to the middle
    of the space and overlays nothing useful. Score each question by how much
    closer it sits to the paper it was written about than to the corpus average,
    and keep the sharpest; those are the ones with real topic words in them.

    One per cluster, because three questions about the same corner of the space
    would land on top of each other.
    """
    labels, _ = _topic_order(g, N_TOPIC_CLUSTERS)
    cluster_of = {d: int(labels[i]) for i, d in enumerate(g.doc_ids)}
    row_of = {d: i for i, d in enumerate(g.doc_ids)}

    sims = g.query_vectors @ g.doc_vectors.T  # (Q, D)
    specificity = {}
    for qi, qid in enumerate(g.query_ids):
        gold = g.query_doc.get(qid)
        if gold in row_of:
            specificity[qid] = float(sims[qi, row_of[gold]] - sims[qi].mean())

    chosen: List[str] = []
    for c in range(N_TOPIC_CLUSTERS):
        if len(chosen) >= n:
            break
        pool = [q for q in specificity if cluster_of.get(g.query_doc[q], -1) == c and 25 <= len(g.query_texts[q]) <= 75]
        if pool:
            chosen.append(max(pool, key=lambda q: (specificity[q], q)))
    return chosen


def fig_model_comparison(galleries: Dict[str, Gallery], out: Path) -> None:
    """The same papers and queries under four encoders, on one shared clustering.

    Each panel is an independent PCA of that model's own vectors; the colours
    are held fixed at the reference model's k-means labels. Where a panel keeps
    its colours separated, that encoder agrees about which papers belong
    together -- where they interleave, it does not.

    The queries are the payload: each is embedded by *that* panel's model with
    that model's query prefix, then projected into the same space, so the plot
    shows where each encoder puts a question relative to the documents that
    answer it. Dot size scales with relevance to the queries.
    """
    from PIL import Image

    from localvectordb.visualization import cluster_embeddings, plot_embedding_map, reduce_dimensions
    from localvectordb.visualization.types import QueryOverlay

    ref = galleries[GALLERY_MODEL]
    clusters = cluster_embeddings(ref.doc_vectors, n_clusters=N_TOPIC_CLUSTERS)
    # Every paper's title on a 60-point panel is unreadable at grid scale; a
    # topic-stratified subset keeps the comparison honest and the labels legible.
    subset = _matrix_docs(ref, N_COMPARE_DOCS)
    rows = [ref.doc_ids.index(d) for d in subset]
    colour_by = [f"Cluster {clusters.labels[i]}" for i in rows]
    ids = [_short(ref.titles.get(d, d), 22) for d in subset]
    query_ids = _pick_queries(ref, N_COMPARE_QUERIES)
    logger.info("query overlays: %s", [ref.query_texts[q] for q in query_ids])

    tmp = Path(tempfile.mkdtemp(prefix="lvdb-viz-"))
    try:
        panels = []
        for key in COMPARE_MODELS:
            g = galleries[key]
            docs = g.doc_vectors[rows]
            projection = reduce_dimensions(docs, method="pca", doc_ids=ids)
            var = projection.explained_variance
            caption = f"{MODEL_POOL[key].model} ({g.doc_vectors.shape[1]}d)"
            if var is not None:
                caption += f" -- PC1+PC2 = {100 * float(var[:2].sum()):.0f}% variance"

            overlays = []
            for qid in query_ids:
                q_vec = g.query_vectors[g.query_ids.index(qid)]
                scores = docs @ q_vec
                # Absolute cosine differs per encoder, so raw scores would size
                # the dots differently in every panel for no meaningful reason.
                # Rank within the panel instead: the *relative* pull is the point.
                spread = float(scores.max() - scores.min()) or 1.0
                overlays.append(
                    QueryOverlay(
                        query_text=_short(g.query_texts[qid], 34),
                        query_embedding=q_vec,
                        scores=(scores - scores.min()) / spread,
                    )
                )

            fig = plot_embedding_map(projection, color_by=colour_by, queries=overlays, title=caption, figsize=(10, 8))
            path = tmp / f"{key}.png"
            _save(fig, path)
            panels.append(Image.open(path).convert("RGB"))

        w = max(p.width for p in panels)
        h = max(p.height for p in panels)
        grid = Image.new("RGB", (2 * w, 2 * h), "white")
        for n, panel in enumerate(panels):
            grid.paste(panel, ((n % 2) * w, (n // 2) * h))
        dest = out / "viz_model_compare.png"
        grid.save(dest, optimize=True)
        logger.info("wrote %s (%.0f KB)", _rel(dest), dest.stat().st_size / 1024)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def fig_model_agreement(galleries: Dict[str, Gallery], out: Path) -> None:
    """Do the encoders agree about *which* papers are alike?

    For each model, take the off-diagonal of its paper-by-paper cosine matrix
    and correlate it (Spearman) with every other model's.

    Drawn directly rather than through ``plot_similarity_matrix``: the axes are
    models rather than documents, the quantity is a rank correlation rather than
    a cosine, and the interesting range is 0.8-1.0 -- all three of which that
    function hard-codes the other way.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    keys = list(COMPARE_MODELS)
    iu = np.triu_indices(len(galleries[keys[0]].doc_ids), k=1)
    profiles = np.vstack([_sim(galleries[k].doc_vectors)[iu] for k in keys])
    rho = np.atleast_2d(np.asarray(spearmanr(profiles, axis=1).statistic, dtype=float))
    logger.info("cross-model Spearman rho: min off-diagonal %.3f", rho[~np.eye(len(keys), dtype=bool)].min())

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(rho, cmap="YlOrRd", vmin=0.8, vmax=1.0)
    fig.colorbar(im, ax=ax, label="Spearman rho")
    ax.set_xticks(range(len(keys)), [MODEL_POOL[k].model for k in keys], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(keys)), [MODEL_POOL[k].model for k in keys], fontsize=9)
    for a in range(len(keys)):
        for b in range(len(keys)):
            # Mid-range cells sit on light orange; white-on-white is unreadable.
            colour = "white" if rho[a, b] > 0.97 else "black"
            ax.text(b, a, f"{rho[a, b]:.2f}", ha="center", va="center", fontsize=10, color=colour)
    ax.set_title(f"Cross-model agreement over all {iu[0].size} document pairs")
    fig.tight_layout()
    _save(fig, out / "viz_model_agreement.png")


# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--papers", type=int, default=N_PAPERS, help=f"Qasper papers to load (default {N_PAPERS}).")
    parser.add_argument("--split", default="dev", help="Qasper split (default dev).")
    parser.add_argument("--model", default=GALLERY_MODEL, choices=sorted(MODEL_POOL), help="Gallery model.")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="Output directory.")
    parser.add_argument("--skip-comparison", action="store_true", help="Gallery figures only.")
    parser.add_argument("--allow-embed", action="store_true", help="Encode spans missing from the cache.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # The chunker warns per over-long chunk; hundreds of those bury the real log.
    logging.getLogger("localvectordb.chunking").setLevel(logging.ERROR)

    import matplotlib

    matplotlib.use("Agg")

    args.out.mkdir(parents=True, exist_ok=True)

    keys = [args.model] if args.skip_comparison else sorted({args.model, *COMPARE_MODELS})
    galleries = {k: build_gallery(k, args.papers, args.split, args.allow_embed) for k in keys}
    g = galleries[args.model]
    logger.info(
        "gallery: %d papers, %d chunks, %d sections (%s)",
        len(g.doc_ids),
        sum(len(v) for v in g.chunk_vectors.values()),
        sum(len(v) for v in g.section_vectors.values()),
        MODEL_POOL[args.model].model,
    )

    fig_embedding_map(g, args.out)
    fig_clusters(g, args.out)
    fig_similarity_matrix(g, args.out)
    fig_similarity_graph(g, args.out)
    fig_synteny(g, args.out)
    fig_chord(g, args.out)
    if not args.skip_comparison:
        fig_model_comparison(galleries, args.out)
        fig_model_agreement(galleries, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
