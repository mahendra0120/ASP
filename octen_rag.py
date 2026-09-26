"""
octen_rag.py
─────────────────────────────────────────────────────────────────
Real embedding-based RAG using the Octen embedding model
(HuggingFace transformers, via sentence-transformers) and LangChain's
FAISS vector store. Replaces the stdlib TF-IDF retriever previously
used in mcp_server.py's rag_search tool.

Verified against installed packages:
  langchain==1.3.11
  langchain-huggingface (HuggingFaceEmbeddings wraps SentenceTransformer
    directly: self._client = SentenceTransformer(model_name, **model_kwargs))
  langchain_community.vectorstores.FAISS (still functional despite the
    deprecation notice pushing toward standalone packages — there is no
    trustworthy standalone FAISS integration package as of this writing;
    a "langchain-faiss" package exists on PyPI but ships an EMPTY module
    with no public API. Don't use it.)

Octen models (confirmed real, HuggingFace Hub):
  Octen/Octen-Embedding-0.6B  — 1024-dim, fine-tuned from Qwen3-Embedding-0.6B
  Octen/Octen-Embedding-4B    — 2560-dim, fine-tuned from Qwen3-Embedding-4B
  Octen/Octen-Embedding-8B    — 4096-dim, fine-tuned from Qwen3-Embedding-8B
  All use last-token pooling + L2 normalization. #1 on RTEB leaderboard
  for open embedding models as of the model card's last update.

This module replaces SKILL-FILE-DRIVEN retrieval (the Pydantic AI agent
calling rag_search as an MCP tool based on instructions in
skills/rag_retrieval.md) with a CODE-DRIVEN retrieval pipeline: loader →
splitter → embed → FAISS index → similarity search. The keyword-trigger
mechanism (check_keyword_triggers) and the A2A delegation to the RAG
Agent are UNCHANGED — only what happens inside rag_search() changes.

Corpus source: forensic_knowledge/*.md (the case-agnostic forensic
pathology reference material shipped in this repo — gunshot wounds,
postmortem changes, thermal injuries, asphyxiation, sharp force injury,
etc.) is loaded directly, one Document per chunk, tagged with the
originating filename as `source`. This replaced an earlier
skills/rag_corpus.json format; that file never actually existed in
this repo, so this is the corpus's real first implementation, not a
migration.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Primary corpus: every markdown file in forensic_knowledge/. Each file
# is one self-contained topical reference (e.g. Gunshot_wounds.md,
# asphyxiation.md) — exactly the material a "evidence of trauma" style
# finding in a Forensic agent report should be grounded against.
FORENSIC_KNOWLEDGE_DIR = Path(__file__).parent / "forensic_knowledge"

# Legacy/optional: a hand-authored {text, source} JSON corpus, if present,
# is merged in alongside the forensic_knowledge markdown files. Nothing
# ships here by default.
CORPUS_PATH = Path(__file__).parent / "skills" / "rag_corpus.json"
INDEX_DIR   = Path(__file__).parent / ".octen_faiss_index"

OCTEN_MODEL_ID = os.getenv("OCTEN_MODEL_ID", "Octen/Octen-Embedding-0.6B")


# ════════════════════════════════════════════════════════════════
#  Embeddings — Octen via langchain_huggingface
# ════════════════════════════════════════════════════════════════

def _patch_legacy_normalize_module() -> None:
    """
    Work around a checkpoint/library incompatibility, independent of
    anything in this file's own code.

    Some Hugging Face SentenceTransformer checkpoints (Octen included)
    still ship a saved config for their Normalize pipeline module using
    an older sentence-transformers convention that included a
    'normalize_embeddings' key. Current sentence-transformers versions
    define Normalize.__init__() with NO parameters (normalization is
    now unconditional/internal), so loading such a checkpoint raises:
        TypeError: Normalize.__init__() got an unexpected keyword
        argument 'normalize_embeddings'
    This can't be fixed from the caller side (it happens inside
    SentenceTransformer's own module-loading code, before any of our
    encode_kwargs/model_kwargs are even consulted), so we patch
    Normalize.__init__ to accept and discard legacy kwargs instead of
    pinning to a specific older sentence-transformers version.
    """
    try:
        from sentence_transformers.base.modules.normalize import Normalize
    except ImportError:
        return  # older/newer sentence-transformers layout; nothing to patch
    if getattr(Normalize, "_octen_legacy_patch_applied", False):
        return
    _original_init = Normalize.__init__

    def _patched_init(self, *args, **kwargs):
        _original_init(self)  # ignore legacy kwargs like normalize_embeddings

    Normalize.__init__ = _patched_init
    Normalize._octen_legacy_patch_applied = True


def build_embeddings(
    model_id: str = OCTEN_MODEL_ID,
    device: str = "cpu",
):
    """
    Build a LangChain Embeddings object backed by an Octen model.

    HuggingFaceEmbeddings wraps sentence_transformers.SentenceTransformer
    directly:  self._client = SentenceTransformer(model_id, **model_kwargs)
    Octen models support that constructor pattern natively per their
    HF model cards, so no custom pooling code is needed here — Octen
    ships a configured SentenceTransformer wrapper that already does
    last-token pooling + normalization internally.

    Args:
        model_id: e.g. "Octen/Octen-Embedding-0.6B" (1024-dim, fastest)
                       "Octen/Octen-Embedding-4B"   (2560-dim)
                       "Octen/Octen-Embedding-8B"   (4096-dim, best quality)
        device:   "cpu", "cuda", "cuda:0", etc.
    """
    _patch_legacy_normalize_module()

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={
            "device": device,
            "trust_remote_code": True,   # Octen ships custom modeling code
        },
        # No encode_kwargs — Octen's SentenceTransformer wrapper already
        # normalizes internally (see docstring above). Passing
        # normalize_embeddings=True explicitly here breaks on newer
        # sentence-transformers versions, where the Normalize module's
        # __init__() no longer accepts that kwarg (normalization is
        # baked into the model's own pipeline instead).
    )


# ════════════════════════════════════════════════════════════════
#  Corpus loading + chunking
# ════════════════════════════════════════════════════════════════

def load_forensic_knowledge_documents(
    knowledge_dir: Path = FORENSIC_KNOWLEDGE_DIR,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> list[Document]:
    """
    Load every *.md file under forensic_knowledge/ and split each into
    LangChain Document chunks, tagged with the source filename (e.g.
    "Gunshot_wounds.md") so retrieval results can cite which reference
    a chunk came from.

    Each chunk also gets a `chunk_index` (its position within that
    file, in original reading order) — main.py's format_rag_section
    uses this to re-sort retrieved chunks back into document order
    instead of leaving them in FAISS's similarity-score order, which
    otherwise interleaves unrelated topics arbitrarily and reads as
    disjointed.

    Explicit `separators` (sentence-ending punctuation before falling
    back to a bare space) bias the splitter toward cutting at sentence
    boundaries rather than mid-sentence — chunk_size=800 (the old
    default) was tight enough that dense forensic-pathology prose,
    which often runs long without a paragraph break, would routinely
    hit the split point mid-sentence. 1200 gives more room to find a
    natural break; format_rag_section also visibly marks with an
    ellipsis any chunk that still doesn't land on one, so a genuine
    excerpt reads as an intentional excerpt rather than an unexplained
    cutoff.
    """
    if not knowledge_dir.exists():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )

    docs: list[Document] = []
    for md_path in sorted(knowledge_dir.glob("*.md")):
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        for i, chunk in enumerate(splitter.split_text(text)):
            docs.append(Document(
                page_content=chunk,
                metadata={"source": md_path.name, "chunk_index": i},
            ))
    return docs


def load_json_corpus_documents(
    corpus_path: Path = CORPUS_PATH,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[Document]:
    """
    Optional legacy corpus: {text, source} objects from a JSON file, if
    one has been placed at skills/rag_corpus.json. Not required — the
    forensic_knowledge/*.md files are the real corpus.
    """
    if not corpus_path.exists():
        return []

    raw = json.loads(corpus_path.read_text(encoding="utf-8"))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    docs: list[Document] = []
    for entry in raw:
        text   = entry.get("text", "")
        source = entry.get("source", "unknown")
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata={"source": source}))
    return docs


def load_corpus_documents(
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[Document]:
    """
    Full corpus: forensic_knowledge/*.md (primary) + skills/rag_corpus.json
    (optional extra, if present). This is what get_vectorstore() indexes.
    """
    docs = load_forensic_knowledge_documents(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    )
    docs += load_json_corpus_documents(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    )
    return docs


# ════════════════════════════════════════════════════════════════
#  Vector store — built once, cached on disk, reused across calls
# ════════════════════════════════════════════════════════════════

_vectorstore = None  # module-level cache; reset by rebuild_index()


def get_vectorstore(
    embeddings=None,
    force_rebuild: bool = False,
):
    """
    Return the FAISS vector store, building it from forensic_knowledge/*.md
    (+ optional skills/rag_corpus.json) on first call, or loading a
    cached index from disk if present.

    Note: FAISS.save_local / load_local pickle metadata — only load
    indexes you built yourself, never an index from an untrusted source.
    """
    global _vectorstore
    from langchain_community.vectorstores import FAISS

    if _vectorstore is not None and not force_rebuild:
        return _vectorstore

    embeddings = embeddings or build_embeddings()

    if INDEX_DIR.exists() and not force_rebuild:
        _vectorstore = FAISS.load_local(
            str(INDEX_DIR), embeddings,
            allow_dangerous_deserialization=True,   # safe: we wrote this index ourselves
        )
        return _vectorstore

    docs = load_corpus_documents()
    if not docs:
        # No corpus yet — return an empty-but-valid store so callers
        # get {results: [], corpus_size: 0} instead of crashing.
        docs = [Document(page_content="", metadata={"source": "empty"})]

    _vectorstore = FAISS.from_documents(docs, embeddings)
    INDEX_DIR.mkdir(exist_ok=True)
    _vectorstore.save_local(str(INDEX_DIR))
    return _vectorstore


def rebuild_index(embeddings=None) -> None:
    """Force a full re-embed + re-index of the corpus (forensic_knowledge/*.md
    + optional skills/rag_corpus.json). Call this after editing either."""
    import shutil
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    get_vectorstore(embeddings=embeddings, force_rebuild=True)


def clear_index_cache() -> bool:
    """
    Delete the on-disk FAISS index cache (.octen_faiss_index/) without
    immediately rebuilding it — the next get_vectorstore()/query() call
    will build a fresh one from whatever's currently in
    forensic_knowledge/*.md (+ skills/rag_corpus.json if present).

    Unlike rebuild_index(), this doesn't force the (blocking) embedding
    pass to happen right now — it just invalidates the cache so the
    first real rag_search call after startup is guaranteed to reflect
    the current corpus + current chunking logic, rather than whatever
    was on disk from a previous run (which is exactly the trap that
    bit chunk_index: the code changed, but a stale cached index kept
    getting silently reused instead of being rebuilt).

    Called automatically on every MCP server startup — see
    Image_delegation_mcp.py — gated by RAG_REBUILD_ON_STARTUP (default
    "true"). Set that to "false" once you're not actively iterating on
    forensic_knowledge/*.md or this file's chunking logic, since
    re-embedding the whole corpus on every restart is unnecessary
    (if slight) startup overhead once things have stabilized.

    Returns True if a cache was found and removed, False if there was
    nothing to clear (also resets the in-memory _vectorstore, in case
    this is called from the same process that has one loaded).
    """
    import shutil
    global _vectorstore
    _vectorstore = None
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
        return True
    return False


# ════════════════════════════════════════════════════════════════
#  Query — drop-in replacement for the old TF-IDF rag_search()
# ════════════════════════════════════════════════════════════════

def query(text: str, top_k: int = 5) -> dict:
    """
    Embedding-based similarity search over the Octen-indexed corpus.

    Returns the SAME shape as the old TF-IDF rag_search tool, so
    mcp_server.py's rag_search just delegates to this function:
        {"results": [{"text", "source", "score"}, ...], "corpus_size": N}

    FAISS similarity_search_with_score returns L2 distance by default
    (lower = more similar); this converts to a 0-1 "higher = better"
    score via 1 / (1 + distance) so the API contract matches the old
    cosine-similarity tool exactly.
    """
    store = get_vectorstore()
    raw = store.similarity_search_with_score(text, k=top_k)

    results = []
    for doc, distance in raw:
        if not doc.page_content:
            continue
        results.append({
            "text":        doc.page_content,
            "source":      doc.metadata.get("source", "unknown"),
            "chunk_index": doc.metadata.get("chunk_index", 0),
            "score":       round(1.0 / (1.0 + float(distance)), 4),
        })

    corpus_size = store.index.ntotal if hasattr(store, "index") else len(results)
    return {"results": results, "corpus_size": corpus_size}


# ════════════════════════════════════════════════════════════════
#  Keyword trigger — shared logic
#
#  Same case-insensitive, whole-phrase matching used by
#  Image_delegation_mcp.check_keyword_triggers (the Profiler agent's
#  MCP tool). Kept here as a plain function (not an @mcp.tool) so it
#  can be called directly from anywhere — including the orchestrator
#  in main.py, which drives the Forensic agent and has no MCP client
#  of its own by design (the Forensic agent is deliberately tool-less).
# ════════════════════════════════════════════════════════════════

# Trigger vocabulary, organized by the forensic_knowledge/*.md category it
# maps to. A real Forensic agent report is far more likely to describe a
# SPECIFIC finding ("entrance wound", "ligature mark", "stab wound") than
# to use the literal word "trauma" — so each category lists the concrete
# terms a report would actually use, not just its file name. Flattened
# into TRAUMA_TRIGGER_KEYWORDS below for the actual scan.
TRAUMA_KEYWORD_CATEGORIES: dict[str, list[str]] = {
    # General / catch-all phrasing
    "general": [
        "evidence of trauma",
        "trauma",
        "traumatic injury",
        "signs of trauma",
        "penetrating trauma",
    ],
    # -> Gunshot_wounds.md
    "gunshot": [
        "gunshot wound",
        "gunshot",
        "bullet wound",
        "bullet hole",
        "entrance wound",
        "exit wound",
        "firearm injury",
        "projectile wound",
        "powder stippling",
        "muzzle imprint",
    ],
    # -> sharp_force_injury.md
    "sharp_force": [
        "sharp force injury",
        "sharp force trauma",
        "stab wound",
        "incised wound",
        "incision wound",
        "puncture wound",
        "cut wound",
        "laceration",
        "chop wound",
    ],
    # -> asphyxiation.md
    "asphyxia": [
        "asphyxia",
        "asphyxiation",
        "asphyxial",
        "strangulation",
        "ligature mark",
        "ligature furrow",
        "ligature",
        "petechial hemorrhage",
        "petechiae",
        "hanging",
        "smothering",
        "suffocation",
        "manual strangulation",
    ],
    # -> Thermal_injuries.md
    "thermal": [
        "thermal injury",
        "burn injury",
        "burns",
        "charring",
        "scald",
        "fire-related death",
        "smoke inhalation",
    ],
    # -> forensic_stuff.md / forensic_stuff2.md (blunt force trauma)
    "blunt_force": [
        "blunt force trauma",
        "blunt force injury",
        "blunt trauma",
        "contusion",
        "abrasion",
        "hematoma",
        "fracture",
        "crush injury",
    ],
    # -> Postmortem_changes.md / Sudden_Natural_Death.md
    "postmortem_and_natural": [
        "postmortem change",
        "livor mortis",
        "rigor mortis",
        "algor mortis",
        "decomposition",
        "sudden natural death",
    ],
}

TRAUMA_TRIGGER_KEYWORDS: list[str] = [
    kw for kws in TRAUMA_KEYWORD_CATEGORIES.values() for kw in kws
]


EVIDENCE_OF_TRAUMA_HEADING_RE = re.compile(
    r"^#{1,6}\s*evidence of trauma\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def extract_evidence_of_trauma_section(report: str) -> Optional[str]:
    """
    Pull out the body of the Forensic agent's mandatory
    '## Evidence of Trauma' section (see forensic_agent.py's system
    prompt), i.e. everything after that heading up to the next
    heading or end of text.

    Returns None if no such heading is found at all, so the caller
    can fall back to keyword scanning (e.g. the model ignored the
    formatting instruction).
    """
    match = EVIDENCE_OF_TRAUMA_HEADING_RE.search(report)
    if not match:
        return None

    start = match.end()
    rest = report[start:]
    next_heading = re.search(r"^#{1,6}\s+\S", rest, re.MULTILINE)
    end = start + next_heading.start() if next_heading else len(report)
    return report[start:end].strip()


def section_says_yes(section_text: str) -> bool:
    """
    True if the Evidence of Trauma section opens with an affirmative
    ("Yes"), false if it opens with a negative ("No"/"None"/"Negative").
    Defaults to False (safer to under-trigger than mis-trigger) if the
    section doesn't clearly start with either.
    """
    first_word = section_text.strip().split(None, 1)[0].strip(".:,").lower() if section_text.strip() else ""
    return first_word == "yes"


def check_keyword_trigger(text: str, keywords: list[str] | None = None) -> dict:
    """
    Scan `text`, sentence by sentence, for any of `keywords` (case-
    insensitive; a trailing "s" is tolerated so "stab wound" also
    matches "stab wounds").

    Negation-aware: a match immediately preceded by a negation cue in
    the same sentence ("no evidence of trauma", "denies ligature
    marks", "without burns") is NOT counted as triggering — a report
    explicitly ruling something out shouldn't pull in reference
    material for it.

    Returns:
        {
          triggered:         bool,
          matched_keywords:  [kw, ...]         (de-duplicated, in first-seen order)
          matched_sentences: [sentence, ...]    (de-duplicated, in order)
          checked_keywords:  the full keyword list that was checked
        }
    """
    kws = keywords or TRAUMA_TRIGGER_KEYWORDS
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    matched_keywords: list[str] = []
    matched_sentences: list[str] = []

    for sentence in sentences:
        sentence_lower = sentence.lower()
        for kw in kws:
            pattern = rf"\b{re.escape(kw.lower())}s?\b"
            for m in re.finditer(pattern, sentence_lower):
                if _is_negated_before(sentence_lower, m.start()):
                    continue
                if kw not in matched_keywords:
                    matched_keywords.append(kw)
                if sentence not in matched_sentences:
                    matched_sentences.append(sentence)
                break  # one hit per keyword per sentence is enough

    return {
        "triggered": bool(matched_keywords),
        "matched_keywords": matched_keywords,
        "matched_sentences": matched_sentences,
        "checked_keywords": kws,
    }


_NEGATION_CUES_RE = re.compile(
    r"\b(no|not|without|absence of|negative for|denies|denied|"
    r"ruled out|free of|excludes?|unremarkable for)\b",
    re.IGNORECASE,
)


def _is_negated_before(sentence_lower: str, match_start: int, window_chars: int = 40) -> bool:
    """True if a negation cue appears in the `window_chars` immediately
    preceding the match within the same sentence (e.g. "no ... trauma",
    "denies ... ligature marks")."""
    preceding = sentence_lower[max(0, match_start - window_chars):match_start]
    return bool(_NEGATION_CUES_RE.search(preceding))


def extract_trigger_context(text: str, matched_keywords: list[str], window: int = 1) -> str:
    """
    Pull out the sentence(s) containing any matched (non-negated)
    keyword, plus `window` sentences of surrounding context on each
    side, to use as a focused retrieval query instead of the whole
    report. Falls back to the full text if sentence splitting finds no
    matches (shouldn't happen if matched_keywords came from
    check_keyword_trigger on the same text).
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    hit_indices = set()
    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        for kw in matched_keywords:
            pattern = rf"\b{re.escape(kw.lower())}s?\b"
            for m in re.finditer(pattern, sentence_lower):
                if not _is_negated_before(sentence_lower, m.start()):
                    hit_indices.update(range(max(0, i - window), min(len(sentences), i + window + 1)))
                    break

    if not hit_indices:
        return text

    return " ".join(sentences[i] for i in sorted(hit_indices))


def check_trauma_and_retrieve(text: str, top_k: int = 5) -> dict:
    """
    One-shot helper for the orchestrator: decide whether the Forensic
    agent's report shows evidence of trauma for THIS case, and if so,
    immediately run the Octen/FAISS similarity search against
    forensic_knowledge/ to fetch grounding reference material.

    Primary mechanism: the Forensic agent's system prompt requires it
    to end every report with a "## Evidence of Trauma" section whose
    first word is literally "Yes" or "No". We just read that — no
    keyword list needed, because the model already states its own
    finding for the specific case, in its own words, and those words
    (whatever they are — "gunshot wound", "ligature mark", a term we
    never anticipated) become the retrieval query directly. Semantic
    (embedding) search doesn't require us to guess the model's exact
    phrasing up front; it only needs a query string, and the "Yes"
    section IS that query.

    Fallback: if that section is missing (the model ignored the
    instruction, or this is running against an older report), fall
    back to scanning the whole report for TRAUMA_TRIGGER_KEYWORDS —
    a safety net, not the primary path.

    Returns:
        {
          "triggered": bool,
          "source": "evidence_of_trauma_section" | "keyword_fallback",
          "matched_keywords": [...],      # [] when source is the section
          "query": str | None,            # what was searched, if triggered
          "results": [{"text","source","score"}, ...],  # [] if not triggered
          "corpus_size": int,
        }
    """
    section = extract_evidence_of_trauma_section(text)

    if section is not None:
        triggered = section_says_yes(section)
        if not triggered:
            return {
                "triggered": False, "source": "evidence_of_trauma_section",
                "matched_keywords": [], "checked_keywords": [],
                "query": None, "results": [], "corpus_size": 0,
            }
        retrieval = query(section, top_k=top_k)
        return {
            "triggered": True, "source": "evidence_of_trauma_section",
            "matched_keywords": [], "checked_keywords": [],
            "query": section, **retrieval,
        }

    # Fallback: no "## Evidence of Trauma" section found in the report.
    trigger = check_keyword_trigger(text, TRAUMA_TRIGGER_KEYWORDS)
    if not trigger["triggered"]:
        return {**trigger, "source": "keyword_fallback", "query": None, "results": [], "corpus_size": 0}

    search_query = extract_trigger_context(text, trigger["matched_keywords"])
    retrieval = query(search_query, top_k=top_k)
    return {**trigger, "source": "keyword_fallback", "query": search_query, **retrieval}


if __name__ == "__main__":
    print(f"Building index from {FORENSIC_KNOWLEDGE_DIR} using {OCTEN_MODEL_ID} ...")
    rebuild_index()
    print("Done. Try: python -c \"from octen_rag import query; print(query('test'))\"")
