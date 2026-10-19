"""
octen_rag.py — retrieval and trauma-type selection over forensic_knowledge/*.md.

* Embeddings: an Octen model (via langchain-huggingface / sentence-transformers).
* Index: LangChain FAISS, cached on disk in .octen_faiss_index/ (main.py clears the
  cache each time the servers are started unless RAG_REBUILD_ON_STARTUP=false).
* Trigger: the Forensic report's mandatory "Evidence of Trauma" section (Yes/No);
  a keyword scan of the whole report is only a fallback when that section is missing.
* One reference file is chosen per distinct trauma type (select_trauma_sources).

Octen models: Octen/Octen-Embedding-0.6B (1024-dim), -4B (2560-dim), -8B (4096-dim).
FAISS.save_local / load_local use pickle: only load indexes this app wrote itself.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

FORENSIC_KNOWLEDGE_DIR = Path(__file__).parent / "forensic_knowledge"
INDEX_DIR = Path(__file__).parent / ".octen_faiss_index"
OCTEN_MODEL_ID = os.getenv("OCTEN_MODEL_ID", "Octen/Octen-Embedding-0.6B")


# ════════════════════════════════════════════════════════════════
#  Embeddings
# ════════════════════════════════════════════════════════════════

def _patch_legacy_normalize_module() -> None:
    """
    Octen's saved SentenceTransformer config passes a legacy `normalize_embeddings`
    kwarg to the Normalize module, whose __init__ takes no parameters in current
    sentence-transformers (TypeError at load). This happens inside SentenceTransformer's
    own module loading, so make Normalize.__init__ accept and discard legacy kwargs.
    """
    try:
        from sentence_transformers.base.modules.normalize import Normalize
    except ImportError:
        return
    if getattr(Normalize, "_octen_legacy_patch_applied", False):
        return
    _original_init = Normalize.__init__

    def _patched_init(self, *args, **kwargs):
        _original_init(self)

    Normalize.__init__ = _patched_init
    Normalize._octen_legacy_patch_applied = True


def build_embeddings(model_id: str = OCTEN_MODEL_ID, device: str = "cpu"):
    """LangChain Embeddings backed by an Octen model (pooling + normalisation are built in,
    so no encode_kwargs: passing normalize_embeddings breaks on current sentence-transformers)."""
    _patch_legacy_normalize_module()
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={"device": device, "trust_remote_code": True},  # Octen ships custom modeling code
    )


# ════════════════════════════════════════════════════════════════
#  Corpus + vector store
# ════════════════════════════════════════════════════════════════

def load_corpus_documents(chunk_size: int = 800, chunk_overlap: int = 100) -> list[Document]:
    """
    Split every forensic_knowledge/*.md into chunks tagged with `source` (file name) and
    `chunk_index` (position within the file). Sentence-ending separators bias the splitter
    toward sentence boundaries.
    """
    if not FORENSIC_KNOWLEDGE_DIR.exists():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    docs: list[Document] = []
    for md_path in sorted(FORENSIC_KNOWLEDGE_DIR.glob("*.md")):
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        for i, chunk in enumerate(splitter.split_text(text)):
            docs.append(Document(
                page_content=chunk,
                metadata={"source": md_path.name, "chunk_index": i},
            ))
    return docs


_vectorstore = None  # module-level cache; reset by clear_index_cache()


def get_vectorstore():
    """The FAISS store: in-memory if already loaded, else loaded from the on-disk cache,
    else built from the corpus (and cached)."""
    global _vectorstore
    from langchain_community.vectorstores import FAISS

    if _vectorstore is not None:
        return _vectorstore

    embeddings = build_embeddings()

    if INDEX_DIR.exists():
        _vectorstore = FAISS.load_local(
            str(INDEX_DIR), embeddings,
            allow_dangerous_deserialization=True,   # safe: we wrote this index ourselves
        )
        return _vectorstore

    docs = load_corpus_documents()
    if not docs:
        # No corpus: an empty-but-valid store, so callers get no results instead of crashing.
        docs = [Document(page_content="", metadata={"source": "empty"})]

    _vectorstore = FAISS.from_documents(docs, embeddings)
    INDEX_DIR.mkdir(exist_ok=True)
    _vectorstore.save_local(str(INDEX_DIR))
    return _vectorstore


def clear_index_cache() -> bool:
    """
    Delete the on-disk index cache and the in-memory store, so the next query() rebuilds
    from the current corpus and chunking logic (a stale cache was once silently reused
    after the chunking code changed). Returns True if a cache was removed.
    """
    global _vectorstore
    _vectorstore = None
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
        return True
    return False


def query(text: str, top_k: int = 5) -> dict:
    """
    Similarity search. Returns {"results": [{text, source, chunk_index, score}, ...],
    "corpus_size": N}. FAISS returns L2 distance (lower = closer); it is converted to a
    0-1 "higher = better" score via 1 / (1 + distance).
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


def select_primary_source(results: list[dict]) -> Optional[str]:
    """The ONE reference file whose retrieved chunks have the highest summed score
    (several good chunks beat a single slightly better one). None if no usable chunks."""
    totals: dict[str, float] = {}
    for r in results:
        src = r.get("source")
        if src and r.get("text", "").strip():
            totals[src] = totals.get(src, 0.0) + float(r.get("score", 0.0))
    return max(totals, key=totals.get) if totals else None


def load_full_source_text(
    source: str, knowledge_dir: Path = FORENSIC_KNOWLEDGE_DIR,
) -> Optional[str]:
    """
    The COMPLETE text of one forensic_knowledge/*.md file by name. Only a bare file name
    inside forensic_knowledge/ is honoured (directory parts are stripped), so a crafted
    `source` can't read elsewhere. None if missing or not a .md file.
    """
    path = knowledge_dir / Path(source).name
    if path.suffix.lower() != ".md" or not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="ignore")


# ════════════════════════════════════════════════════════════════
#  Trauma trigger vocabulary
# ════════════════════════════════════════════════════════════════

# Concrete terms a Forensic report would use, grouped by the reference file they map to
# (see TRAUMA_CATEGORY_FILES). Flattened into TRAUMA_TRIGGER_KEYWORDS for the fallback scan.
TRAUMA_KEYWORD_CATEGORIES: dict[str, list[str]] = {
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
    # -> Asphyxiation.md
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
    # -> Blunt_Force_Trauma.md
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
    # -> Postmortem_changes.md / Sudden_Natural_Death.md (never selected as a trauma file)
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

# Matches "## Evidence of Trauma", "## Evidence of Trauma (EOT)" and a bare "## EOT".
EVIDENCE_OF_TRAUMA_HEADING_RE = re.compile(
    r"^#{1,6}\s*(?:evidence of trauma(?:\s*\(\s*eot\s*\))?|eot)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def extract_evidence_of_trauma_section(report: str) -> Optional[str]:
    """Body of the Forensic report's 'Evidence of Trauma' section (up to the next heading
    or end of text), or None if there is no such heading."""
    match = EVIDENCE_OF_TRAUMA_HEADING_RE.search(report)
    if not match:
        return None

    start = match.end()
    rest = report[start:]
    next_heading = re.search(r"^#{1,6}\s+\S", rest, re.MULTILINE)
    end = start + next_heading.start() if next_heading else len(report)
    return report[start:end].strip()


def section_says_yes(section_text: str) -> bool:
    """True if the section opens with "Yes". Anything else (No/None/unclear) is False:
    better to under-trigger than mis-trigger."""
    first_word = section_text.strip().split(None, 1)[0].strip(".:,").lower() if section_text.strip() else ""
    return first_word == "yes"


_NEGATION_CUES_RE = re.compile(
    r"\b(no|not|without|absence of|negative for|denies|denied|"
    r"ruled out|free of|excludes?|unremarkable for)\b",
    re.IGNORECASE,
)


def _is_negated_before(sentence_lower: str, match_start: int, window_chars: int = 40) -> bool:
    """True if a negation cue sits in the `window_chars` before the match, in the same sentence."""
    preceding = sentence_lower[max(0, match_start - window_chars):match_start]
    return bool(_NEGATION_CUES_RE.search(preceding))


def check_keyword_trigger(text: str, keywords: list[str]) -> dict:
    """
    Scan `text` sentence by sentence for `keywords` (case-insensitive; a trailing "s" is
    tolerated). A match preceded by a negation cue in the same sentence ("no burns",
    "without ligature marks") does not count.

    Returns {"triggered": bool, "matched_keywords": [...]} (de-duplicated, first-seen order).
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    matched_keywords: list[str] = []

    for sentence in sentences:
        sentence_lower = sentence.lower()
        for kw in keywords:
            pattern = rf"\b{re.escape(kw.lower())}s?\b"
            for m in re.finditer(pattern, sentence_lower):
                if _is_negated_before(sentence_lower, m.start()):
                    continue
                if kw not in matched_keywords:
                    matched_keywords.append(kw)
                break  # one hit per keyword per sentence is enough

    return {"triggered": bool(matched_keywords), "matched_keywords": matched_keywords}


def extract_trigger_context(text: str, matched_keywords: list[str], window: int = 1) -> str:
    """The sentences containing a matched (non-negated) keyword plus `window` sentences either
    side, used as a focused retrieval query. Falls back to the full text if none match."""
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


# Trauma categories that count as a distinct "trauma", and the reference file for each.
# "general" and "postmortem_and_natural" are deliberately absent: they never add a file.
TRAUMA_CATEGORY_FILES: dict[str, str] = {
    "gunshot":     "Gunshot_wounds.md",
    "sharp_force": "sharp_force_injury.md",
    "asphyxia":    "asphyxiation.md",
    "thermal":     "Thermal_injuries.md",
    "blunt_force": "Blunt_Force_Trauma.md",
}

# The Forensic agent labels every injury "Type: <Gunshot | Sharp force | Blunt force |
# Asphyxia | Thermal | Other>"; those labels are the most reliable way to count distinct
# trauma types, so they take priority over keyword scanning.
_TYPE_LABEL_TO_CATEGORY: list[tuple[str, str]] = [
    ("sharp force", "sharp_force"), ("blunt force", "blunt_force"),
    ("gunshot", "gunshot"), ("firearm", "gunshot"),
    ("asphyx", "asphyxia"), ("thermal", "thermal"), ("burn", "thermal"),
    ("sharp", "sharp_force"), ("blunt", "blunt_force"),
]
_TYPE_LABEL_RE = re.compile(
    r"^[\s>*\-\d.)]*\**\s*type\s*\**\s*[:\-\u2013\u2014]\s*\**\s*([A-Za-z][A-Za-z \-]*)",
    re.IGNORECASE | re.MULTILINE,
)


def _labelled_categories(text: str) -> list[str]:
    """Trauma categories named by explicit 'Type: ...' labels, in order of first appearance."""
    cats: list[str] = []
    for m in _TYPE_LABEL_RE.finditer(text):
        label = m.group(1).strip().lower()
        for key, category in _TYPE_LABEL_TO_CATEGORY:
            if label.startswith(key):
                if category not in cats:
                    cats.append(category)
                break
    return cats


def select_trauma_sources(text: str, knowledge_dir: Path = FORENSIC_KNOWLEDGE_DIR) -> list[str]:
    """
    One reference file per DISTINCT trauma type the report describes, in order of first
    appearance. Explicit "Type: ..." labels decide when present; otherwise each category's
    vocabulary is scanned for (ignoring negated mentions). A category whose file is missing
    from forensic_knowledge/ is skipped. [] if nothing is found.
    """
    labelled = [
        TRAUMA_CATEGORY_FILES[c] for c in _labelled_categories(text)
        if c in TRAUMA_CATEGORY_FILES and (knowledge_dir / TRAUMA_CATEGORY_FILES[c]).is_file()
    ]
    if labelled:
        return labelled

    lowered = text.lower()
    found: list[tuple[int, str]] = []
    for category, filename in TRAUMA_CATEGORY_FILES.items():
        if not (knowledge_dir / filename).is_file():
            continue
        hit = check_keyword_trigger(text, TRAUMA_KEYWORD_CATEGORIES[category])
        if not hit["triggered"]:
            continue
        pos = min(
            (lowered.find(kw.lower()) for kw in hit["matched_keywords"] if kw.lower() in lowered),
            default=len(lowered),
        )
        found.append((pos, filename))
    return [filename for _, filename in sorted(found)]


def check_trauma_and_retrieve(text: str, top_k: int = 5) -> dict:
    """
    Decide whether the Forensic report shows evidence of trauma and, if so, retrieve
    grounding chunks from forensic_knowledge/.

    Primary trigger: the report's "## Evidence of Trauma" section (its first word is
    "Yes"/"No"); the section text is the retrieval query. Fallback, only when that section
    is missing: scan the whole report for TRAUMA_TRIGGER_KEYWORDS.

    Returns {triggered, source ("evidence_of_trauma_section" | "keyword_fallback"), query,
    results, corpus_size, primary_source, trauma_sources}; the last two only when triggered.
    """
    section = extract_evidence_of_trauma_section(text)

    if section is not None:
        if not section_says_yes(section):
            return {"triggered": False, "source": "evidence_of_trauma_section",
                    "query": None, "results": [], "corpus_size": 0}
        retrieval = query(section, top_k=top_k)
        return {
            "triggered": True, "source": "evidence_of_trauma_section",
            "query": section, **retrieval,
            "primary_source": select_primary_source(retrieval["results"]),
            "trauma_sources": select_trauma_sources(section),
        }

    trigger = check_keyword_trigger(text, TRAUMA_TRIGGER_KEYWORDS)
    if not trigger["triggered"]:
        return {"triggered": False, "source": "keyword_fallback",
                "query": None, "results": [], "corpus_size": 0}

    search_query = extract_trigger_context(text, trigger["matched_keywords"])
    retrieval = query(search_query, top_k=top_k)
    return {
        "triggered": True, "source": "keyword_fallback",
        "query": search_query, **retrieval,
        "primary_source": select_primary_source(retrieval["results"]),
        "trauma_sources": select_trauma_sources(text),
    }
