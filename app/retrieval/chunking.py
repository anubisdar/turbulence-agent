"""
Segment NTSB Part 121 case narratives into retrievable chunks.

Chunking is structural, not a fixed sliding window. NTSB cases are already
divided into sections that mean different things:

  probable_cause  the Board's formal conclusion. Short, dense, highest value
                  per character. Never split - splitting a probable cause
                  statement severs the causal clause from its subject.
  analysis        the Board's reasoning. Multi-paragraph, ~3.6k chars median.
  factual         raw investigative record: weather, duty times, maintenance
                  history. Runs to 65k chars. High volume, low signal density.

Each chunk carries a context header into the embedding. A paragraph deep in an
analysis narrative may never name the aircraft, so a semantic query about a
737 MAX would not match it on content alone. The header restores what the
prose assumes.

Chunk text is stored unmodified. The header is prepended only to the string
that gets embedded, so retrieved text is always what the investigator wrote.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from enum import Enum


class Section(str, Enum):
    PROBABLE_CAUSE = "probable_cause"
    ANALYSIS = "analysis"
    FACTUAL = "factual"
    PRELIMINARY = "preliminary"


#: Section -> source field in the CAROL export.
SECTION_FIELDS: dict[Section, str] = {
    Section.PROBABLE_CAUSE: "cm_probableCause",
    Section.ANALYSIS: "analysisNarrative",
    Section.FACTUAL: "factualNarrative",
    Section.PRELIMINARY: "prelimNarrative",
}

#: Retrieval preference when scores are close. Conclusions beat raw record.
SECTION_PRIORITY: dict[Section, int] = {
    Section.PROBABLE_CAUSE: 0,
    Section.ANALYSIS: 1,
    Section.PRELIMINARY: 2,
    Section.FACTUAL: 3,
}

TARGET_CHARS = 1200
MAX_CHARS = 1600
MIN_CHARS = 250

#: Below this, a chunk carries no retrievable meaning. A six-character
#: fragment matches queries by embedding accident, not by content.
MIN_CHUNK_CHARS = 80

#: Sections that go into the vector index.
#:
#: Factual narratives are 66% of the corpus by chunk count and contain the raw
#: investigative record - registration boilerplate, crew certificates, weather
#: observations, wreckage description. They are stored in full and retrievable
#: by case, but they are not embedded. Semantic search runs over the Board's
#: conclusions; the underlying record is pulled by reference once a case is
#: surfaced. This keeps the index at ~34% of full size and stops one verbose
#: investigation from supplying an entire result set.
EMBEDDED_SECTIONS: set = set()  # populated below, after Section is defined


EMBEDDED_SECTIONS = {
    Section.PROBABLE_CAUSE,
    Section.ANALYSIS,
    Section.PRELIMINARY,
}


@dataclass(frozen=True)
class Chunk:
    mkey: int
    ntsb_num: str
    section: Section
    ordinal: int
    text: str
    context_header: str
    meta: dict = field(default_factory=dict)

    @property
    def embedded(self) -> bool:
        """Whether this chunk goes into the vector index.

        Stored either way. False means retrievable by case, not by similarity.
        """
        return self.section in EMBEDDED_SECTIONS

    @property
    def embed_text(self) -> str:
        return f"{self.context_header}\n\n{self.text}"

    @property
    def char_count(self) -> int:
        return len(self.text)


# ------------------------------------------------------------------ cleaning

_ENTITY_RE = re.compile(r"&#x?[0-9A-Fa-f]+;|&[a-zA-Z]+;")

#: Investigator prose arrives with typographic punctuation that renders as
#: mojibake on non-UTF8 terminals and adds nothing to retrieval.
_TYPOGRAPHIC = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u2026": "...", "\u00a0": " ", "\u200b": "",
})
_WS_RE = re.compile(r"[ \t]+")
_BREAKS_RE = re.compile(r"(?:\r\n|\r|\n){2,}")


def clean_text(raw: str | None) -> str:
    """Undo the encoding damage in the CAROL export.

    Narratives arrive with HTML entities left in place - `&#x0D;` for a
    carriage return is common - and inconsistent line endings.
    """
    if not raw:
        return ""
    s = html.unescape(raw)
    s = s.translate(_TYPOGRAPHIC)
    s = _ENTITY_RE.sub(" ", s)          # anything unescape missed
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _WS_RE.sub(" ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


# ------------------------------------------------------------------ splitting

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_paragraphs(text: str) -> list[str]:
    parts = _BREAKS_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _pack(units: list[str], target: int, hard_max: int) -> list[str]:
    """Greedily pack units into chunks near `target`, never exceeding hard_max
    unless a single unit is itself too long."""
    chunks: list[str] = []
    buf = ""
    for unit in units:
        if not buf:
            buf = unit
        elif len(buf) + 2 + len(unit) <= target:
            buf = f"{buf}\n\n{unit}"
        else:
            chunks.append(buf)
            buf = unit
    if buf:
        chunks.append(buf)

    # A single paragraph longer than hard_max gets split on sentences.
    out: list[str] = []
    for c in chunks:
        if len(c) <= hard_max:
            out.append(c)
            continue
        sentences = _split_sentences(c)
        if len(sentences) <= 1:
            out.extend(c[i:i + hard_max] for i in range(0, len(c), hard_max))
        else:
            out.extend(_pack(sentences, target, hard_max * 10))

    # Fold a runt tail into its predecessor rather than emitting a stub.
    if len(out) > 1 and len(out[-1]) < MIN_CHARS:
        tail = out.pop()
        if len(out[-1]) + len(tail) <= hard_max * 2:
            out[-1] = f"{out[-1]}\n\n{tail}"
        else:
            out.append(tail)
    return out


def split_section(text: str, section: Section) -> list[str]:
    """Probable cause is atomic. Everything else packs by paragraph."""
    text = clean_text(text)
    if not text:
        return []
    if section is Section.PROBABLE_CAUSE:
        return [text] if len(text) >= MIN_CHUNK_CHARS else []
    pieces = _pack(_split_paragraphs(text), TARGET_CHARS, MAX_CHARS)
    return [p for p in pieces if len(p) >= MIN_CHUNK_CHARS]


# ------------------------------------------------------------------ headers

def build_context_header(case: dict, aircraft: list) -> str:
    """One line of orienting context prepended to the embedded string.

    Uses the canonical type from the normalizer, not the raw NTSB string, so
    `737-8H4` and `737-832` produce the same header and embed alike.
    """
    year = (case.get("cm_eventDate") or "")[:4]
    types = []
    for a in aircraft:
        label = a.variant or a.family
        if label and label not in types:
            types.append(label)
    operators = []
    for v in (case.get("cm_vehicles") or []):
        op = (v.get("operatorName") or "").strip()
        if op and op not in operators:
            operators.append(op)

    bits = [", ".join(types) or "aircraft type not recorded"]
    if operators:
        bits.append(operators[0].title())
    if year:
        bits.append(year)
    bits.append(f"NTSB {case.get('cm_ntsbNum', '')}".strip())

    report = case.get("cm_mostRecentReportType")
    if report and report.lower() != "final":
        bits.append(f"{report} report")

    return " | ".join(b for b in bits if b)


# ------------------------------------------------------------------ entry point

def has_narrative(case: dict) -> bool:
    """True if any narrative field holds usable text.

    12.3% of Part 121 cases have none. Those cases are real accidents that are
    invisible to semantic retrieval, and the retrieval tool must report them
    rather than let a thin result read as a clean record.
    """
    return any(
        len(clean_text(case.get(f))) >= MIN_CHUNK_CHARS
        for f in SECTION_FIELDS.values()
    )


def chunk_case(case: dict, aircraft: list) -> list[Chunk]:
    """Turn one CAROL case into its chunks.

    `aircraft` is the list of resolved AircraftType for the case's vehicles.
    Passed in rather than resolved here so chunking stays a pure function of
    its inputs and the normalizer can be tested independently.
    """
    header = build_context_header(case, aircraft)
    mkey = case.get("cm_mkey")
    ntsb_num = case.get("cm_ntsbNum") or ""
    out: list[Chunk] = []

    for section, source_field in SECTION_FIELDS.items():
        pieces = split_section(case.get(source_field), section)
        for i, piece in enumerate(pieces):
            out.append(Chunk(
                mkey=mkey,
                ntsb_num=ntsb_num,
                section=section,
                ordinal=i,
                text=piece,
                context_header=header,
                meta={"of": len(pieces)},
            ))
    return out
