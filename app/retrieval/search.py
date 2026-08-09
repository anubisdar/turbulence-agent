"""
Aircraft reputation retrieval over the NTSB Part 121 index.

The one tool this layer exposes to the agent. Four rules shape it, each
carried over from a decision made earlier in the project:

METADATA FILTER FIRST. Aircraft type is matched exactly against the canonical
    fields, never inferred from vector proximity. Embeddings treat `737-8` and
    `737-8H4` as near-identical strings; the aircraft are twenty years apart in
    certification. Similarity ranks within a type, it never selects the type.

TWO TIERS, LABELLED. 19% of Part 121 cases record a family but no variant. A
    strict variant filter would hide 78 bare-`737` cases. Family matches are
    returned as a second, labelled tier so "same family, variant unrecorded"
    is never presented as "same aircraft".

CAP PER CASE. One verbose investigation can supply many chunks. Without a cap
    a result set can be eight chunks from one accident, which reads to a user
    as eight independent sources agreeing.

ABSENCE IS REPORTED, NOT IMPLIED. 12.9% of cases have no narrative at all, and
    that rises to a third in the 2020s. Every result carries the corpus counts
    so a thin answer can never be mistaken for a clean record.

Ranking is exact, not approximate. The metadata filter reduces the candidate
set to a size where a full scan is cheap, so there is no ANN recall loss.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from typing import Sequence

from app.retrieval.aircraft_types import AircraftType, Confidence, resolve
from app.retrieval.embedding import embed_query

DEFAULT_K = 8
DEFAULT_MAX_PER_CASE = 2

#: Leading words that identify a manufacturer in a free-text query.
_MAKE_HINTS: list[tuple[str, str]] = [
    (r"^BOEING\b", "BOEING"),
    (r"^AIRBUS\b", "AIRBUS"),
    (r"^EMBRAER\b", "EMBRAER"),
    (r"^BOMBARDIER\b", "BOMBARDIER"),
    (r"^MCDONNELL\s*DOUGLAS\b", "MCDONNELL DOUGLAS"),
    (r"^DOUGLAS\b", "DOUGLAS"),
    (r"^LOCKHEED\b", "LOCKHEED"),
    (r"^FOKKER\b", "FOKKER"),
]

#: When the query names no manufacturer, infer one from the model shape.
_MODEL_HINTS: list[tuple[str, str]] = [
    (r"^7[0-9]7", "BOEING"),
    (r"^A2\d{2}", "AIRBUS CANADA"),
    (r"^A3\d{2}", "AIRBUS"),
    (r"^(CRJ|CL-?600|CL-?65)", "BOMBARDIER"),
    (r"^(ERJ|EMB|E1\d{2})", "EMBRAER"),
    (r"^(MD-?\d|DC-?\d)", "MCDONNELL DOUGLAS"),
    (r"^L-?1011", "LOCKHEED"),
    (r"^F-?\d{2}", "FOKKER"),
]


def parse_type_query(text: str) -> AircraftType:
    """Resolve a free-text aircraft type the way a user would write it.

    `"737 MAX 8"`, `"Boeing 737 MAX 8"` and `"737-8"` all land on the same
    canonical type, which is the same one the corpus's `737-8` resolves to.
    """
    s = re.sub(r"\s+", " ", (text or "").upper().strip())
    for pattern, make in _MAKE_HINTS:
        if re.search(pattern, s):
            return resolve(make, re.sub(pattern, "", s).strip())
    for pattern, make in _MODEL_HINTS:
        if re.match(pattern, s):
            return resolve(make, s)
    return resolve("", s)


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    mkey: int
    ntsb_num: str
    event_date: str | None
    event_year: int | None
    section: str
    text: str
    score: float
    tier: str                 # "variant" | "family"
    variant: str | None
    family: str | None
    generation: str | None
    type_confidence: str
    raw_model: str | None
    operator: str | None
    report_type: str | None
    source: str
    source_class: str

    @property
    def provisional(self) -> bool:
        """Preliminary reports carry findings that may still change."""
        return (self.report_type or "").lower().startswith("prelim")


@dataclass
class Coverage:
    """What the corpus holds, independent of what the query matched.

    The two tiers are counted separately and never summed. A query for
    `737 MAX 8` matches 11 cases exactly; a further 76 are filed as bare
    `737` with no variant recorded and may be Classics from the 1980s.
    Reporting 87 would re-merge what the tier split exists to keep apart.
    """
    cases_variant: int = 0
    cases_variant_with_text: int = 0
    cases_family: int = 0
    cases_family_with_text: int = 0
    newest_event_year: int | None = None
    oldest_event_year: int | None = None

    @property
    def cases_variant_without_text(self) -> int:
        return self.cases_variant - self.cases_variant_with_text

    @property
    def cases_family_without_text(self) -> int:
        return self.cases_family - self.cases_family_with_text


@dataclass
class Retrieval:
    query: str
    resolved_type: AircraftType | None
    hits: list[Hit] = field(default_factory=list)
    coverage: Coverage = field(default_factory=Coverage)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.resolved_type is not None and self.resolved_type.usable


# ------------------------------------------------------------------ internals

def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob)//4}f", blob))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _candidate_sql(tier: str) -> str:
    """SQL for one tier.

    The family tier means *variant not recorded*, never *a different variant
    of the same family*. Allowing sibling variants through here would
    reintroduce exactly the collapse the exact filter prevents: a MAX 8 query
    would pick up 737-800 cases by the back door.
    """
    column = "a.variant" if tier == "variant" else "a.family"
    return f"""
        SELECT DISTINCT ch.id AS chunk_id, ch.mkey, ch.section,
               ch.section_priority, ch.text,
               c.ntsb_num, c.event_date, c.event_year, c.report_type,
               c.source, c.source_class,
               a.variant, a.family, a.generation, a.type_confidence,
               a.raw_model, a.operator_name
        FROM chunks ch
        JOIN cases c ON c.mkey = ch.mkey
        JOIN case_aircraft a ON a.mkey = ch.mkey
        WHERE {column} = ?
          AND a.far_part = '121'
          AND ch.embedded_at IS NOT NULL
          {"AND a.variant IS NULL" if tier == "family" else ""}
    """


_TIER_COUNT_SQL = """
    SELECT COUNT(DISTINCT a.mkey) AS total,
           COUNT(DISTINCT CASE WHEN ch.id IS NOT NULL THEN a.mkey END) AS with_text,
           MIN(c.event_year) AS oldest, MAX(c.event_year) AS newest
    FROM case_aircraft a
    JOIN cases c ON c.mkey = a.mkey
    LEFT JOIN chunks ch ON ch.mkey = a.mkey AND ch.embedded_at IS NOT NULL
    WHERE a.far_part = '121' AND {predicate}
"""


def _collect_coverage(conn, atype: AircraftType) -> Coverage:
    cov = Coverage()

    if atype.variant:
        row = conn.execute(
            _TIER_COUNT_SQL.format(predicate="a.variant = ?"), (atype.variant,)
        ).fetchone()
        if row:
            cov.cases_variant = row["total"] or 0
            cov.cases_variant_with_text = row["with_text"] or 0
            cov.oldest_event_year = row["oldest"]
            cov.newest_event_year = row["newest"]

    if atype.family:
        row = conn.execute(
            _TIER_COUNT_SQL.format(
                predicate="a.family = ? AND a.variant IS NULL"), (atype.family,)
        ).fetchone()
        if row:
            cov.cases_family = row["total"] or 0
            cov.cases_family_with_text = row["with_text"] or 0
            if cov.oldest_event_year is None:
                cov.oldest_event_year = row["oldest"]
                cov.newest_event_year = row["newest"]
    return cov


def _rank(conn, rows, qvec, tier, k, max_per_case) -> list[Hit]:
    if not rows:
        return []
    ids = [r["chunk_id"] for r in rows]
    placeholders = ",".join("?" for _ in ids)
    vectors = {
        r["chunk_id"]: _unpack(r["embedding"])
        for r in conn.execute(
            f"SELECT chunk_id, embedding FROM chunk_vec "
            f"WHERE chunk_id IN ({placeholders})", ids
        )
    }

    scored = []
    for r in rows:
        vec = vectors.get(r["chunk_id"])
        if vec is None:
            continue
        scored.append((_dot(qvec, vec), r))

    # Higher score first; ties broken toward conclusions over raw record, then
    # by recency, so ordering is deterministic for identical inputs.
    scored.sort(
        key=lambda t: (-t[0], t[1]["section_priority"], -(t[1]["event_year"] or 0))
    )

    per_case: dict[int, int] = {}
    hits: list[Hit] = []
    for score, r in scored:
        if len(hits) >= k:
            break
        if per_case.get(r["mkey"], 0) >= max_per_case:
            continue
        per_case[r["mkey"]] = per_case.get(r["mkey"], 0) + 1
        hits.append(Hit(
            chunk_id=r["chunk_id"], mkey=r["mkey"], ntsb_num=r["ntsb_num"],
            event_date=r["event_date"], event_year=r["event_year"],
            section=r["section"], text=r["text"], score=round(score, 4),
            tier=tier, variant=r["variant"], family=r["family"],
            generation=r["generation"], type_confidence=r["type_confidence"],
            raw_model=r["raw_model"], operator=r["operator_name"],
            report_type=r["report_type"], source=r["source"],
            source_class=r["source_class"],
        ))
    return hits


# ------------------------------------------------------------------ the tool

def search_aircraft_reputation(
    conn,
    encoder,
    aircraft_type: str,
    query: str = "safety incidents and accidents",
    k: int = DEFAULT_K,
    max_per_case: int = DEFAULT_MAX_PER_CASE,
    include_family_tier: bool = True,
) -> Retrieval:
    """Retrieve NTSB material for an aircraft type.

    `aircraft_type` is filtered exactly. `query` ranks within that filter.
    """
    atype = parse_type_query(aircraft_type)
    out = Retrieval(query=query, resolved_type=atype)

    if not atype.usable:
        out.notes.append(
            f"Aircraft type '{aircraft_type}' could not be resolved to a known "
            f"type. No search was run - returning nothing rather than guessing "
            f"at a similar type."
        )
        return out

    out.coverage = _collect_coverage(conn, atype)
    qvec = embed_query(encoder, query)

    if atype.variant:
        rows = conn.execute(_candidate_sql("variant"), (atype.variant,)).fetchall()
        out.hits.extend(_rank(conn, rows, qvec, "variant", k, max_per_case))

    remaining = k - len(out.hits)
    if include_family_tier and atype.family and remaining > 0:
        rows = conn.execute(_candidate_sql("family"), (atype.family,)).fetchall()
        seen = {h.chunk_id for h in out.hits}
        rows = [r for r in rows if r["chunk_id"] not in seen]
        out.hits.extend(_rank(conn, rows, qvec, "family", remaining, max_per_case))

    _add_notes(out, atype)
    return out


def _add_notes(out: Retrieval, atype: AircraftType) -> None:
    cov = out.coverage

    if atype.confidence is Confidence.FAMILY_ONLY:
        out.notes.append(
            f"Query resolved only to the {atype.family} family. Results span "
            f"all variants of that family, which may differ substantially."
        )

    label = atype.variant or atype.family

    if cov.cases_variant == 0 and cov.cases_family == 0:
        out.notes.append(
            f"No Part 121 cases exist in this corpus for {label}. This is an "
            f"absence of records, not evidence of a clean safety record."
        )
        return

    if atype.variant:
        if cov.cases_variant == 0:
            out.notes.append(
                f"No Part 121 case is filed specifically against {label}. "
                f"This is an absence of records, not a clean safety record."
            )
        else:
            missing = cov.cases_variant_without_text
            line = (f"{cov.cases_variant} Part 121 case(s) match {label} exactly; "
                    f"{cov.cases_variant_with_text} have retrievable narrative text")
            if missing:
                pct = missing / cov.cases_variant * 100
                line += (f", {missing} ({pct:.0f}%) do not - typically an open "
                         f"investigation. Absent narrative is not an absent event")
            out.notes.append(line + ".")

    if cov.cases_family:
        out.notes.append(
            f"A further {cov.cases_family} case(s) are filed against the "
            f"{atype.family} family with no variant recorded "
            f"({cov.cases_family_with_text} with narrative text). These are "
            f"counted separately because they may involve a different variant."
        )

    if any(h.tier == "family" for h in out.hits):
        n = sum(1 for h in out.hits if h.tier == "family")
        out.notes.append(
            f"{n} result(s) shown come from that family-only group and may or "
            f"may not involve {label}."
        )

    if any(h.provisional for h in out.hits):
        out.notes.append(
            "One or more results come from preliminary reports. Preliminary "
            "findings are provisional and may change."
        )

    if (cov.cases_variant or cov.cases_family) and not out.hits:
        out.notes.append(
            f"{cov.cases_variant + cov.cases_family} case(s) exist across both "
            f"tiers but none had embedded narrative text to search."
        )
