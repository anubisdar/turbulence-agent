"""
SQLite schema for the NTSB retrieval index.

Two design constraints drove this shape, both measured from the corpus:

1. 19% of Part 121 cases record a family but no variant (`737`, not
   `737-800`). Family and variant are therefore separate indexed columns.
   A variant query joins on variant for tier 1 and family for tier 2, and
   the tiers are labelled in the result so the caller never mistakes
   "same family, variant unrecorded" for "same aircraft".

2. Chunk counts per case vary by two orders of magnitude - a probable cause
   is one chunk, a long factual narrative is fifty. Retrieval must be able to
   cap per-case contribution, so `mkey` is indexed on chunks.

Provenance is stored on every row, per the project rule that provenance
travels with the data. `report_type` matters especially: a preliminary
report's narrative is provisional and must be labelled as such downstream.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL
);

-- One row per NTSB investigation.
CREATE TABLE IF NOT EXISTS cases (
    mkey                INTEGER PRIMARY KEY,
    ntsb_num            TEXT NOT NULL,
    event_date          TEXT,            -- ISO8601, UTC
    event_year          INTEGER,
    event_type          TEXT,            -- ACC / INC
    report_type         TEXT,            -- Final / Preliminary / Factual
    completion_status   TEXT,
    highest_injury      TEXT,
    fatal_count         INTEGER,
    city                TEXT,
    state               TEXT,
    country             TEXT,
    latitude            REAL,
    longitude           REAL,
    -- provenance
    source              TEXT NOT NULL DEFAULT 'NTSB CAROL',
    source_class        TEXT NOT NULL DEFAULT 'formal',
    source_url          TEXT,
    ingested_at         TEXT NOT NULL,
    export_window       TEXT             -- which cached pull this came from
);

CREATE INDEX IF NOT EXISTS idx_cases_year ON cases(event_year);
CREATE INDEX IF NOT EXISTS idx_cases_report_type ON cases(report_type);

-- One row per aircraft in a case. A case may have more than one.
CREATE TABLE IF NOT EXISTS case_aircraft (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    mkey                INTEGER NOT NULL REFERENCES cases(mkey) ON DELETE CASCADE,
    vehicle_num         INTEGER,
    far_part            TEXT,            -- 121, 135, 091 ...
    -- raw, kept so any normalizer decision can be audited later
    raw_make            TEXT,
    raw_model           TEXT,
    -- canonical, from app.retrieval.aircraft_types
    manufacturer        TEXT,
    family              TEXT,
    variant             TEXT,
    generation          TEXT,            -- MAX / NG / Classic / neo / ceo
    type_confidence     TEXT NOT NULL,   -- exact/derived/family_only/unresolved
    operator_name       TEXT,
    registration        TEXT,
    damage_level        TEXT
);

-- The filter path. Variant for tier 1, family for tier 2.
CREATE INDEX IF NOT EXISTS idx_aircraft_variant ON case_aircraft(variant);
CREATE INDEX IF NOT EXISTS idx_aircraft_family ON case_aircraft(family);
CREATE INDEX IF NOT EXISTS idx_aircraft_generation ON case_aircraft(generation);
CREATE INDEX IF NOT EXISTS idx_aircraft_mkey ON case_aircraft(mkey);
CREATE INDEX IF NOT EXISTS idx_aircraft_far_part ON case_aircraft(far_part);

-- Findings taxonomy. Structured, not embedded; used to explain a result.
CREATE TABLE IF NOT EXISTS case_findings (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    mkey                INTEGER NOT NULL REFERENCES cases(mkey) ON DELETE CASCADE,
    finding_code        TEXT,
    finding_text        TEXT,
    in_probable_cause   INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_findings_mkey ON case_findings(mkey);

-- Retrievable text.
CREATE TABLE IF NOT EXISTS chunks (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    mkey                INTEGER NOT NULL REFERENCES cases(mkey) ON DELETE CASCADE,
    section             TEXT NOT NULL,   -- probable_cause / analysis / factual
    section_priority    INTEGER NOT NULL,
    ordinal             INTEGER NOT NULL,
    ordinal_of          INTEGER NOT NULL,
    text                TEXT NOT NULL,   -- exactly as written by the investigator
    context_header      TEXT NOT NULL,   -- prepended only when embedding
    char_count          INTEGER NOT NULL,
    embedded_at         TEXT,
    embedding_model     TEXT,
    UNIQUE(mkey, section, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_chunks_mkey ON chunks(mkey);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section);

-- Convenience view: everything the retrieval tool needs in one join.
CREATE VIEW IF NOT EXISTS v_chunk_context AS
SELECT
    ch.id            AS chunk_id,
    ch.mkey,
    ch.section,
    ch.section_priority,
    ch.ordinal,
    ch.text,
    ch.char_count,
    c.ntsb_num,
    c.event_date,
    c.event_year,
    c.event_type,
    c.report_type,
    c.source,
    c.source_class,
    c.source_url,
    a.manufacturer,
    a.family,
    a.variant,
    a.generation,
    a.type_confidence,
    a.far_part,
    a.operator_name
FROM chunks ch
JOIN cases c        ON c.mkey = ch.mkey
LEFT JOIN case_aircraft a ON a.mkey = ch.mkey;
"""

# sqlite-vec virtual table. Separate because it needs the extension loaded,
# and schema init should succeed on a box without it.
VEC_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding FLOAT[{dim}]
);
"""


def connect(path: str | Path, load_vec: bool = False) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    if load_vec:
        import sqlite_vec  # imported lazily; optional dependency
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    return conn


def init_db(conn: sqlite3.Connection, embedding_dim: int | None = None) -> None:
    conn.executescript(DDL)
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('version', ?)",
        (str(SCHEMA_VERSION),),
    )
    if embedding_dim:
        conn.executescript(VEC_DDL.format(dim=embedding_dim))
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('embedding_dim', ?)",
            (str(embedding_dim),),
        )
    conn.commit()


def schema_version(conn: sqlite3.Connection) -> int | None:
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'version'"
    ).fetchone()
    return int(row["value"]) if row else None
