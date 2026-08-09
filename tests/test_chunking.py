"""Tests for narrative chunking and the retrieval schema."""

import sqlite3

import pytest

from app.retrieval.chunking import (
    has_narrative,
    MAX_CHARS,
    Chunk,
    Section,
    build_context_header,
    chunk_case,
    clean_text,
    split_section,
)
from app.retrieval.schema import DDL, init_db, schema_version

# A real fragment from the CAROL export, entity damage included.
REAL_NARRATIVE = (
    "The flight instructor and student pilot were conducting an instructional "
    "flight. While on the downwind leg, their preset fuel reminder to switch "
    "fuel tanks alerted on the GPS. &#x0D;\nPostaccident examination of the "
    "airplane found that the fuel selector handle was positioned toward the "
    "OFF position, but it was not completely seated in the detent."
)


class FakeType:
    def __init__(self, family=None, variant=None):
        self.family = family
        self.variant = variant


def make_case(**over):
    case = {
        "cm_mkey": 193746,
        "cm_ntsbNum": "CEN24LA108",
        "cm_eventDate": "2024-01-31T18:22:00Z",
        "cm_mostRecentReportType": "Final",
        "cm_probableCause": "The flight crew's failure to maintain adequate "
                            "airspeed during the approach, which resulted in "
                            "an aerodynamic stall at low altitude.",
        "analysisNarrative": "The first paragraph of the analysis describes "
                             "the approach and the crew's configuration of "
                             "the airplane.\n\nThe second paragraph describes "
                             "the recorded flight data and the sequence that "
                             "followed the initial deviation.",
        "factualNarrative": "",
        "prelimNarrative": None,
        "cm_vehicles": [{"operatorName": "SOUTHWEST AIRLINES CO"}],
    }
    case.update(over)
    return case


class TestCleaning:
    def test_html_entities_are_removed(self):
        out = clean_text(REAL_NARRATIVE)
        assert "&#x0D;" not in out
        assert "&" not in out

    def test_empty_input_is_empty_not_none(self):
        assert clean_text(None) == ""
        assert clean_text("") == ""

    def test_collapses_runs_of_blank_lines(self):
        assert clean_text("a\n\n\n\n\nb") == "a\n\nb"


class TestProbableCauseIsAtomic:
    def test_never_split_regardless_of_length(self):
        long_cause = "The flight crew's failure to " + ("maintain airspeed " * 200)
        pieces = split_section(long_cause, Section.PROBABLE_CAUSE)
        assert len(pieces) == 1

    def test_typical_cause_survives_intact(self):
        cause = ("The student pilot's improper movement of the fuel selector "
                 "to the OFF position, which resulted in fuel starvation and "
                 "a total loss of engine power.")
        assert split_section(cause, Section.PROBABLE_CAUSE) == [cause]


class TestNarrativeSplitting:
    def test_narrative_below_target_is_one_chunk(self):
        text = ("A single paragraph of analysis that comfortably exceeds the "
                "minimum chunk length while staying well below the target "
                "size, so it should emerge as exactly one chunk.")
        assert len(split_section(text, Section.ANALYSIS)) == 1

    def test_long_narrative_splits(self):
        text = "\n\n".join(f"Paragraph {i}. " + "word " * 100 for i in range(20))
        pieces = split_section(text, Section.FACTUAL)
        assert len(pieces) > 1

    def test_no_chunk_wildly_exceeds_the_cap(self):
        text = "\n\n".join("word " * 400 for _ in range(10))
        for piece in split_section(text, Section.FACTUAL):
            assert len(piece) <= MAX_CHARS * 2

    def test_a_single_giant_paragraph_still_splits(self):
        text = " ".join(f"Sentence number {i} goes here." for i in range(400))
        pieces = split_section(text, Section.FACTUAL)
        assert len(pieces) > 1

    def test_no_empty_chunks(self):
        text = "Para one.\n\n\n\n\nPara two.\n\n   \n\nPara three."
        assert all(p.strip() for p in split_section(text, Section.ANALYSIS))


class TestContextHeader:
    def test_uses_canonical_type_not_raw_string(self):
        header = build_context_header(make_case(), [FakeType("737", "737-800")])
        assert "737-800" in header

    def test_variant_absent_falls_back_to_family(self):
        header = build_context_header(make_case(), [FakeType("737", None)])
        assert "737" in header

    def test_missing_type_is_stated_not_omitted(self):
        header = build_context_header(make_case(), [FakeType(None, None)])
        assert "not recorded" in header

    def test_preliminary_reports_are_flagged_in_the_header(self):
        case = make_case(cm_mostRecentReportType="Preliminary")
        header = build_context_header(case, [FakeType("737", "737-800")])
        assert "Preliminary" in header

    def test_final_reports_are_not_flagged(self):
        header = build_context_header(make_case(), [FakeType("737", "737-800")])
        assert "report" not in header.lower()


class TestChunkCase:
    def test_produces_chunks_for_each_populated_section(self):
        chunks = chunk_case(make_case(), [FakeType("737", "737-800")])
        sections = {c.section for c in chunks}
        assert Section.PROBABLE_CAUSE in sections
        assert Section.ANALYSIS in sections
        assert Section.FACTUAL not in sections   # empty in the fixture

    def test_embed_text_includes_header_but_stored_text_does_not(self):
        chunk = chunk_case(make_case(), [FakeType("737", "737-800")])[0]
        assert chunk.context_header in chunk.embed_text
        assert chunk.context_header not in chunk.text

    def test_ordinals_are_contiguous_within_a_section(self):
        text = "\n\n".join(f"Para {i}. " + "word " * 120 for i in range(12))
        chunks = chunk_case(make_case(factualNarrative=text), [FakeType("737")])
        factual = [c for c in chunks if c.section is Section.FACTUAL]
        assert [c.ordinal for c in factual] == list(range(len(factual)))

    def test_case_with_no_narrative_yields_no_chunks(self):
        empty = make_case(cm_probableCause=None, analysisNarrative=None,
                          factualNarrative=None, prelimNarrative=None)
        assert chunk_case(empty, [FakeType("737")]) == []


class TestSchema:
    @pytest.fixture
    def conn(self):
        c = sqlite3.connect(":memory:")
        c.row_factory = sqlite3.Row
        init_db(c)
        yield c
        c.close()

    def test_init_is_idempotent(self, conn):
        init_db(conn)
        assert schema_version(conn) == 1

    def test_expected_tables_exist(self, conn):
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"cases", "case_aircraft", "case_findings", "chunks"} <= names

    def test_variant_and_family_are_separately_indexed(self, conn):
        idx = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'")}
        assert "idx_aircraft_variant" in idx
        assert "idx_aircraft_family" in idx

    def test_chunks_are_unique_per_section_ordinal(self, conn):
        conn.execute("INSERT INTO cases(mkey, ntsb_num, ingested_at) "
                     "VALUES (1, 'X', '2026-08-08')")
        args = (1, "analysis", 1, 0, 1, "t", "h", 1)
        conn.execute("INSERT INTO chunks(mkey, section, section_priority, ordinal,"
                     " ordinal_of, text, context_header, char_count)"
                     " VALUES (?,?,?,?,?,?,?,?)", args)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO chunks(mkey, section, section_priority,"
                         " ordinal, ordinal_of, text, context_header, char_count)"
                         " VALUES (?,?,?,?,?,?,?,?)", args)

    def test_deleting_a_case_removes_its_chunks(self, conn):
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("INSERT INTO cases(mkey, ntsb_num, ingested_at) "
                     "VALUES (2, 'Y', '2026-08-08')")
        conn.execute("INSERT INTO chunks(mkey, section, section_priority, ordinal,"
                     " ordinal_of, text, context_header, char_count)"
                     " VALUES (2,'analysis',1,0,1,'t','h',1)")
        conn.execute("DELETE FROM cases WHERE mkey = 2")
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0

    def test_view_joins_case_aircraft_and_chunk(self, conn):
        conn.execute("INSERT INTO cases(mkey, ntsb_num, ingested_at) "
                     "VALUES (3, 'Z', '2026-08-08')")
        conn.execute("INSERT INTO case_aircraft(mkey, family, variant,"
                     " type_confidence) VALUES (3, '737', '737 MAX 8', 'exact')")
        conn.execute("INSERT INTO chunks(mkey, section, section_priority, ordinal,"
                     " ordinal_of, text, context_header, char_count)"
                     " VALUES (3,'probable_cause',0,0,1,'t','h',1)")
        row = conn.execute("SELECT * FROM v_chunk_context").fetchone()
        assert row["variant"] == "737 MAX 8"
        assert row["ntsb_num"] == "Z"


class TestEmbedPolicy:
    """Factual narratives are stored, not embedded."""

    def test_conclusions_are_embedded(self):
        chunks = chunk_case(make_case(), [FakeType("737", "737-800")])
        for c in chunks:
            if c.section in (Section.PROBABLE_CAUSE, Section.ANALYSIS):
                assert c.embedded

    def test_factual_is_stored_but_not_embedded(self):
        text = "\n\n".join("word " * 100 for _ in range(6))
        chunks = chunk_case(make_case(factualNarrative=text), [FakeType("737")])
        factual = [c for c in chunks if c.section is Section.FACTUAL]
        assert factual, "factual chunks should still be produced"
        assert not any(c.embedded for c in factual)

    def test_preliminary_is_embedded_since_it_is_the_only_text_for_open_cases(self):
        case = make_case(cm_probableCause=None, analysisNarrative=None,
                         factualNarrative=None,
                         prelimNarrative="A preliminary account of the event "
                                         "that is long enough to clear the "
                                         "minimum chunk length floor without "
                                         "being trimmed away as noise.")
        chunks = chunk_case(case, [FakeType("737")])
        assert chunks and all(c.embedded for c in chunks)


class TestMinimumLength:
    def test_runt_chunks_are_dropped(self):
        assert split_section("Too short.", Section.ANALYSIS) == []

    def test_short_probable_cause_is_dropped_not_kept_as_noise(self):
        assert split_section("Unknown.", Section.PROBABLE_CAUSE) == []

    def test_every_emitted_chunk_clears_the_floor(self):
        from app.retrieval.chunking import MIN_CHUNK_CHARS
        text = "\n\n".join("word " * 90 for _ in range(8)) + "\n\nx."
        for piece in split_section(text, Section.FACTUAL):
            assert len(piece) >= MIN_CHUNK_CHARS


class TestNarrativeCoverage:
    def test_case_with_narrative_is_detected(self):
        assert has_narrative(make_case())

    def test_case_without_narrative_is_detected(self):
        empty = make_case(cm_probableCause=None, analysisNarrative=None,
                          factualNarrative=None, prelimNarrative=None)
        assert not has_narrative(empty)

    def test_trivial_text_does_not_count_as_narrative(self):
        assert not has_narrative(make_case(
            cm_probableCause="Unknown.", analysisNarrative=None,
            factualNarrative=None, prelimNarrative=None))
