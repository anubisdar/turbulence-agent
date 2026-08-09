"""Tests for the aircraft reputation retrieval tool.

Builds a small real index in memory - real schema, real sqlite-vec, fake
encoder - so the filter, tier, cap and coverage logic are exercised against
actual SQL rather than mocks.
"""

import hashlib
import sqlite3
import struct

import pytest

from app.retrieval.embedding import normalize
from app.retrieval.schema import init_db
from app.retrieval.search import (
    Confidence,
    parse_type_query,
    search_aircraft_reputation,
)

DIM = 8


class FakeEncoder:
    """Deterministic vectors keyed by text."""

    name = "fake-model"
    dim = DIM

    def encode(self, texts):
        out = []
        for t in texts:
            d = hashlib.sha256(t.encode()).digest()
            out.append(normalize([d[i % len(d)] / 255.0 for i in range(DIM)]))
        return out


ENC = FakeEncoder()


def _vec(text):
    return struct.pack(f"{DIM}f", *ENC.encode([text])[0])


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception:  # pragma: no cover
        pytest.skip("sqlite-vec not installed")
    init_db(conn, embedding_dim=DIM)

    def add_case(mkey, ntsb, year, variant, family, gen, conf,
                 chunks, report="Final", raw_model=None):
        conn.execute(
            "INSERT INTO cases(mkey, ntsb_num, event_date, event_year,"
            " report_type, ingested_at) VALUES (?,?,?,?,?,?)",
            (mkey, ntsb, f"{year}-01-01", year, report, "2026-08-08"))
        conn.execute(
            "INSERT INTO case_aircraft(mkey, far_part, raw_make, raw_model,"
            " manufacturer, family, variant, generation, type_confidence,"
            " operator_name) VALUES (?,'121','BOEING',?,?,?,?,?,?,?)",
            (mkey, raw_model or variant or family, "Boeing", family, variant,
             gen, conf, "TEST AIR"))
        for i, text in enumerate(chunks):
            cur = conn.execute(
                "INSERT INTO chunks(mkey, section, section_priority, ordinal,"
                " ordinal_of, text, context_header, char_count, embedded_at,"
                " embedding_model) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (mkey, "probable_cause", 0, i, len(chunks), text, "hdr",
                 len(text), "2026-08-08", "fake-model"))
            conn.execute("INSERT INTO chunk_vec(chunk_id, embedding) VALUES (?,?)",
                         (cur.lastrowid, _vec(text)))

    # Two MAX 8 cases
    add_case(1, "DCA19MA101", 2019, "737 MAX 8", "737", "MAX", "exact",
             ["Uncommanded nose-down trim from the flight control system."])
    add_case(2, "DCA18RA100", 2018, "737 MAX 8", "737", "MAX", "exact",
             ["Repeated activation of the stabiliser trim during climb."])
    # Three 737-800 cases - must never surface on a MAX 8 query
    for n in range(3, 6):
        add_case(n, f"NG{n}", 2010 + n, "737-800", "737", "NG", "derived",
                 [f"A hard landing at the destination airport, case {n}."],
                 raw_model="737-8H4")
    # One bare-737 case: family tier
    add_case(6, "FAM6", 2015, None, "737", None, "family_only",
             ["Engine surge during the takeoff roll on a 737."])
    # A verbose case that would dominate without the per-case cap
    add_case(7, "VERBOSE", 2020, "737 MAX 8", "737", "MAX", "exact",
             [f"Detailed finding number {i} about trim." for i in range(10)])
    # A case with no narrative at all
    conn.execute(
        "INSERT INTO cases(mkey, ntsb_num, event_date, event_year,"
        " report_type, ingested_at) VALUES (8,'NOTEXT','2023-01-01',2023,"
        "'Preliminary','2026-08-08')")
    conn.execute(
        "INSERT INTO case_aircraft(mkey, far_part, family, variant,"
        " generation, type_confidence) VALUES (8,'121','737','737 MAX 8',"
        "'MAX','exact')")
    conn.commit()
    yield conn
    conn.close()


def search(db, type_str, **kw):
    return search_aircraft_reputation(db, ENC, type_str, **kw)


class TestTypeFilterIsExact:
    def test_max_8_query_never_returns_737_800_cases(self, db):
        out = search(db, "737 MAX 8", k=20)
        assert out.hits
        for hit in out.hits:
            assert hit.variant != "737-800"
            assert hit.generation != "NG"

    def test_737_800_query_never_returns_max_cases(self, db):
        out = search(db, "737-800", k=20, include_family_tier=False)
        assert out.hits
        for hit in out.hits:
            assert hit.generation != "MAX"

    def test_customer_code_query_finds_the_same_fleet(self, db):
        """A user typing the raw NTSB string gets the canonical bucket."""
        a = {h.mkey for h in search(db, "737-8H4", k=20,
                                    include_family_tier=False).hits}
        b = {h.mkey for h in search(db, "737-800", k=20,
                                    include_family_tier=False).hits}
        assert a == b and a


class TestTiers:
    def test_family_only_cases_are_labelled_not_merged(self, db):
        out = search(db, "737 MAX 8", k=20)
        tiers = {h.tier for h in out.hits}
        assert "variant" in tiers
        if "family" in tiers:
            fam = [h for h in out.hits if h.tier == "family"]
            assert all(h.variant is None for h in fam)

    def test_family_tier_can_be_disabled(self, db):
        out = search(db, "737 MAX 8", k=20, include_family_tier=False)
        assert all(h.tier == "variant" for h in out.hits)

    def test_family_results_are_flagged_in_notes(self, db):
        out = search(db, "737 MAX 8", k=20)
        if any(h.tier == "family" for h in out.hits):
            assert any("no variant recorded" in n for n in out.notes)


class TestPerCaseCap:
    def test_one_case_cannot_fill_the_result_set(self, db):
        out = search(db, "737 MAX 8", k=8, max_per_case=2)
        counts = {}
        for h in out.hits:
            counts[h.mkey] = counts.get(h.mkey, 0) + 1
        assert max(counts.values()) <= 2

    def test_cap_of_one_gives_distinct_cases(self, db):
        out = search(db, "737 MAX 8", k=8, max_per_case=1)
        mkeys = [h.mkey for h in out.hits]
        assert len(mkeys) == len(set(mkeys))


class TestCoverageReporting:
    def test_cases_without_narrative_are_counted(self, db):
        out = search(db, "737 MAX 8")
        assert out.coverage.cases_variant_without_text >= 1
        assert any("not an absent event" in n for n in out.notes)

    def test_tiers_are_counted_separately_never_summed(self, db):
        """Reporting variant + family as one total re-merges what the tier
        split exists to keep apart."""
        out = search(db, "737 MAX 8")
        cov = out.coverage
        assert cov.cases_variant == 4     # three MAX 8 cases + the no-text one
        assert cov.cases_family == 1      # the bare-737 case
        exact_note = next(n for n in out.notes if "match" in n)
        # the note must lead with the variant count, not the combined total
        leading = int(exact_note.split()[0])
        assert leading == cov.cases_variant
        assert leading != cov.cases_variant + cov.cases_family

    def test_family_group_is_reported_as_a_separate_line(self, db):
        out = search(db, "737 MAX 8")
        assert any("filed against the 737 family with no variant recorded" in n
                   for n in out.notes)

    def test_unknown_type_reports_absence_not_safety(self, db):
        out = search(db, "787-9")
        assert out.hits == []
        assert any("not evidence of a clean safety record" in n
                   for n in out.notes)

    def test_totals_reflect_the_corpus_not_the_result_set(self, db):
        out = search(db, "737 MAX 8", k=1)
        assert len(out.hits) == 1
        assert out.coverage.cases_variant > 1


class TestUnresolvableQueries:
    @pytest.mark.parametrize("q", ["banana", "", "a flying saucer"])
    def test_no_search_is_run_and_nothing_is_guessed(self, db, q):
        out = search(db, q)
        assert not out.ok
        assert out.hits == []
        assert any("could not be resolved" in n for n in out.notes)


class TestDeterminism:
    def test_identical_queries_give_identical_ordering(self, db):
        a = search(db, "737 MAX 8", k=5)
        b = search(db, "737 MAX 8", k=5)
        assert [h.chunk_id for h in a.hits] == [h.chunk_id for h in b.hits]


class TestQueryParsing:
    @pytest.mark.parametrize("q", ["737 MAX 8", "Boeing 737 MAX 8", "737-8"])
    def test_max_8_spellings_converge(self, q):
        assert parse_type_query(q).variant == "737 MAX 8"

    def test_marketing_neo_spelling(self):
        assert parse_type_query("A320neo").variant == "A320neo"

    def test_bare_family_is_family_only(self):
        t = parse_type_query("Boeing 737")
        assert t.family == "737"
        assert t.confidence is Confidence.FAMILY_ONLY


class TestFamilyTierIsNotABackDoor:
    """The family tier means 'variant not recorded', never 'sibling variant'.

    Regression guard: an earlier version admitted any case in the family whose
    variant differed from the query, which let 737-800 cases reach a MAX 8
    result set through the second tier - the exact collapse the first tier
    exists to prevent.
    """

    def test_sibling_variants_never_enter_via_the_family_tier(self, db):
        out = search(db, "737 MAX 8", k=50)
        family_hits = [h for h in out.hits if h.tier == "family"]
        assert all(h.variant is None for h in family_hits)

    def test_no_ng_case_appears_anywhere_in_a_max_result(self, db):
        out = search(db, "737 MAX 8", k=50)
        assert all(h.generation != "NG" for h in out.hits)
        assert all(h.raw_model != "737-8H4" for h in out.hits)

    def test_the_bare_family_case_does_appear(self, db):
        """Tier two must still work - it exists so 19% of cases stay visible."""
        out = search(db, "737 MAX 8", k=50)
        assert any(h.ntsb_num == "FAM6" for h in out.hits)
