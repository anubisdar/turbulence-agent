"""Tests for the search-service payload assembly, not the search itself.

Found via a real false positive: the page's top-line severity badge and the
explanation paragraph beneath it disagreed, because the badge was the worst
reading among every corridor that survived the beam search while the
explanation described only the winning corridor's own evidence. These lock
in the fix - the headline is the winner's own reading, and the beam-wide
worst becomes a separate, named note instead of a silent substitute.
"""

from types import SimpleNamespace

from app.reasoning.controller import SearchResult
from app.reasoning.critic import Corridor, Evidence, Geometry, Provenance, Severity
from app.web.service import (_partial_route_note, _route_substitution_note,
                             _winner_reading, _worst_survivor_note)


def corridor(cid, reading, label=""):
    return Corridor(
        id=cid,
        provenance=Provenance.FILED_ROUTE,
        geometry=Geometry(length_nm=500.0, great_circle_nm=480.0,
                          max_dogleg_deg=5.0,
                          endpoints_match_airports=True,
                          altitude_profile_valid=True),
        evidence=Evidence(reading=reading),
        label=label,
    )


class TestWinnerReading:
    def test_the_winners_own_reading_is_used(self):
        winner = corridor("a", Severity.MODERATE)
        other = corridor("b", Severity.SEVERE)
        result = SearchResult(survivors=[winner, other],
                              reading=Severity.SEVERE)
        assert _winner_reading(result) is Severity.MODERATE

    def test_no_winner_is_unresolved(self):
        result = SearchResult(survivors=[], reading=Severity.UNRESOLVED)
        assert _winner_reading(result) is Severity.UNRESOLVED

    def test_a_single_survivor_is_its_own_reading(self):
        winner = corridor("a", Severity.LIGHT)
        result = SearchResult(survivors=[winner], reading=Severity.LIGHT)
        assert _winner_reading(result) is Severity.LIGHT


class TestWorstSurvivorNote:
    """The note that used to be missing entirely - see the module
    docstring for the bug this closes."""

    def test_a_worse_unselected_corridor_produces_a_note(self):
        winner = corridor("a", Severity.MODERATE)
        other = corridor("b", Severity.SEVERE)
        result = SearchResult(survivors=[winner, other],
                              reading=Severity.SEVERE)
        note = _worst_survivor_note(result, _winner_reading(result))
        assert note is not None
        assert "severe" in note
        assert "moderate" in note
        assert "disagree" in note.lower()

    def test_agreement_produces_no_note(self):
        """Every survivor at the same reading as the winner - nothing to
        say, so nothing is said."""
        winner = corridor("a", Severity.MODERATE)
        other = corridor("b", Severity.MODERATE)
        result = SearchResult(survivors=[winner, other],
                              reading=Severity.MODERATE)
        assert _worst_survivor_note(result, _winner_reading(result)) is None

    def test_a_single_survivor_produces_no_note(self):
        """Nothing to disagree with when there is only one corridor."""
        winner = corridor("a", Severity.SEVERE)
        result = SearchResult(survivors=[winner], reading=Severity.SEVERE)
        assert _worst_survivor_note(result, _winner_reading(result)) is None

    def test_no_winner_produces_no_note(self):
        result = SearchResult(survivors=[], reading=Severity.UNRESOLVED)
        assert _worst_survivor_note(result, _winner_reading(result)) is None

    def test_an_unresolved_worst_produces_no_note(self):
        """result.reading only reaches UNRESOLVED when every survivor is
        unresolved (see worst() in evidence.py) - including the winner, so
        there would be nothing to contrast it with anyway."""
        winner = corridor("a", Severity.UNRESOLVED)
        result = SearchResult(survivors=[winner], reading=Severity.UNRESOLVED)
        assert _worst_survivor_note(result, _winner_reading(result)) is None

    def test_the_winner_itself_being_the_worst_produces_no_note(self):
        """The selected route already carries the worst reading - there is
        no different, hidden corridor to disclose."""
        winner = corridor("a", Severity.SEVERE)
        other = corridor("b", Severity.LIGHT)
        result = SearchResult(survivors=[winner, other],
                              reading=Severity.SEVERE)
        assert _worst_survivor_note(result, _winner_reading(result)) is None


class TestPartialRouteNote:
    """A depth-3 winner can be half of a longitudinally split corridor -
    this note is what stops that reading from being read as covering the
    whole trip. See `_partial_route_note`'s docstring in service.py."""

    def test_a_split_winner_produces_a_note(self):
        winner = corridor("track/high/first", Severity.SEVERE,
                          label="first half")
        result = SearchResult(survivors=[winner], reading=Severity.SEVERE)
        note = _partial_route_note(result)
        assert note is not None
        assert "first half" in note

    def test_the_other_half_is_named_correctly_too(self):
        winner = corridor("track/high/second", Severity.MODERATE,
                          label="second half")
        result = SearchResult(survivors=[winner], reading=Severity.MODERATE)
        note = _partial_route_note(result)
        assert note is not None
        assert "second half" in note

    def test_an_unsplit_winner_produces_no_note(self):
        """An ordinary corridor - one never routed through the
        longitudinal split - has no half-of-the-route caveat to add."""
        winner = corridor("track", Severity.SEVERE, label="")
        result = SearchResult(survivors=[winner], reading=Severity.SEVERE)
        assert _partial_route_note(result) is None

    def test_an_altitude_band_winner_produces_no_note(self):
        """A depth-2 altitude-band label is not a longitudinal split and
        must not be mistaken for one."""
        winner = corridor("track/high", Severity.SEVERE,
                          label="high band FL350")
        result = SearchResult(survivors=[winner], reading=Severity.SEVERE)
        assert _partial_route_note(result) is None

    def test_no_winner_produces_no_note(self):
        result = SearchResult(survivors=[], reading=Severity.UNRESOLVED)
        assert _partial_route_note(result) is None


class TestRouteSubstitutionNote:
    """When a pair has no nonstop service, the search can run on the
    long-haul leg of a real connecting itinerary instead - this note is
    what stops that reading from being read as covering the whole
    requested trip. See `_route_substitution_note`'s docstring in
    service.py, and `CorridorGenerator._get_flight` for where the
    substitution itself happens."""

    def test_a_substitution_produces_a_note(self):
        generator = SimpleNamespace(
            route_substitution=("KSAN", "RJTT", "KLAX", "RJTT"))
        note = _route_substitution_note(generator)
        assert note is not None
        assert "KSAN to RJTT" in note
        assert "KLAX to RJTT" in note

    def test_no_substitution_produces_no_note(self):
        generator = SimpleNamespace(route_substitution=None)
        assert _route_substitution_note(generator) is None
