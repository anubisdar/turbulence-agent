"""Turning what a person types into what the provider wants.

Fliers know IATA codes. The provider speaks ICAO. The tempting shortcut -
prepend K to any three-letter code - is wrong in exactly the places that
matter, and wrong in the worst way: KANC does not exist, so the search
finds nothing and the result is indistinguishable from a quiet route.
"""

import pytest

from app.retrieval.airports import (ASSUMED, EXACT, KNOWN, Airport,
                               local_to_utc, resolve_airport, resolve_pair,
                               timezone_for)


class TestTheCommonCase:
    def test_a_three_letter_code_becomes_four(self):
        assert resolve_airport("LAX").code == "KLAX"

    def test_lower_case_is_accepted(self):
        assert resolve_airport("lax").code == "KLAX"

    def test_surrounding_space_is_ignored(self):
        assert resolve_airport("  pit  ").code == "KPIT"

    def test_a_four_letter_code_passes_through(self):
        found = resolve_airport("KPIT")
        assert found.code == "KPIT" and found.how == EXACT


class TestThePlacesTheKRuleBreaks:
    """The reason this is a table rather than a string concatenation."""

    @pytest.mark.parametrize("typed,expected", [
        ("ANC", "PANC"),      # Alaska
        ("FAI", "PAFA"),
        ("HNL", "PHNL"),      # Hawaii
        ("OGG", "PHOG"),
        ("SJU", "TJSJ"),      # Puerto Rico
        ("GUM", "PGUM"),      # Guam
        ("YYZ", "CYYZ"),      # Canada
        ("LHR", "EGLL"),      # United Kingdom
        ("NRT", "RJAA"),      # Japan
        ("SYD", "YSSY"),      # Australia
    ])
    def test_the_code_is_looked_up_not_computed(self, typed, expected):
        found = resolve_airport(typed)
        assert found.code == expected
        assert found.how == KNOWN, "should come from the table, not the rule"

    def test_prefixing_k_would_have_been_wrong_for_all_of_them(self):
        """Stated as its own assertion because it is the whole argument."""
        for typed in ("ANC", "HNL", "SJU", "LHR", "NRT"):
            assert resolve_airport(typed).code != f"K{typed}"


class TestTheGuessIsLabelled:
    def test_an_unknown_code_falls_back_to_the_k_rule(self):
        found = resolve_airport("XYZ")
        assert found.code == "KXYZ"

    def test_and_says_that_it_guessed(self):
        assert resolve_airport("XYZ").how == ASSUMED
        assert resolve_airport("XYZ").is_guess

    def test_the_note_tells_the_reader_it_was_a_guess(self):
        note = resolve_airport("XYZ").note()
        assert "guess" in note and "KXYZ" in note

    def test_a_looked_up_code_reports_what_it_resolved_to(self):
        note = resolve_airport("ANC").note()
        assert "ANC" in note and "PANC" in note
        assert "guess" not in note

    def test_a_code_typed_in_full_needs_no_note(self):
        assert resolve_airport("KPIT").note() is None


class TestInputThatIsNotACode:
    @pytest.mark.parametrize("typed", ["", "  ", "1", "AB", "ABCDE",
                                       "12345", "K1TT", "LA-X"])
    def test_returns_nothing_rather_than_guessing(self, typed):
        """None so the caller can say the input was not a code. Coercing
        it into something four letters long would search for an airport
        nobody asked about."""
        assert resolve_airport(typed) is None

    def test_none_input_is_handled(self):
        assert resolve_airport("") is None


class TestBothEndsTogether:
    def test_a_pair_resolves_independently(self):
        o, d = resolve_pair("LAX", "KBOS")
        assert o.code == "KLAX" and o.how == KNOWN
        assert d.code == "KBOS" and d.how == EXACT

    def test_mixed_formats_are_fine(self):
        """A person may know one code and copy the other."""
        o, d = resolve_pair("pit", "RJTT")
        assert (o.code, d.code) == ("KPIT", "RJTT")

    def test_the_same_airport_typed_two_ways_resolves_the_same(self):
        """BOS and KBOS are one airport, and the caller compares the
        resolved codes so it can say so rather than searching a route
        from an airport to itself."""
        o, d = resolve_pair("BOS", "KBOS")
        assert o.code == d.code


class TestTimezoneFor:
    def test_a_known_airport_has_a_timezone(self):
        assert timezone_for("KPIT") == "America/New_York"

    def test_arizona_has_no_dst_zone(self):
        """The whole reason this is a table and not the K rule's cousin:
        Phoenix and Tucson share a state with the rest of Mountain time
        but not its DST, so both need their own explicit zone rather than
        inheriting America/Denver."""
        assert timezone_for("KPHX") == "America/Phoenix"
        assert timezone_for("KTUS") == "America/Phoenix"

    def test_indiana_is_not_a_bare_new_york_zone(self):
        assert timezone_for("KIND") == "America/Indiana/Indianapolis"

    def test_an_unresolved_code_has_no_timezone(self):
        assert timezone_for("KXXX") is None

    def test_an_assumed_guess_has_no_timezone_either(self):
        """ASSUMED codes are guesses at the airport itself; this table
        never compounds that guess with a second one about its clock."""
        guess = resolve_airport("ZZZ")
        assert guess.how == ASSUMED
        assert timezone_for(guess.code) is None


class TestLocalToUtc:
    def test_pittsburgh_afternoon_converts_to_utc(self):
        """Eastern Daylight Time in September is UTC-4."""
        date, time_ = local_to_utc("2026-09-15", "16:15",
                                   "America/New_York")
        assert (date, time_) == ("2026-09-15", "20:15")

    def test_a_winter_departure_uses_standard_time_not_daylight(self):
        """Eastern Standard Time is UTC-5 - the same clock time converts
        differently six months apart, which is the entire reason this
        goes through zoneinfo rather than a fixed offset."""
        date, time_ = local_to_utc("2026-01-15", "16:15",
                                   "America/New_York")
        assert (date, time_) == ("2026-01-15", "21:15")

    def test_an_early_local_departure_can_roll_the_date_backward(self):
        """2 AM in Tokyo (UTC+9) is still the evening before in UTC - a
        conversion that only fixed the clock and left the date alone
        would silently misplace the departure by a day."""
        date, time_ = local_to_utc("2026-09-15", "02:00", "Asia/Tokyo")
        assert (date, time_) == ("2026-09-14", "17:00")

    def test_a_late_local_departure_can_roll_the_date_forward(self):
        """8:30 PM in Los Angeles (UTC-7 in September) is already past
        midnight in UTC."""
        date, time_ = local_to_utc("2026-09-15", "20:30",
                                   "America/Los_Angeles")
        assert (date, time_) == ("2026-09-16", "03:30")

    def test_arizona_gives_the_same_offset_year_round(self):
        summer = local_to_utc("2026-07-01", "10:00", "America/Phoenix")
        winter = local_to_utc("2026-01-01", "10:00", "America/Phoenix")
        assert summer[1] == winter[1] == "17:00"
