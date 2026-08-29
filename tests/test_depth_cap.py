"""The interface must not offer a depth the generator cannot honour.

The controller's default was 3 and the form allowed up to 4, while the
generator implements two levels: corridor source at depth 1, cruise
altitude band at depth 2. A third pass produced no candidates and the
search stopped, so it cost no API calls and returned the same answer -
which is the problem. A caller who chose depth 3 reasonably assumed they
had searched deeper, and nothing said otherwise.

A control that appears to do something and does not is worse than one
that refuses, because there is nothing to notice.
"""

import pytest

from app.reasoning.controller import (DEFAULT_DEPTH_LIMIT,
                                      MAX_IMPLEMENTED_DEPTH)


class TestTheDefaultMatchesWhatExists:
    def test_the_default_is_within_what_is_implemented(self):
        assert DEFAULT_DEPTH_LIMIT <= MAX_IMPLEMENTED_DEPTH

    def test_two_levels_are_implemented(self):
        """Corridor source, then altitude band. Raising this constant
        without writing a third expansion re-creates the original bug."""
        assert MAX_IMPLEMENTED_DEPTH == 2


class TestTheFormOffersOnlyWhatWorks:
    def _depth_input(self):
        from pathlib import Path
        import re
        here = Path(__file__).resolve().parents[1]
        html = (here / "app" / "web" / "static" / "index.html").read_text()
        found = re.search(r'<input[^>]*id="depth"[^>]*>', html)
        assert found, "the depth control is missing from the page"
        return found.group(0)

    def test_the_maximum_matches_the_implementation(self):
        import re
        tag = self._depth_input()
        cap = re.search(r'max="(\d+)"', tag)
        assert cap, "the depth control has no maximum"
        assert int(cap.group(1)) == MAX_IMPLEMENTED_DEPTH

    def test_the_default_value_matches_the_controller(self):
        import re
        tag = self._depth_input()
        val = re.search(r'value="(\d+)"', tag)
        assert val and int(val.group(1)) == DEFAULT_DEPTH_LIMIT


class TestTheApiClampsRatherThanPretending:
    """The form caps the control, but the same field is reachable over
    HTTP. Clamping silently would reproduce the bug for API callers."""

    def test_a_request_beyond_the_limit_is_clamped_and_reported(self):
        from dataclasses import replace

        from app.web.service import SearchRequest
        req = SearchRequest(origin="KPIT", dest="KBOS", depth_limit=5)
        assert req.depth_limit > MAX_IMPLEMENTED_DEPTH

        clamped = replace(req, depth_limit=MAX_IMPLEMENTED_DEPTH)
        assert clamped.depth_limit == MAX_IMPLEMENTED_DEPTH

    def test_the_request_default_matches_the_controller(self):
        from app.web.service import SearchRequest
        assert SearchRequest().depth_limit == DEFAULT_DEPTH_LIMIT
