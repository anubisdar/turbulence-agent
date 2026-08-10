"""
Canonical aircraft type resolution for NTSB Part 121 records.

NTSB make/model fields are free text entered by investigators over 60 years.
The same airframe appears as `737-7H4`, `737 7H4`, and `737-700`. Different
airframes appear as `737-8` (a MAX 8) and `737-8H4` (a 737-800, certified two
decades earlier). A retrieval filter that matches raw strings will both miss
most of a fleet and silently mix generations.

This module maps (make, model) to a canonical AircraftType.

Design rule carried over from Checkpoint 2.1: a string this module cannot
resolve is marked UNRESOLVED. It is never guessed into a nearby bucket, and
never silently dropped into the family it superficially resembles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Confidence(str, Enum):
    EXACT = "exact"              # canonical variant recovered with certainty
    DERIVED = "derived"          # variant decoded from a customer/engine code
    FAMILY_ONLY = "family_only"  # manufacturer + family, no variant available
    UNRESOLVED = "unresolved"    # do not use for type-filtered retrieval


@dataclass(frozen=True)
class AircraftType:
    manufacturer: str | None
    family: str | None
    variant: str | None
    generation: str | None
    confidence: Confidence
    raw_make: str
    raw_model: str

    @property
    def key(self) -> str:
        """Stable identifier for the metadata filter."""
        if self.confidence is Confidence.UNRESOLVED:
            return "UNRESOLVED"
        return self.variant or self.family or "UNRESOLVED"

    @property
    def usable(self) -> bool:
        return self.confidence is not Confidence.UNRESOLVED


# --------------------------------------------------------------- manufacturer

_MANUFACTURER_PATTERNS: list[tuple[str, str]] = [
    (r"^BOEING\s*-?\s*CANADA|^DE\s*HAVILLAND|^DEHAVILLAND", "De Havilland Canada"),
    (r"^BOEING", "Boeing"),
    (r"^AIRBUS\s*CANADA", "A220"),
    (r"^AIRBUS", "Airbus"),
    (r"^MCDONNELL\s*DOUGLAS|^DOUGLAS", "McDonnell Douglas"),
    (r"^EMBRAER", "Embraer"),
    (r"^MITSUBISHI.*BOMBARDIER", "Bombardier"),
    (r"^BOMBARDIER|^CANADAIR", "Bombardier"),
    (r"^BEECH|^BEECHCRAFT|^RAYTHEON|^TEXTRON", "Beechcraft"),
    (r"^SHORT", "Shorts"),
    (r"^SWEARINGEN", "Swearingen"),
    (r"^MITSUBISHI", "Bombardier"),
    (r"^SAAB", "Saab"),
    (r"^FOKKER", "Fokker"),
    (r"^AEROSPATIALE|^ATR", "ATR"),
    (r"^BRITISH\s*AEROSPACE|^BAE|^JETSTREAM", "BAe"),
    (r"^FAIRCHILD|^DORNIER", "Dornier"),
    (r"^LOCKHEED", "Lockheed"),
    (r"^CONVAIR", "Convair"),
    (r"^CESSNA", "Cessna"),
    (r"^PIPER", "Piper"),
    (r"^GRUMMAN", "Grumman"),
    (r"^BELL", "Bell"),
    (r"^SIKORSKY", "Sikorsky"),
]


def normalize_manufacturer(make: str) -> str | None:
    s = re.sub(r"\s+", " ", (make or "").upper().strip())
    for pattern, canonical in _MANUFACTURER_PATTERNS:
        if re.search(pattern, s):
            return canonical
    return None


# --------------------------------------------------------------- model tidying

def _tidy(model: str) -> str:
    """Uppercase, collapse whitespace, unify separators, drop stray hyphens.

    `A-319-114` -> `A319-114`
    `737 7H4`   -> `737-7H4`
    `CL 600 2B19` -> `CL-600-2B19`
    """
    s = (model or "").upper().strip()
    s = re.sub(r"[\u2010-\u2015]", "-", s)   # unicode dashes
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    s = re.sub(r"-?SERIES$", "", s)
    s = re.sub(r"^A-(\d)", r"A\1", s)        # A-319-114 -> A319-114
    s = re.sub(r"^([A-Z]+)-(\d{3})$", r"\1\2", s)  # MD-11 stays, handled later
    return s


# --------------------------------------------------------------- Boeing

# Series digit -> generation, for the customer-code families
_B737_GENERATION = {
    "1": "Original", "2": "Original",
    "3": "Classic", "4": "Classic", "5": "Classic",
    "6": "NG", "7": "NG", "8": "NG", "9": "NG",
}

# Boeing MAX marketing names. Bare single-digit suffixes on the 737 only.
_B737_MAX = {
    "7": "737 MAX 7", "8": "737 MAX 8", "9": "737 MAX 9",
    "10": "737 MAX 10", "8200": "737 MAX 200",
}
_B777X = {"8": "777-8", "9": "777-9"}
# 787 suffixes ARE series numbers, not a new generation. Kept explicit so the
# 737 rule can never leak onto them.
_B787_SERIES = {"8": "787-8", "9": "787-9", "10": "787-10"}

_SUFFIX_TAGS = r"(?:ER|LR|F|SF|BCF|BDSF|M|W|X)?"

# Series that actually exist. A decoded customer code that implies a series
# outside this set means the source string was mis-entered - `A320-321` decodes
# to a nonexistent "A320-300". We fall back to family only rather than invent
# an aircraft. Same rule as everywhere else in this project: no defaults.
_VALID_SERIES: dict[str, set[str]] = {
    "707": {"1", "3"},
    "717": {"2"},
    "727": {"1", "2"},
    "737": {"1", "2", "3", "4", "5", "6", "7", "8", "9"},
    "747": {"1", "2", "3", "4", "8"},
    "757": {"2", "3"},
    "767": {"2", "3", "4"},
    "777": {"2", "3"},
    "787": {"8", "9", "10"},
    "A318": {"1"},
    "A319": {"1"},
    "A320": {"1", "2"},
    "A321": {"1", "2"},
    "A330": {"2", "3", "8", "9"},
    "A340": {"2", "3", "5", "6"},
    "A350": {"9", "10"},
    "A380": {"8"},
}


def _series_ok(family: str, series: str) -> bool:
    valid = _VALID_SERIES.get(family)
    return valid is None or series in valid


_BOEING_SHORTHAND = {"B744": "747-400", "B742": "747-200", "B743": "747-300",
                     "B762": "767-200", "B763": "767-300", "B764": "767-400",
                     "B772": "777-200", "B773": "777-300",
                     "B752": "757-200", "B753": "757-300"}


def _resolve_boeing(model: str) -> tuple[str | None, str | None, str | None, Confidence]:
    s = re.sub(r"^BOEING-?", "", model)

    # Some records prefix the model with B: `B-737-832`, `B767-332`, `B744`.
    if s in _BOEING_SHORTHAND:
        v = _BOEING_SHORTHAND[s]
        return v.split("-")[0], v, None, Confidence.DERIVED
    stripped = re.sub(r"^B-?(?=7[0-9]7)", "", s)
    if stripped != s:
        s = stripped

    # DC/MD types filed under Boeing after the 1997 merger
    if s.startswith("DC"):
        return _resolve_mcdonnell(s)
    if s.startswith("DHC") or s.startswith("DH-8"):
        return _resolve_dehavilland(s)

    if "MAX" in s:
        m = re.search(r"MAX-?(\d{1,2})", s)
        if m and m.group(1) in _B737_MAX:
            return "737", _B737_MAX[m.group(1)], "MAX", Confidence.EXACT
        return "737", None, "MAX", Confidence.FAMILY_ONLY

    # MD-11 etc. filed under Boeing after the merger
    if s.startswith("MD"):
        return _resolve_mcdonnell(s)

    m = re.match(rf"^(7[0-9]7)(?:-([0-9]{{1,4}}[A-Z0-9]*?){_SUFFIX_TAGS})?$", s)
    if not m:
        m = re.match(r"^(7[0-9]7)-?([0-9A-Z]+)?$", s)
        if not m:
            return None, None, None, Confidence.UNRESOLVED

    family, suffix = m.group(1), m.group(2)

    if suffix is None:
        return family, None, None, Confidence.FAMILY_ONLY

    # --- the collision that matters -------------------------------------
    # 737: a bare 1-2 char numeric suffix is a MAX marketing designation.
    #      a 3-char suffix is series digit + 2-char customer code.
    if family == "737":
        if suffix in _B737_MAX:
            return family, _B737_MAX[suffix], "MAX", Confidence.EXACT
        if re.fullmatch(r"[0-9][A-Z0-9]{2}", suffix):
            series = suffix[0]
            if not _series_ok(family, series):
                return family, None, None, Confidence.FAMILY_ONLY
            return (family, f"737-{series}00", _B737_GENERATION.get(series),
                    Confidence.EXACT if suffix.endswith("00") else Confidence.DERIVED)
        return family, None, None, Confidence.FAMILY_ONLY

    if family == "777":
        if suffix in _B777X:
            return family, _B777X[suffix], "777X", Confidence.EXACT
        if re.fullmatch(r"[0-9][A-Z0-9]{2}", suffix):
            if not _series_ok(family, suffix[0]):
                return family, None, None, Confidence.FAMILY_ONLY
            return (family, f"777-{suffix[0]}00", None,
                    Confidence.EXACT if suffix.endswith("00") else Confidence.DERIVED)
        return family, None, None, Confidence.FAMILY_ONLY

    if family == "787":
        if suffix in _B787_SERIES:
            return family, _B787_SERIES[suffix], None, Confidence.EXACT
        return family, None, None, Confidence.FAMILY_ONLY

    # 707/717/727/747/757/767: series digit + customer code
    if re.fullmatch(r"[0-9][A-Z0-9]{2}", suffix):
        if not _series_ok(family, suffix[0]):
            return family, None, None, Confidence.FAMILY_ONLY
        return (family, f"{family}-{suffix[0]}00", None,
                Confidence.EXACT if suffix.endswith("00") else Confidence.DERIVED)

    return family, None, None, Confidence.FAMILY_ONLY


# --------------------------------------------------------------- Airbus

def _resolve_airbus(model: str) -> tuple[str | None, str | None, str | None, Confidence]:
    s = re.sub(r"^AIRBUS-?", "", model)
    s = re.sub(r"^A-?", "A", s)
    if re.match(r"^\d{3}", s):          # bare `320`, `330-200`
        s = "A" + s

    # Marketing spellings a user would type: `A320NEO`, `A321-NEO`, `A320CEO`
    m = re.match(r"^A(3[1-2]\d)-?(NEO|CEO)$", s)
    if m:
        family = f"A{m.group(1)}"
        gen = m.group(2).lower()
        return family, (f"{family}neo" if gen == "neo" else None), gen, (
            Confidence.EXACT if gen == "neo" else Confidence.FAMILY_ONLY)

    # A300/A310 use a different scheme
    m = re.match(r"^A(30\d|310)(?:[B-F]\d)?-?(\d{3})?", s)
    if m and m.group(1) in ("300", "301", "310"):
        fam = f"A{m.group(1)}"
        return fam, fam, None, Confidence.FAMILY_ONLY

    # Series is one digit for most families but two for the A350-1000, so
    # the group is 1-2 digits and backtracking resolves the ambiguity:
    # "232" splits as 2/32, "1000" as 10/00.
    m = re.match(rf"^A(3[1-8]\d)(?:-(\d{{1,2}})(\d{{2}})(NX|N)?){_SUFFIX_TAGS}?$", s)
    if not m:
        m = re.match(r"^A(3[0-8]\d)$", s)
        if m:
            return f"A{m.group(1)}", None, None, Confidence.FAMILY_ONLY
        return None, None, None, Confidence.UNRESOLVED

    family = f"A{m.group(1)}"
    series, _engine, neo = m.group(2), m.group(3), m.group(4)

    if neo:
        return family, f"{family}neo", "neo", Confidence.EXACT
    if not _series_ok(family, series):
        return family, None, None, Confidence.FAMILY_ONLY
    generation = "ceo" if family in ("A319", "A320", "A321") else None
    return family, f"{family}-{series}00", generation, Confidence.DERIVED


# --------------------------------------------------------------- McDonnell Douglas

def _resolve_mcdonnell(model: str) -> tuple[str | None, str | None, str | None, Confidence]:
    s = model
    # DC-9-82 is the type-certificate name for the MD-82
    m = re.match(r"^DC-?9-?(8\d)$", s)
    if m:
        return "MD-80", f"MD-{m.group(1)}", None, Confidence.DERIVED
    m = re.match(r"^DC-?(\d{1,2})-?(\d{2})?", s)
    if m:
        fam = f"DC-{m.group(1)}"
        return fam, (f"{fam}-{m.group(2)}" if m.group(2) else None), None, (
            Confidence.EXACT if m.group(2) else Confidence.FAMILY_ONLY)
    m = re.match(rf"^MD-?(\d{{2}})(?:-(\d{{2}}))?-?{_SUFFIX_TAGS}$", s)
    if m:
        num = m.group(1)
        fam = "MD-80" if num.startswith("8") and num != "80" else f"MD-{num}"
        return fam, f"MD-{num}", None, Confidence.EXACT
    return None, None, None, Confidence.UNRESOLVED


# --------------------------------------------------------------- regionals

_BOMBARDIER_CL600 = {
    "2B19": ("CRJ-200", "CRJ"),
    "2C10": ("CRJ-700", "CRJ"),
    "2D15": ("CRJ-705", "CRJ"),
    "2D24": ("CRJ-900", "CRJ"),
    "2E25": ("CRJ-1000", "CRJ"),
    "1A11": ("Challenger 600", None),
    "2A12": ("Challenger 601", None),
    "2B16": ("Challenger 604", None),
}


def _resolve_bombardier(model: str) -> tuple[str | None, str | None, str | None, Confidence]:
    if model.startswith("BD-500") or model.startswith("BD500"):
        return _resolve_airbus_canada(model)
    m = re.match(r"^CL-?600-?([0-9A-Z]{4})$", model)
    if m and m.group(1) in _BOMBARDIER_CL600:
        variant, gen = _BOMBARDIER_CL600[m.group(1)]
        family = "CRJ" if gen == "CRJ" else "Challenger"
        return family, variant, gen, Confidence.DERIVED
    if re.fullmatch(r"CRJ-?1?", model):
        return "CRJ", None, "CRJ", Confidence.FAMILY_ONLY
    if re.fullmatch(r"CL-?600", model):
        return "Challenger", None, None, Confidence.FAMILY_ONLY
    if re.fullmatch(r"CL-?65", model):
        return "CRJ", "CRJ-200", "CRJ", Confidence.DERIVED
    # `CL600 2D24-900`, `CL-600-2D24 - 900` etc.
    m = re.match(r"^CL-?600-?2([A-Z])(\d)$", model)
    if m:  # truncated code, e.g. CL-600-2B1
        prefix = f"2{m.group(1)}{m.group(2)}"
        hits = [k for k in _BOMBARDIER_CL600 if k.startswith(prefix)]
        if len(hits) == 1:
            variant, gen = _BOMBARDIER_CL600[hits[0]]
            return ("CRJ" if gen == "CRJ" else "Challenger"), variant, gen, Confidence.DERIVED
        return "CRJ", None, "CRJ", Confidence.FAMILY_ONLY
    m = re.match(r"^CL-?600-?2([A-Z])(\d{2})", model)
    if m:
        code = f"2{m.group(1)}{m.group(2)}"
        if code in _BOMBARDIER_CL600:
            variant, gen = _BOMBARDIER_CL600[code]
            return ("CRJ" if gen == "CRJ" else "Challenger"), variant, gen, Confidence.DERIVED
    m = re.match(r"^CRJ-?(\d{3})", model)
    if m:
        return "CRJ", f"CRJ-{m.group(1)}", "CRJ", Confidence.EXACT

    m = re.match(r"^(DHC-?8|Q)-?(\d{3})?", model)
    if m:
        return "Dash 8", (f"Dash 8-{m.group(2)}" if m.group(2) else None), None, (
            Confidence.DERIVED if m.group(2) else Confidence.FAMILY_ONLY)
    return None, None, None, Confidence.UNRESOLVED


def _resolve_embraer(model: str) -> tuple[str | None, str | None, str | None, Confidence]:
    s = model
    # Bare numbers (`145LR`, `120`, `170`) and marketing names (`E175`)
    if re.match(r"^\d{3}", s):
        s = "EMB-" + s
    m = re.match(r"^E-?(1[0-9]{2})$", s)
    if m and m.group(1) in ("170", "175", "190", "195"):
        return "E-Jet", f"E{m.group(1)}", "E-Jet", Confidence.EXACT
    # ERJ 170-200 is the type-certificate name for the E175
    m = re.match(r"^(?:ERJ|EMB)-?170-?(\d{3})", s)
    if m:
        return ("E-Jet", {"100": "E170", "200": "E175"}.get(m.group(1)),
                "E-Jet", Confidence.DERIVED if m.group(1) in ("100", "200")
                else Confidence.FAMILY_ONLY)
    m = re.match(r"^(?:ERJ|EMB)-?190-?(\d{3})", s)
    if m:
        return ("E-Jet", {"100": "E190", "200": "E195"}.get(m.group(1)),
                "E-Jet", Confidence.DERIVED)
    m = re.match(r"^(?:ERJ|EMB)-?(1[0-9]{2})", s)
    if m:
        num = m.group(1)
        if num in ("135", "140", "145"):
            return "ERJ", f"ERJ-{num}", "ERJ", Confidence.EXACT
        if num == "120":
            return "EMB-120", "EMB-120 Brasilia", None, Confidence.EXACT
        if num in ("170", "175", "190", "195"):
            return "E-Jet", f"E{num}", "E-Jet", Confidence.EXACT
    return None, None, None, Confidence.UNRESOLVED


_FOKKER_MK = {"0100": "Fokker 100", "0070": "Fokker 70", "0060": "Fokker 60"}


def _resolve_fokker(model: str) -> tuple[str | None, str | None, str | None, Confidence]:
    model = model.replace(".", "").replace("FK", "F")
    m = re.match(r"^F-?(\d{2})-?MK-?0?(\d{2,4})$", model)
    if m:
        base, mk = m.group(1), m.group(2).lstrip("0") or "0"
        if base == "28" and mk in ("100", "70", "60"):
            return "Fokker 100", f"Fokker {mk}", None, Confidence.DERIVED
        if base == "27":
            return "Fokker 27", f"Fokker 27-{mk}", None, Confidence.DERIVED
    m = re.match(r"^F-?27-?(\d{3})$", model)
    if m:
        return "Fokker 27", f"Fokker 27-{m.group(1)}", None, Confidence.DERIVED
    m = re.match(r"^F-?28-?(\d{4})$", model)
    if m:
        return "Fokker 28", "Fokker 28", None, Confidence.DERIVED
    m = re.match(r"^100$", model)
    if m:
        return "Fokker 100", "Fokker 100", None, Confidence.DERIVED
    m = re.match(r"^F-?28-?MK-?(\d{4})$", model)
    if m and m.group(1) in _FOKKER_MK:
        return "Fokker 100", _FOKKER_MK[m.group(1)], None, Confidence.DERIVED
    m = re.match(r"^F-?(28|50|70|100)", model)
    if m:
        return f"Fokker {m.group(1)}", f"Fokker {m.group(1)}", None, Confidence.EXACT
    return None, None, None, Confidence.UNRESOLVED


def _resolve_lockheed(model: str) -> tuple[str | None, str | None, str | None, Confidence]:
    if re.match(r"^L-?1011", model):
        return "L-1011", "L-1011 TriStar", None, Confidence.DERIVED
    if re.match(r"^L?-?382", model):
        return "L-382", "L-382 Hercules", None, Confidence.DERIVED
    return None, None, None, Confidence.UNRESOLVED


def _resolve_airbus_canada(model: str) -> tuple[str | None, str | None, str | None, Confidence]:
    """A220, formerly Bombardier CSeries. Current fleet, so worth resolving."""
    if re.match(r"^BD-?500-?1A11", model):
        return "A220", "A220-300", None, Confidence.DERIVED
    if re.match(r"^BD-?500-?1A10", model):
        return "A220", "A220-100", None, Confidence.DERIVED
    m = re.match(r"^A?-?220-?(\d{3})?N?$", model)
    if m:
        return "A220", (f"A220-{m.group(1)}" if m.group(1) else None), None, (
            Confidence.DERIVED if m.group(1) else Confidence.FAMILY_ONLY)
    return None, None, None, Confidence.UNRESOLVED


def _resolve_shorts(model: str) -> tuple[str | None, str | None, str | None, Confidence]:
    m = re.match(r"^SD-?3-?(30|60)", model)
    if m:
        return "Shorts 3", f"Shorts 3{m.group(1)}", None, Confidence.DERIVED
    return None, None, None, Confidence.UNRESOLVED


def _resolve_swearingen(model: str) -> tuple[str | None, str | None, str | None, Confidence]:
    if re.match(r"^SW-?[234]", model) or "METRO" in model:
        return "Metroliner", "SA227 Metroliner", None, Confidence.DERIVED
    return None, None, None, Confidence.UNRESOLVED


def _resolve_dehavilland(model: str) -> tuple[str | None, str | None, str | None, Confidence]:
    m = re.match(r"^(?:DHC-?8|DH-?8|DASH-?8|Q)-?(\d{3})?", model)
    if m:
        n = m.group(1)
        return "Dash 8", (f"Dash 8-{n[0]}00" if n else None), None, (
            Confidence.DERIVED if n else Confidence.FAMILY_ONLY)
    m = re.match(r"^DHC-?(\d)-?(\d{3})?", model)
    if m:
        fam = f"DHC-{m.group(1)}"
        return fam, fam, None, Confidence.FAMILY_ONLY
    return None, None, None, Confidence.UNRESOLVED


def _resolve_simple(family_prefix: str, model: str) -> tuple[str | None, str | None, str | None, Confidence]:
    m = re.match(r"^([A-Z]*)-?(\d{2,4})([A-Z]{0,2})", model)
    if m:
        num = m.group(2)
        suffix = m.group(3)
        return family_prefix, f"{family_prefix} {num}{suffix}".strip(), None, Confidence.DERIVED
    return family_prefix, None, None, Confidence.FAMILY_ONLY


# --------------------------------------------------------------- entry point

def resolve(make: str, model: str) -> AircraftType:
    manufacturer = normalize_manufacturer(make)
    tidy = _tidy(model)

    if manufacturer is None or not tidy:
        return AircraftType(manufacturer, None, None, None,
                            Confidence.UNRESOLVED, make, model)

    dispatch = {
        "Boeing": _resolve_boeing,
        "Airbus": _resolve_airbus,
        "McDonnell Douglas": _resolve_mcdonnell,
        "Bombardier": _resolve_bombardier,
        "Embraer": _resolve_embraer,
        "Fokker": _resolve_fokker,
        "Lockheed": _resolve_lockheed,
        "De Havilland Canada": _resolve_dehavilland,
        "Shorts": _resolve_shorts,
        "Swearingen": _resolve_swearingen,
        "A220": _resolve_airbus_canada,
    }

    if manufacturer == "Airbus" and re.match(r"^A?-?220", tidy):
        manufacturer = "A220"

    if manufacturer in dispatch:
        family, variant, generation, conf = dispatch[manufacturer](tidy)
    elif manufacturer in ("Beechcraft", "Saab", "ATR",
                          "De Havilland Canada", "Dornier", "BAe"):
        family, variant, generation, conf = _resolve_simple(manufacturer, tidy)
    else:
        family, variant, generation, conf = None, None, None, Confidence.UNRESOLVED

    if family is None:
        conf = Confidence.UNRESOLVED

    return AircraftType(manufacturer, family, variant, generation,
                        conf, make, model)


# --------------------------------------------------------------- ICAO types

#: ICAO type designators, as reported by flight-tracking data, mapped to the
#: (make, model) strings this module already understands.
#:
#: This is the bridge between live flight data and the NTSB corpus. AeroAPI
#: says `BCS3`; NTSB files an `AIRBUS CANADA LP / BD-500-1A11`. Without a
#: mapping there is no way to ask what has happened to the aircraft a
#: passenger is actually booked on.
#:
#: Deliberately limited to types in scheduled airline service. A designator
#: that is not here resolves to UNRESOLVED rather than to something close,
#: which is the same rule the rest of this module follows.
ICAO_DESIGNATORS: dict[str, tuple[str, str]] = {
    # Boeing 737
    "B731": ("BOEING", "737-100"), "B732": ("BOEING", "737-200"),
    "B733": ("BOEING", "737-300"), "B734": ("BOEING", "737-400"),
    "B735": ("BOEING", "737-500"), "B736": ("BOEING", "737-600"),
    "B737": ("BOEING", "737-700"), "B738": ("BOEING", "737-800"),
    "B739": ("BOEING", "737-900"),
    # 737 MAX - a different generation behind a near-identical designator
    "B37M": ("BOEING", "737-7"), "B38M": ("BOEING", "737-8"),
    "B39M": ("BOEING", "737-9"), "B3XM": ("BOEING", "737-10"),
    # Boeing widebodies
    "B741": ("BOEING", "747-100"), "B742": ("BOEING", "747-200"),
    "B743": ("BOEING", "747-300"), "B744": ("BOEING", "747-400"),
    "B748": ("BOEING", "747-800"),
    "B752": ("BOEING", "757-200"), "B753": ("BOEING", "757-300"),
    "B762": ("BOEING", "767-200"), "B763": ("BOEING", "767-300"),
    "B764": ("BOEING", "767-400"),
    "B772": ("BOEING", "777-200"), "B77L": ("BOEING", "777-200"),
    "B773": ("BOEING", "777-300"), "B77W": ("BOEING", "777-300"),
    "B788": ("BOEING", "787-8"), "B789": ("BOEING", "787-9"),
    "B78X": ("BOEING", "787-10"),
    "B712": ("BOEING", "717-200"),
    # Airbus
    "A318": ("AIRBUS", "A318-100"), "A319": ("AIRBUS", "A319-100"),
    "A320": ("AIRBUS", "A320-200"), "A321": ("AIRBUS", "A321-200"),
    "A19N": ("AIRBUS", "A319NEO"), "A20N": ("AIRBUS", "A320NEO"),
    "A21N": ("AIRBUS", "A321NEO"),
    "A306": ("AIRBUS", "A300-600"), "A310": ("AIRBUS", "A310"),
    "A332": ("AIRBUS", "A330-200"), "A333": ("AIRBUS", "A330-300"),
    "A338": ("AIRBUS", "A330-800"), "A339": ("AIRBUS", "A330-900"),
    "A342": ("AIRBUS", "A340-200"), "A343": ("AIRBUS", "A340-300"),
    "A345": ("AIRBUS", "A340-500"), "A346": ("AIRBUS", "A340-600"),
    "A359": ("AIRBUS", "A350-900"), "A35K": ("AIRBUS", "A350-1000"),
    "A388": ("AIRBUS", "A380-800"),
    # A220, formerly Bombardier CSeries
    "BCS1": ("AIRBUS CANADA LP", "BD-500-1A10"),
    "BCS3": ("AIRBUS CANADA LP", "BD-500-1A11"),
    # Embraer
    "E170": ("EMBRAER", "ERJ 170-100"), "E75S": ("EMBRAER", "ERJ 170-200"),
    "E75L": ("EMBRAER", "ERJ 170-200"), "E175": ("EMBRAER", "ERJ 170-200"),
    "E190": ("EMBRAER", "ERJ 190-100"), "E195": ("EMBRAER", "ERJ 190-200"),
    "E290": ("EMBRAER", "ERJ 190-300"), "E295": ("EMBRAER", "ERJ 190-400"),
    "E135": ("EMBRAER", "EMB-135"), "E145": ("EMBRAER", "EMB-145"),
    "E120": ("EMBRAER", "EMB-120"),
    # Bombardier regional jets
    "CRJ1": ("BOMBARDIER", "CL-600-2B19"), "CRJ2": ("BOMBARDIER", "CL-600-2B19"),
    "CRJ7": ("BOMBARDIER", "CL-600-2C10"), "CRJ9": ("BOMBARDIER", "CL-600-2D24"),
    "CRJX": ("BOMBARDIER", "CL-600-2E25"),
    # Turboprops in scheduled service
    "DH8A": ("DE HAVILLAND", "DHC-8-100"), "DH8B": ("DE HAVILLAND", "DHC-8-200"),
    "DH8C": ("DE HAVILLAND", "DHC-8-300"), "DH8D": ("DE HAVILLAND", "DHC-8-400"),
    "AT43": ("ATR", "ATR-42-300"), "AT45": ("ATR", "ATR-42-500"),
    "AT72": ("ATR", "ATR-72-200"), "AT76": ("ATR", "ATR-72-600"),
    "SF34": ("SAAB", "SAAB 340B"), "B190": ("BEECH", "1900D"),
    # McDonnell Douglas
    "MD82": ("MCDONNELL DOUGLAS", "MD-82"), "MD83": ("MCDONNELL DOUGLAS", "MD-83"),
    "MD87": ("MCDONNELL DOUGLAS", "MD-87"), "MD88": ("MCDONNELL DOUGLAS", "MD-88"),
    "MD90": ("MCDONNELL DOUGLAS", "MD-90"), "MD11": ("MCDONNELL DOUGLAS", "MD-11"),
    "DC93": ("MCDONNELL DOUGLAS", "DC-9-30"),
    "DC10": ("MCDONNELL DOUGLAS", "DC-10"),
}


def resolve_icao(designator: str) -> AircraftType:
    """Resolve an ICAO type designator such as `B38M` or `BCS3`.

    Unknown designators return UNRESOLVED. Guessing from a prefix would be
    worse than useless here: `B738` and `B38M` differ by one character and
    are twenty years apart in certification.
    """
    key = (designator or "").strip().upper()
    pair = ICAO_DESIGNATORS.get(key)
    if pair is None:
        return AircraftType(None, None, None, None, Confidence.UNRESOLVED,
                            designator or "", designator or "")
    return resolve(*pair)
