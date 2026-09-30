# install-to: app/retrieval
"""Turning what a person types into what the flight data provider wants.

Fliers know IATA codes. They have three letters, they are on the boarding
pass, and they are what anyone means when they say "LAX". The flight data
provider speaks ICAO: four letters, and the mapping is regular only in the
contiguous United States.

The obvious shortcut - prepend K to any three-letter code - is wrong in
exactly the places that matter. Anchorage is PANC, Honolulu is PHNL, San
Juan is TJSJ, and nothing outside the United States follows the rule at
all. KANC does not exist, so the shortcut does not fail loudly; it
produces a code the provider has never heard of and a search that finds
nothing.

That failure has the shape this whole system is built to avoid: a wrong
answer that looks like an absence. So a guess is always reported as a
guess, and the interface shows what a code resolved to.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

#: How a four-letter code was arrived at. The caller shows this, because a
#: code that was looked up and one that was assembled by rule deserve
#: different confidence.
EXACT = "exact"          # typed as ICAO already
KNOWN = "known"          # found in the table
ASSUMED = "assumed"      # built by the K rule, unverified


@dataclass(frozen=True)
class Airport:
    """A resolved code, and how much to trust it."""

    code: str            # the ICAO code to search with
    how: str             # EXACT, KNOWN or ASSUMED
    typed: str           # what the person actually entered

    @property
    def is_guess(self) -> bool:
        return self.how == ASSUMED

    def note(self) -> str | None:
        """What to tell the reader, if anything."""
        if self.how == KNOWN:
            return f"{self.typed} is {self.code}"
        if self.how == ASSUMED:
            return (f"{self.typed} was read as {self.code}. That is a guess "
                    f"from the US three-letter pattern, not a lookup - if "
                    f"the route looks wrong, enter the four-letter code.")
        return None


#: IATA to ICAO. Scoped to airports a passenger might plausibly fly from,
#: which is a different set from every airport that exists. The Alaska,
#: Hawaii and territory entries are here specifically because they break
#: the K rule; without them the rule fails silently on them.
_IATA: dict[str, str] = {
    # --- US, largest by passenger volume
    "ATL": "KATL", "DFW": "KDFW", "DEN": "KDEN", "ORD": "KORD",
    "LAX": "KLAX", "CLT": "KCLT", "MCO": "KMCO", "LAS": "KLAS",
    "PHX": "KPHX", "MIA": "KMIA", "SEA": "KSEA", "IAH": "KIAH",
    "JFK": "KJFK", "EWR": "KEWR", "SFO": "KSFO", "DTW": "KDTW",
    "BOS": "KBOS", "MSP": "KMSP", "FLL": "KFLL", "LGA": "KLGA",
    "PHL": "KPHL", "SLC": "KSLC", "BWI": "KBWI", "DCA": "KDCA",
    "IAD": "KIAD", "SAN": "KSAN", "TPA": "KTPA", "AUS": "KAUS",
    "BNA": "KBNA", "MDW": "KMDW", "RDU": "KRDU", "HOU": "KHOU",
    "STL": "KSTL", "DAL": "KDAL", "PDX": "KPDX", "SMF": "KSMF",
    "MSY": "KMSY", "SJC": "KSJC", "SNA": "KSNA", "MCI": "KMCI",
    "OAK": "KOAK", "RSW": "KRSW", "CLE": "KCLE", "PIT": "KPIT",
    "CVG": "KCVG", "IND": "KIND", "CMH": "KCMH", "PBI": "KPBI",
    "JAX": "KJAX", "MKE": "KMKE", "BDL": "KBDL", "ONT": "KONT",
    "BUF": "KBUF", "OMA": "KOMA", "BOI": "KBOI", "RNO": "KRNO",
    "OKC": "KOKC", "TUS": "KTUS", "ABQ": "KABQ", "MEM": "KMEM",
    "RIC": "KRIC", "SAT": "KSAT", "ELP": "KELP", "GRR": "KGRR",
    "TUL": "KTUL", "ORF": "KORF", "PVD": "KPVD", "SDF": "KSDF",
    "CHS": "KCHS", "GSP": "KGSP", "SAV": "KSAV", "MYR": "KMYR",
    "ALB": "KALB", "SYR": "KSYR", "ROC": "KROC", "BHM": "KBHM",
    "LIT": "KLIT", "DSM": "KDSM", "MSN": "KMSN", "GEG": "KGEG",
    "ANC": "PANC", "FAI": "PAFA", "JNU": "PAJN",       # Alaska: P
    "HNL": "PHNL", "OGG": "PHOG", "KOA": "PHKO",       # Hawaii: PH
    "LIH": "PHLI", "ITO": "PHTO",
    "SJU": "TJSJ", "STT": "TIST", "STX": "TISX",       # territories
    "GUM": "PGUM",

    # --- Canada
    "YYZ": "CYYZ", "YVR": "CYVR", "YUL": "CYUL", "YYC": "CYYC",
    "YEG": "CYEG", "YOW": "CYOW", "YHZ": "CYHZ", "YWG": "CYWG",

    # --- Mexico, Caribbean, Central and South America
    "MEX": "MMMX", "CUN": "MMUN", "GDL": "MMGL", "MTY": "MMMY",
    "SJD": "MMSD", "PVR": "MMPR", "NAS": "MYNN", "MBJ": "MKJS",
    "PUJ": "MDPC", "SDQ": "MDSD", "AUA": "TNCA", "CUR": "TNCC",
    "PTY": "MPTO", "SJO": "MROC", "GRU": "SBGR", "GIG": "SBGL",
    "EZE": "SAEZ", "SCL": "SCEL", "BOG": "SKBO", "LIM": "SPJC",

    # --- Europe
    "LHR": "EGLL", "LGW": "EGKK", "STN": "EGSS", "MAN": "EGCC",
    "EDI": "EGPH", "DUB": "EIDW", "CDG": "LFPG", "ORY": "LFPO",
    "NCE": "LFMN", "AMS": "EHAM", "BRU": "EBBR", "FRA": "EDDF",
    "MUC": "EDDM", "BER": "EDDB", "DUS": "EDDL", "HAM": "EDDH",
    "ZRH": "LSZH", "GVA": "LSGG", "VIE": "LOWW", "CPH": "EKCH",
    "ARN": "ESSA", "OSL": "ENGM", "HEL": "EFHK", "KEF": "BIKF",
    "MAD": "LEMD", "BCN": "LEBL", "AGP": "LEMG", "PMI": "LEPA",
    "LIS": "LPPT", "OPO": "LPPR", "FCO": "LIRF", "MXP": "LIMC",
    "VCE": "LIPZ", "NAP": "LIRN", "ATH": "LGAV", "IST": "LTFM",
    "WAW": "EPWA", "PRG": "LKPR", "BUD": "LHBP",

    # --- Middle East and Africa
    "DXB": "OMDB", "AUH": "OMAA", "DOH": "OTHH", "RUH": "OERK",
    "JED": "OEJN", "TLV": "LLBG", "CAI": "HECA", "JNB": "FAOR",
    "CPT": "FACT", "NBO": "HKJK", "CMN": "GMMN", "LOS": "DNMM",

    # --- Asia and Oceania
    "NRT": "RJAA", "HND": "RJTT", "KIX": "RJBB", "CTS": "RJCC",
    "ICN": "RKSI", "GMP": "RKSS", "PEK": "ZBAA", "PKX": "ZBAD",
    "PVG": "ZSPD", "SHA": "ZSSS", "CAN": "ZGGG", "SZX": "ZGSZ",
    "HKG": "VHHH", "TPE": "RCTP", "SIN": "WSSS", "KUL": "WMKK",
    "BKK": "VTBS", "DMK": "VTBD", "HAN": "VVNB", "SGN": "VVTS",
    "MNL": "RPLL", "CGK": "WIII", "DPS": "WADD", "DEL": "VIDP",
    "BOM": "VABB", "BLR": "VOBL", "MAA": "VOMM", "HYD": "VOHS",
    "CCU": "VECC", "SYD": "YSSY", "MEL": "YMML", "BNE": "YBBN",
    "PER": "YPPH", "ADL": "YPAD", "AKL": "NZAA", "CHC": "NZCH",
    "WLG": "NZWN",
}

#: IANA timezone per ICAO code, for the same airports `_IATA` resolves to.
#: Keyed by the code a passenger's trip actually resolves to, not by what
#: they typed - so an ASSUMED (K-rule) guess never gets a timezone entry
#: here, on purpose. A guessed code is already an unverified guess at the
#: airport; guessing its timezone on top of that compounds one uncertain
#: answer with another, and the whole point of this table is to replace a
#: guess with a lookup, not to add a second guess next to the first.
#:
#: This is the one place in this file where a wrong entry is not caught by
#: any downstream check the way a bad airport code is. Arizona does not
#: observe DST (America/Phoenix, not just "Mountain"), Indiana is mostly
#: Eastern with `America/Indiana/Indianapolis` rather than a bare
#: `America/New_York`, and a handful of international entries use a
#: region-specific zone name rather than the country's largest city
#: (Vietnam is `Asia/Ho_Chi_Minh` for Hanoi too; Indonesia's Bali uses
#: `Asia/Makassar`, not `Asia/Jakarta`). Each of those is deliberate, not
#: an oversight to be "simplified" later.
_TIMEZONE: dict[str, str] = {
    # --- US, largest by passenger volume
    "KATL": "America/New_York", "KDFW": "America/Chicago",
    "KDEN": "America/Denver", "KORD": "America/Chicago",
    "KLAX": "America/Los_Angeles", "KCLT": "America/New_York",
    "KMCO": "America/New_York", "KLAS": "America/Los_Angeles",
    "KPHX": "America/Phoenix", "KMIA": "America/New_York",
    "KSEA": "America/Los_Angeles", "KIAH": "America/Chicago",
    "KJFK": "America/New_York", "KEWR": "America/New_York",
    "KSFO": "America/Los_Angeles", "KDTW": "America/New_York",
    "KBOS": "America/New_York", "KMSP": "America/Chicago",
    "KFLL": "America/New_York", "KLGA": "America/New_York",
    "KPHL": "America/New_York", "KSLC": "America/Denver",
    "KBWI": "America/New_York", "KDCA": "America/New_York",
    "KIAD": "America/New_York", "KSAN": "America/Los_Angeles",
    "KTPA": "America/New_York", "KAUS": "America/Chicago",
    "KBNA": "America/Chicago", "KMDW": "America/Chicago",
    "KRDU": "America/New_York", "KHOU": "America/Chicago",
    "KSTL": "America/Chicago", "KDAL": "America/Chicago",
    "KPDX": "America/Los_Angeles", "KSMF": "America/Los_Angeles",
    "KMSY": "America/Chicago", "KSJC": "America/Los_Angeles",
    "KSNA": "America/Los_Angeles", "KMCI": "America/Chicago",
    "KOAK": "America/Los_Angeles", "KRSW": "America/New_York",
    "KCLE": "America/New_York", "KPIT": "America/New_York",
    "KCVG": "America/New_York",
    "KIND": "America/Indiana/Indianapolis",
    "KCMH": "America/New_York", "KPBI": "America/New_York",
    "KJAX": "America/New_York", "KMKE": "America/Chicago",
    "KBDL": "America/New_York", "KONT": "America/Los_Angeles",
    "KBUF": "America/New_York", "KOMA": "America/Chicago",
    "KBOI": "America/Boise", "KRNO": "America/Los_Angeles",
    "KOKC": "America/Chicago", "KTUS": "America/Phoenix",
    "KABQ": "America/Denver", "KMEM": "America/Chicago",
    "KRIC": "America/New_York", "KSAT": "America/Chicago",
    "KELP": "America/Denver", "KGRR": "America/New_York",
    "KTUL": "America/Chicago", "KORF": "America/New_York",
    "KPVD": "America/New_York", "KSDF": "America/New_York",
    "KCHS": "America/New_York", "KGSP": "America/New_York",
    "KSAV": "America/New_York", "KMYR": "America/New_York",
    "KALB": "America/New_York", "KSYR": "America/New_York",
    "KROC": "America/New_York", "KBHM": "America/Chicago",
    "KLIT": "America/Chicago", "KDSM": "America/Chicago",
    "KMSN": "America/Chicago", "KGEG": "America/Los_Angeles",
    "PANC": "America/Anchorage", "PAFA": "America/Anchorage",
    "PAJN": "America/Juneau",
    "PHNL": "Pacific/Honolulu", "PHOG": "Pacific/Honolulu",
    "PHKO": "Pacific/Honolulu", "PHLI": "Pacific/Honolulu",
    "PHTO": "Pacific/Honolulu",
    "TJSJ": "America/Puerto_Rico",
    "TIST": "America/St_Thomas", "TISX": "America/St_Thomas",
    "PGUM": "Pacific/Guam",

    # --- Canada
    "CYYZ": "America/Toronto", "CYVR": "America/Vancouver",
    "CYUL": "America/Toronto", "CYYC": "America/Edmonton",
    "CYEG": "America/Edmonton", "CYOW": "America/Toronto",
    "CYHZ": "America/Halifax", "CYWG": "America/Winnipeg",

    # --- Mexico, Caribbean, Central and South America
    "MMMX": "America/Mexico_City", "MMUN": "America/Cancun",
    "MMGL": "America/Mexico_City", "MMMY": "America/Monterrey",
    "MMSD": "America/Mazatlan", "MMPR": "America/Bahia_Banderas",
    "MYNN": "America/Nassau", "MKJS": "America/Jamaica",
    "MDPC": "America/Santo_Domingo", "MDSD": "America/Santo_Domingo",
    "TNCA": "America/Aruba", "TNCC": "America/Curacao",
    "MPTO": "America/Panama", "MROC": "America/Costa_Rica",
    "SBGR": "America/Sao_Paulo", "SBGL": "America/Sao_Paulo",
    "SAEZ": "America/Argentina/Buenos_Aires", "SCEL": "America/Santiago",
    "SKBO": "America/Bogota", "SPJC": "America/Lima",

    # --- Europe
    "EGLL": "Europe/London", "EGKK": "Europe/London",
    "EGSS": "Europe/London", "EGCC": "Europe/London",
    "EGPH": "Europe/London", "EIDW": "Europe/Dublin",
    "LFPG": "Europe/Paris", "LFPO": "Europe/Paris",
    "LFMN": "Europe/Paris", "EHAM": "Europe/Amsterdam",
    "EBBR": "Europe/Brussels", "EDDF": "Europe/Berlin",
    "EDDM": "Europe/Berlin", "EDDB": "Europe/Berlin",
    "EDDL": "Europe/Berlin", "EDDH": "Europe/Berlin",
    "LSZH": "Europe/Zurich", "LSGG": "Europe/Zurich",
    "LOWW": "Europe/Vienna", "EKCH": "Europe/Copenhagen",
    "ESSA": "Europe/Stockholm", "ENGM": "Europe/Oslo",
    "EFHK": "Europe/Helsinki", "BIKF": "Atlantic/Reykjavik",
    "LEMD": "Europe/Madrid", "LEBL": "Europe/Madrid",
    "LEMG": "Europe/Madrid", "LEPA": "Europe/Madrid",
    "LPPT": "Europe/Lisbon", "LPPR": "Europe/Lisbon",
    "LIRF": "Europe/Rome", "LIMC": "Europe/Rome",
    "LIPZ": "Europe/Rome", "LIRN": "Europe/Rome",
    "LGAV": "Europe/Athens", "LTFM": "Europe/Istanbul",
    "EPWA": "Europe/Warsaw", "LKPR": "Europe/Prague",
    "LHBP": "Europe/Budapest",

    # --- Middle East and Africa
    "OMDB": "Asia/Dubai", "OMAA": "Asia/Dubai", "OTHH": "Asia/Qatar",
    "OERK": "Asia/Riyadh", "OEJN": "Asia/Riyadh",
    "LLBG": "Asia/Jerusalem", "HECA": "Africa/Cairo",
    "FAOR": "Africa/Johannesburg", "FACT": "Africa/Johannesburg",
    "HKJK": "Africa/Nairobi", "GMMN": "Africa/Casablanca",
    "DNMM": "Africa/Lagos",

    # --- Asia and Oceania
    "RJAA": "Asia/Tokyo", "RJTT": "Asia/Tokyo", "RJBB": "Asia/Tokyo",
    "RJCC": "Asia/Tokyo", "RKSI": "Asia/Seoul", "RKSS": "Asia/Seoul",
    "ZBAA": "Asia/Shanghai", "ZBAD": "Asia/Shanghai",
    "ZSPD": "Asia/Shanghai", "ZSSS": "Asia/Shanghai",
    "ZGGG": "Asia/Shanghai", "ZGSZ": "Asia/Shanghai",
    "VHHH": "Asia/Hong_Kong", "RCTP": "Asia/Taipei",
    "WSSS": "Asia/Singapore", "WMKK": "Asia/Kuala_Lumpur",
    "VTBS": "Asia/Bangkok", "VTBD": "Asia/Bangkok",
    "VVNB": "Asia/Ho_Chi_Minh", "VVTS": "Asia/Ho_Chi_Minh",
    "RPLL": "Asia/Manila", "WIII": "Asia/Jakarta",
    "WADD": "Asia/Makassar",
    "VIDP": "Asia/Kolkata", "VABB": "Asia/Kolkata",
    "VOBL": "Asia/Kolkata", "VOMM": "Asia/Kolkata",
    "VOHS": "Asia/Kolkata", "VECC": "Asia/Kolkata",
    "YSSY": "Australia/Sydney", "YMML": "Australia/Melbourne",
    "YBBN": "Australia/Brisbane", "YPPH": "Australia/Perth",
    "YPAD": "Australia/Adelaide",
    "NZAA": "Pacific/Auckland", "NZCH": "Pacific/Auckland",
    "NZWN": "Pacific/Auckland",
}

#: Prefixes that mean a four-letter code is already ICAO. Not exhaustive
#: and does not need to be - anything four characters long is passed
#: through, because the provider is the authority on whether it exists.
_ICAO_LENGTH = 4


def resolve_airport(typed: str) -> Airport | None:
    """Turn what a person typed into a code the provider understands.

    Returns None for input that could not be a code at all, so the caller
    can say so rather than searching for something meaningless.
    """
    if not typed:
        return None
    code = "".join(typed.split()).upper()
    if not code.isalpha():
        return None

    if len(code) == _ICAO_LENGTH:
        return Airport(code=code, how=EXACT, typed=code)

    if len(code) == 3:
        known = _IATA.get(code)
        if known:
            return Airport(code=known, how=KNOWN, typed=code)
        # The K rule, applied only after the lookup fails and always
        # labelled as a guess. It is right for most of the contiguous
        # United States and wrong everywhere else, which is why it is the
        # fallback rather than the first move.
        return Airport(code=f"K{code}", how=ASSUMED, typed=code)

    return None


def resolve_pair(origin: str, dest: str) -> tuple[Airport | None,
                                                  Airport | None]:
    """Both ends of a trip, resolved independently."""
    return resolve_airport(origin), resolve_airport(dest)


def timezone_for(code: str) -> str | None:
    """The IANA zone for a resolved airport code, or None if this table
    doesn't have it - either the code is real but not in `_TIMEZONE` yet,
    or it's an ASSUMED guess that was never entered here on purpose (see
    `_TIMEZONE`'s docstring). Either way, None means "don't guess a
    timezone on top of it" to the caller, not "treat it as UTC" - that
    decision belongs to whoever is converting a time, not to this lookup.
    """
    return _TIMEZONE.get(code)


def local_to_utc(date: str, time_of_day: str, tz_name: str
                 ) -> tuple[str, str]:
    """Convert a local wall-clock date and time to their UTC equivalent.

    `date` is "YYYY-MM-DD", `time_of_day` is "HH:MM", both as stated in
    the zone named by `tz_name`. Returns the same two formats, in UTC -
    the date can differ from the one given, since a local evening in most
    zones east of UTC is already the next UTC day, and a local time near
    midnight can cross a day either direction depending on the offset.

    DST is resolved from the date itself: `zoneinfo` (stdlib, no extra
    dependency) knows when each zone's clocks change and applies whichever
    offset was in effect on that day, so the same clock time converts
    differently in June than in December without any special-casing here.
    A local time that lands in a spring-forward gap (it never happened) or
    a fall-back overlap (it happened twice) is still resolved by
    `zoneinfo` rather than rejected - the ambiguous hour picks its first
    occurrence, which is wrong at most once a year and only by an hour,
    against silently refusing every departure time near a transition.
    """
    local = datetime.strptime(f"{date} {time_of_day}", "%Y-%m-%d %H:%M")
    local = local.replace(tzinfo=ZoneInfo(tz_name))
    as_utc = local.astimezone(timezone.utc)
    return as_utc.strftime("%Y-%m-%d"), as_utc.strftime("%H:%M")
