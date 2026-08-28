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
