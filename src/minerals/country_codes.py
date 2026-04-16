"""
ISO 3166-1 numeric country code → standard country name mapping.

CEPII BACI data uses ISO 3166-1 numeric codes for some countries,
mixed with string names for others.  This module normalizes everything
to standard string names so downstream analysis is interpretable.

Sources:
  - ISO 3166-1 numeric standard
  - CEPII BACI country concordance (baci_country_codes.csv)
  - UN Comtrade M49 codes for a few non-standard entries

Usage
-----
    from src.minerals.country_codes import normalize_country_names
    df = normalize_country_names(df)   # in-place on 'exporter'/'importer'
"""

from __future__ import annotations

from typing import Dict

# Complete ISO 3166-1 numeric → standard name mapping
# Covers all codes observed in CEPII BACI graphite/lithium/cobalt/nickel data
ISO_NUMERIC: Dict[str, str] = {
    "004": "Afghanistan",
    "008": "Albania",
    "012": "Algeria",
    "016": "American Samoa",
    "020": "Andorra",
    "024": "Angola",
    "028": "Antigua and Barbuda",
    "031": "Azerbaijan",
    "032": "Argentina",
    "036": "Australia",
    "040": "Austria",
    "044": "Bahamas",
    "048": "Bahrain",
    "050": "Bangladesh",
    "051": "Armenia",
    "052": "Barbados",
    "056": "Belgium",
    "058": "Belgium-Luxembourg",   # old CEPII code pre-2000
    "060": "Bermuda",
    "064": "Bhutan",
    "068": "Bolivia",
    "070": "Bosnia and Herzegovina",
    "072": "Botswana",
    "076": "Brazil",
    "084": "Belize",
    "090": "Solomon Islands",
    "092": "British Virgin Islands",
    "096": "Brunei",
    "100": "Bulgaria",
    "104": "Myanmar",
    "108": "Burundi",
    "112": "Belarus",
    "116": "Cambodia",
    "120": "Cameroon",
    "124": "Canada",
    "132": "Cabo Verde",
    "136": "Cayman Islands",
    "140": "Central African Republic",
    "144": "Sri Lanka",
    "148": "Chad",
    "152": "Chile",
    "156": "China",
    "158": "Taiwan",
    "162": "Christmas Island",
    "166": "Cocos Islands",
    "170": "Colombia",
    "174": "Comoros",
    "175": "Mayotte",
    "178": "Congo",
    "180": "DRC",
    "184": "Cook Islands",
    "188": "Costa Rica",
    "191": "Croatia",
    "192": "Cuba",
    "196": "Cyprus",
    "203": "Czechia",
    "204": "Benin",
    "208": "Denmark",
    "212": "Dominica",
    "214": "Dominican Republic",
    "218": "Ecuador",
    "818": "Egypt",
    "222": "El Salvador",
    "226": "Equatorial Guinea",
    "232": "Eritrea",
    "233": "Estonia",
    "231": "Ethiopia",
    "238": "Falkland Islands",
    "242": "Fiji",
    "246": "Finland",
    "250": "France",
    "251": "France (overseas)",
    "258": "French Polynesia",
    "266": "Gabon",
    "270": "Gambia",
    "268": "Georgia",
    "276": "Germany",
    "288": "Ghana",
    "292": "Gibraltar",
    "300": "Greece",
    "304": "Greenland",
    "308": "Grenada",
    "320": "Guatemala",
    "324": "Guinea",
    "624": "Guinea-Bissau",
    "328": "Guyana",
    "332": "Haiti",
    "340": "Honduras",
    "344": "Hong Kong",
    "348": "Hungary",
    "352": "Iceland",
    "356": "India",
    "360": "Indonesia",
    "364": "Iran",
    "368": "Iraq",
    "372": "Ireland",
    "376": "Israel",
    "380": "Italy",
    "384": "Ivory Coast",
    "388": "Jamaica",
    "392": "Japan",
    "400": "Jordan",
    "398": "Kazakhstan",
    "404": "Kenya",
    "408": "North Korea",
    "410": "South Korea",
    "414": "Kuwait",
    "417": "Kyrgyzstan",
    "418": "Laos",
    "422": "Lebanon",
    "426": "Lesotho",
    "428": "Latvia",
    "430": "Liberia",
    "434": "Libya",
    "438": "Liechtenstein",
    "440": "Lithuania",
    "442": "Luxembourg",
    "446": "Macao",
    "450": "Madagascar",
    "454": "Malawi",
    "458": "Malaysia",
    "462": "Maldives",
    "466": "Mali",
    "470": "Malta",
    "478": "Mauritania",
    "480": "Mauritius",
    "484": "Mexico",
    "490": "Taiwan",
    "496": "Mongolia",
    "498": "Moldova",
    "499": "Montenegro",
    "500": "Montserrat",
    "504": "Morocco",
    "508": "Mozambique",
    "512": "Oman",
    "516": "Namibia",
    "524": "Nepal",
    "528": "Netherlands",
    "530": "Netherlands Antilles",
    "531": "Curacao",
    "533": "Aruba",
    "540": "New Caledonia",
    "548": "Vanuatu",
    "554": "New Zealand",
    "558": "Nicaragua",
    "562": "Niger",
    "566": "Nigeria",
    "570": "Niue",
    "578": "Norway",
    "579": "Norway",   # CEPII variant code
    "583": "Micronesia",
    "584": "Marshall Islands",
    "585": "Palau",
    "586": "Pakistan",
    "591": "Panama",
    "598": "Papua New Guinea",
    "600": "Paraguay",
    "604": "Peru",
    "608": "Philippines",
    "616": "Poland",
    "620": "Portugal",
    "626": "Timor-Leste",
    "630": "Puerto Rico",
    "634": "Qatar",
    "642": "Romania",
    "643": "Russia",
    "646": "Rwanda",
    "654": "Saint Helena",
    "659": "Saint Kitts and Nevis",
    "660": "Anguilla",
    "662": "Saint Lucia",
    "670": "Saint Vincent and the Grenadines",
    "674": "San Marino",
    "678": "Sao Tome and Principe",
    "682": "Saudi Arabia",
    "686": "Senegal",
    "688": "Serbia",
    "690": "Seychelles",
    "694": "Sierra Leone",
    "699": "India",    # CEPII variant — same as 356
    "702": "Singapore",
    "703": "Slovakia",
    "705": "Slovenia",
    "706": "Somalia",
    "710": "South Africa",
    "711": "South Africa",  # CEPII SACU variant
    "716": "Zimbabwe",
    "724": "Spain",
    "728": "South Sudan",
    "729": "Sudan",
    "736": "Sudan (former)",
    "740": "Suriname",
    "748": "Eswatini",
    "752": "Sweden",
    "756": "Switzerland",
    "757": "Switzerland",  # CEPII variant
    "760": "Syria",
    "762": "Tajikistan",
    "764": "Thailand",
    "768": "Togo",
    "772": "Tokelau",
    "776": "Tonga",
    "780": "Trinidad and Tobago",
    "788": "Tunisia",
    "792": "Turkey",
    "795": "Turkmenistan",
    "796": "Turks and Caicos",
    "798": "Tuvalu",
    "800": "Uganda",
    "804": "Ukraine",
    "784": "United Arab Emirates",
    "826": "United Kingdom",
    "834": "Tanzania",
    "840": "USA",
    "854": "Burkina Faso",
    "858": "Uruguay",
    "860": "Uzbekistan",
    "862": "Venezuela",
    "882": "Samoa",
    "887": "Yemen",
    "891": "Serbia and Montenegro",
    "894": "Zambia",
    "807": "North Macedonia",
}

# Alias table: non-standard names observed in CEPII/Comtrade to canonical form
_NAME_ALIASES: Dict[str, str] = {
    # Common variants
    "United States": "USA",
    "United States of America": "USA",
    "U.S.A.": "USA",
    "US": "USA",
    "Korea, Republic of": "South Korea",
    "Korea, Rep.": "South Korea",
    "Dem. People's Rep. of Korea": "North Korea",
    "Viet Nam": "Vietnam",
    "Viet nam": "Vietnam",
    "Iran (Islamic Republic of)": "Iran",
    "Syrian Arab Republic": "Syria",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
    "Tanzania, United Republic of": "Tanzania",
    "Congo, the Democratic Republic of the": "DRC",
    "Congo, Democratic Republic of the": "DRC",
    "Dem. Rep. of the Congo": "DRC",
    "Lao People's Democratic Republic": "Laos",
    "Lao PDR": "Laos",
    "Czechia": "Czechia",
    "Czech Republic": "Czechia",
    "Russian Federation": "Russia",
    "Republic of Moldova": "Moldova",
    "Chinese Taipei": "Taiwan",
    "UAE": "United Arab Emirates",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Eswatini": "Eswatini",
    "Swaziland": "Eswatini",
    "North Macedonia": "North Macedonia",
    "Republic of North Macedonia": "North Macedonia",
    "Cabo Verde": "Cabo Verde",
    "Cape Verde": "Cabo Verde",
    "Brunei Darussalam": "Brunei",
    "Myanmar": "Myanmar",
    "Burma": "Myanmar",
}


def resolve_code(code: str) -> str:
    """
    Resolve a single country identifier to a canonical name.

    Handles:
      - ISO 3166-1 numeric codes (as strings: "699", "450")
      - Known name aliases
      - Already-canonical names (returned as-is)
    """
    s = str(code).strip()
    # Numeric code (zero-padded or not)
    if s.isdigit():
        padded = s.zfill(3)
        if padded in ISO_NUMERIC:
            return ISO_NUMERIC[padded]
        # Try without padding
        if s in ISO_NUMERIC:
            return ISO_NUMERIC[s]
        return f"Unknown-{s}"
    # Name alias
    if s in _NAME_ALIASES:
        return _NAME_ALIASES[s]
    return s


def normalize_country_names(df, exporter_col: str = "exporter", importer_col: str = "importer"):
    """
    Normalize country name columns in a CEPII/Comtrade DataFrame in-place.

    Converts all numeric ISO codes to standard country names and
    applies name aliases (e.g. "United States" → "USA").

    Args:
        df: DataFrame with bilateral trade data.
        exporter_col: Column containing exporter names/codes.
        importer_col: Column containing importer names/codes.

    Returns:
        The same DataFrame with normalized country columns.
    """
    if exporter_col in df.columns:
        df[exporter_col] = df[exporter_col].astype(str).map(resolve_code)
    if importer_col in df.columns:
        df[importer_col] = df[importer_col].astype(str).map(resolve_code)
    return df
