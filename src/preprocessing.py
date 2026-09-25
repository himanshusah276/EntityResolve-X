"""
Preprocessing module for Person 4 Entity Resolution Pipeline.

Performs conservative text, address, numeric, and country normalization.
Preserves raw values while extracting multi-scale token, n-gram, and numeric representations.
No external geocoding or external API calls (strictly within competition constraints).
"""
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd


# Common international legal suffixes (ordered longest to shortest for proper regex matching)
LEGAL_SUFFIXES = [
    "private limited",
    "pvt ltd",
    "pvt. ltd.",
    "pvt. ltd",
    "pvt limited",
    "public limited company",
    "plc",
    "limited liability company",
    "llc",
    "l.l.c.",
    "llp",
    "corporation",
    "corp",
    "corp.",
    "incorporated",
    "inc",
    "inc.",
    "limited",
    "ltd",
    "ltd.",
    "company",
    "co",
    "co.",
    "gmbh",
    "sarl",
    "s.a.r.l.",
    "s.a.",
    "sa",
    "b.v.",
    "bv",
    "holding",
    "holdings",
    "group",
]

# Regex pattern for legal suffix matching at end of string or standalone
_SUFFIX_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(s) for s in sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)

# Common address abbreviations
ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "rd.": "road",
    "st": "street",
    "st.": "street",
    "str": "street",
    "ave": "avenue",
    "ave.": "avenue",
    "blvd": "boulevard",
    "blvd.": "boulevard",
    "dr": "drive",
    "dr.": "drive",
    "ln": "lane",
    "ln.": "lane",
    "ct": "court",
    "ct.": "court",
    "pl": "place",
    "pl.": "place",
    "sq": "square",
    "sq.": "square",
    "pkwy": "parkway",
    "hwy": "highway",
    "ste": "suite",
    "ste.": "suite",
    "apt": "apartment",
    "apt.": "apartment",
    "fl": "floor",
    "fl.": "floor",
    "bldg": "building",
    "bldg.": "building",
    "opp": "opposite",
    "opp.": "opposite",
    "nr": "near",
    "nr.": "near",
}


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to ASCII/NFKD equivalent."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in normalized if not unicodedata.combining(c))


def clean_text(text: str) -> str:
    """
    Conservative string cleaning:
    - Lowercase
    - Unicode normalization
    - Replace punctuation with spaces while preserving alphanumeric characters
    - Collapse multiple spaces
    """
    if not text:
        return ""
    text = normalize_unicode(text).lower()
    # Replace non-alphanumeric (except standard whitespace) with space
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple whitespaces
    return re.sub(r"\s+", " ", text).strip()


def extract_numbers(text: str) -> List[str]:
    """Extract all individual numeric sequences from text."""
    if not text:
        return []
    return re.findall(r"\b\d+\b", text)


def generate_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generate character n-grams from cleaned text."""
    clean = clean_text(text).replace(" ", "")
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i : i + n] for i in range(len(clean) - n + 1)}


def tokenize_words(text: str, min_len: int = 2) -> List[str]:
    """Tokenize cleaned text into word tokens of minimum length."""
    clean = clean_text(text)
    if not clean:
        return []
    return [w for w in clean.split() if len(w) >= min_len]


def normalize_country(country: str) -> str:
    """
    Open-set country normalization:
    Preserves raw value, standardizes whitespace and case.
    Never hard-codes allowed country lists.
    """
    if not country:
        return "UNKNOWN"
    norm = normalize_unicode(country).strip().upper()
    return norm if norm else "UNKNOWN"


def extract_legal_suffixes(name: str) -> Tuple[str, List[str]]:
    """
    Detects legal business suffixes, extracts them, and returns (name_without_suffixes, found_suffixes).
    Conservative: does not remove tokens if it would leave the business name empty.
    """
    clean = clean_text(name)
    found: List[str] = []
    
    for match in _SUFFIX_PATTERN.finditer(clean):
        found.append(match.group(0).strip())

    # Safely strip suffixes from name
    stripped = _SUFFIX_PATTERN.sub(" ", clean)
    stripped = re.sub(r"\s+", " ", stripped).strip()

    # Conservative fallback: if stripping leaves nothing, keep original cleaned name
    final_name = stripped if stripped else clean
    return final_name, found


def normalize_address_tokens(address: str) -> List[str]:
    """
    Address tokenization with abbreviation expansion and component extraction.
    """
    clean = clean_text(address)
    tokens = clean.split()
    expanded_tokens: List[str] = []
    
    for t in tokens:
        expanded = ADDRESS_ABBREVIATIONS.get(t, t)
        if len(expanded) >= 2:
            expanded_tokens.append(expanded)
            
    return expanded_tokens


def extract_address_components(address: str) -> Dict[str, Any]:
    """
    Extracts structural address components without external geocoding:
    - numbers: list of all numeric tokens
    - postal_candidates: 4-6 digit tokens (PIN/Zip codes) or alphanumeric postal codes
    - street_number: first numeric token if found at start of address
    """
    clean = clean_text(address)
    numbers = extract_numbers(clean)
    
    # Postal candidates: 4-6 consecutive digits (covers US 5-digit, India 6-digit, Europe 4-5 digit)
    postal_candidates = [n for n in numbers if 4 <= len(n) <= 6]
    
    # Check for street number at start of address string
    street_num = None
    first_token_match = re.match(r"^(\d+)\b", clean)
    if first_token_match:
        street_num = first_token_match.group(1)

    return {
        "numbers": numbers,
        "postal_candidates": postal_candidates,
        "street_number": street_num,
    }


@dataclass
class ProcessedRecord:
    """Preprocessed representation of a single business entity record."""
    entity_id: str
    country_raw: str
    country_norm: str
    
    # Business name fields
    raw_name: str
    clean_name: str
    core_name: str  # Legal suffix removed
    legal_suffixes: List[str]
    name_tokens: List[str]
    name_ngrams: Set[str]
    name_numbers: List[str]
    
    # Business address fields
    raw_address: str
    clean_address: str
    address_tokens: List[str]
    address_ngrams: Set[str]
    address_numbers: List[str]
    postal_candidates: List[str]
    street_number: Optional[str]


def preprocess_record(record_dict: Dict[str, str]) -> ProcessedRecord:
    """Convert raw record dictionary into a ProcessedRecord instance."""
    eid = str(record_dict.get("entity_id", "")).strip()
    raw_name = str(record_dict.get("business_name", "")).strip()
    raw_addr = str(record_dict.get("business_address", "")).strip()
    raw_country = str(record_dict.get("country", "")).strip()

    country_norm = normalize_country(raw_country)
    clean_n = clean_text(raw_name)
    core_n, suffixes = extract_legal_suffixes(raw_name)
    name_toks = tokenize_words(core_n)
    name_ngrams = generate_char_ngrams(core_n, n=3)
    name_nums = extract_numbers(clean_n)

    clean_a = clean_text(raw_addr)
    addr_toks = normalize_address_tokens(raw_addr)
    addr_ngrams = generate_char_ngrams(clean_a, n=3)
    addr_comps = extract_address_components(raw_addr)

    return ProcessedRecord(
        entity_id=eid,
        country_raw=raw_country,
        country_norm=country_norm,
        raw_name=raw_name,
        clean_name=clean_n,
        core_name=core_n,
        legal_suffixes=suffixes,
        name_tokens=name_toks,
        name_ngrams=name_ngrams,
        name_numbers=name_nums,
        raw_address=raw_addr,
        clean_address=clean_a,
        address_tokens=addr_toks,
        address_ngrams=addr_ngrams,
        address_numbers=addr_comps["numbers"],
        postal_candidates=addr_comps["postal_candidates"],
        street_number=addr_comps["street_number"],
    )


def preprocess_dataframe(df: pd.DataFrame) -> List[ProcessedRecord]:
    """Process an entire DataFrame of records into a list of ProcessedRecord objects."""
    records: List[ProcessedRecord] = []
    # Dict conversion per row is fast and avoids pandas indexing overhead
    for r in df.to_dict(orient="records"):
        records.append(preprocess_record(r))
    return records
