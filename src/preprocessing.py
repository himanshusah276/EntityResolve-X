"""
Text preprocessing and domain normalization for business names, addresses, and countries.
"""
import re
import unicodedata
from typing import List, Set, Tuple


# Common English and French legal entity suffixes / stopwords
LEGAL_SUFFIXES = {
    # English
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited", 
    "pvt", "private", "llp", "co", "company", "enterprises", "enterprise",
    "holdings", "holding", "group", "services", "solutions", "technologies",
    "associates", "consulting", "industries", "trust",
    # French
    "sarl", "sasu", "sas", "eurl", "sa", "snc", "fils", "cie", "ets",
}

# Common address abbreviations mapping
ADDRESS_ABBREVIATIONS = {
    "st": "street",
    "saint": "street",
    "str": "street",
    "rd": "road",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "ter": "terrace",
    "terr": "terrace",
    "pkwy": "parkway",
    "cir": "circle",
    "fl": "floor",
    "ste": "suite",
    "apt": "apartment",
    "bldg": "building",
    "hwy": "highway",
    "rt": "route",
    "rte": "route",
    "no": "number",
    "kh": "khasra",
    "near": "nr",
}

STOPWORDS = {
    "the", "and", "of", "in", "at", "for", "on", "a", "an", "de", "du", "des",
    "et", "la", "le", "les", "d", "l", "sur"
}


def strip_accents(text: str) -> str:
    """Normalize unicode characters and remove diacritics/accents."""
    if not text:
        return ""
    nfkd_form = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def clean_text(text: str) -> str:
    """Basic lowercasing, accent stripping, and symbol normalization."""
    if not text:
        return ""
    text = strip_accents(text.lower())
    # Replace & with and
    text = re.sub(r"&", " and ", text)
    # Remove web prefixes/domains if present
    text = re.sub(r"https?://|www\.", "", text)
    # Replace punctuation and special characters with spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse multiple whitespaces
    return " ".join(text.split())


def normalize_business_name(name: str) -> Tuple[str, str, List[str]]:
    """
    Returns:
      (normalized_name, name_without_legal, tokens)
    """
    cleaned = clean_text(name)
    tokens = cleaned.split()
    
    # Filter legal suffixes and stopwords for core name representation
    core_tokens = [t for t in tokens if t not in LEGAL_SUFFIXES and t not in STOPWORDS]
    if not core_tokens:
        core_tokens = tokens  # fallback if all were filtered
        
    name_without_legal = " ".join(core_tokens)
    return cleaned, name_without_legal, core_tokens


def extract_numbers(text: str) -> Set[str]:
    """Extract numeric sequences from text (house numbers, pins, zipcodes)."""
    if not text:
        return set()
    cleaned = clean_text(text)
    # Extract digit sequences of length 1 or more
    numbers = set(re.findall(r"\b\d+\b", cleaned))
    # Normalize numbers with leading zeros (e.g. 0684 -> 684)
    normalized_numbers = set()
    for n in numbers:
        normalized_numbers.add(n.lstrip("0") or "0")
    return normalized_numbers


def normalize_address(address: str) -> Tuple[str, List[str], Set[str]]:
    """
    Returns:
      (normalized_address, address_tokens, numbers)
    """
    cleaned = clean_text(address)
    tokens = cleaned.split()
    
    # Expand or normalize street abbreviations
    expanded_tokens = []
    for t in tokens:
        if t in STOPWORDS or t == "null":
            continue
        mapped = ADDRESS_ABBREVIATIONS.get(t, t)
        expanded_tokens.append(mapped)
        
    normalized_addr = " ".join(expanded_tokens)
    numbers = extract_numbers(address)
    return normalized_addr, expanded_tokens, numbers


def normalize_country(country: str) -> str:
    """Normalize country string to exact uppercase trimmed format."""
    if not country:
        return "UNKNOWN"
    return country.strip().upper()
