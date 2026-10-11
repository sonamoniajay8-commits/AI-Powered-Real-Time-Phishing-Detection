
import ipaddress
import re
from urllib.parse import urlsplit

import pandas as pd


FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "num_dots",
    "num_hyphens",
    "num_digits",
    "num_at_symbols",
    "num_question_marks",
    "num_equals",
    "num_percent_symbols",
    "num_subdomains",
    "has_https",
    "has_ip_address",
    "has_suspicious_keyword",
    "has_shortener",
]

SUSPICIOUS_KEYWORDS = [
    "login",
    "verify",
    "account",
    "secure",
    "update",
    "password",
    "signin",
    "banking",
    "wallet",
    "confirm",
]

SHORTENER_DOMAINS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "is.gd",
    "cutt.ly",
    "shorturl.at",
}


def extract_features(url):
    """Convert one URL into a dictionary of numeric features."""

    if not isinstance(url, str) or not url.strip():
        raise ValueError("Please enter a non-empty URL.")

    url = url.strip()

    if any(ord(char) < 32 or ord(char) == 127 for char in url):
        raise ValueError("URL contains invalid control characters.")

    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        url = "https://" + url

    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname or ""
    except ValueError as exc:
        raise ValueError("Invalid URL format.") from exc

    if not hostname or "." not in hostname:
        raise ValueError("Enter a URL with a valid-looking domain.")

    hostname = hostname.lower().rstrip(".")

    try:
        ipaddress.ip_address(hostname)
        has_ip = 1
    except ValueError:
        has_ip = 0

    domain_parts = hostname.split(".")

    # Exclude the final two labels as an approximate
    # top-level domain and registered-domain boundary.
    # This is only a simple heuristic.
    num_subdomains = max(0, len(domain_parts) - 2)

    url_lower = url.lower()

    features = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": sum(char.isdigit() for char in url),
        "num_at_symbols": url.count("@"),
        "num_question_marks": url.count("?"),
        "num_equals": url.count("="),
        "num_percent_symbols": url.count("%"),
        "num_subdomains": num_subdomains,
        "has_https": int(parsed.scheme.lower() == "https"),
        "has_ip_address": has_ip,
        "has_suspicious_keyword": int(
            any(word in url_lower for word in SUSPICIOUS_KEYWORDS)
        ),
        "has_shortener": int(hostname in SHORTENER_DOMAINS),
    }

    return features


def extract_many(urls):
    """Create a feature table for a list of URLs."""

    rows = [extract_features(url) for url in urls]

    return pd.DataFrame(rows, columns=FEATURE_NAMES)
