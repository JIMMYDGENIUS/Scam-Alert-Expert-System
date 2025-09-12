import re
import textdistance
from typing import List, Set

SCAM_TERMS = {"otp", "one-time password", "seed", "seed phrase", "recovery phrase", "private key"}
URGENCY = {"urgent", "immediately", "within 10 minutes", "within 5 minutes"}
SECRECY = {"don't tell", "confidential", "keep this between us"}

def contains_any(text: str, terms: Set[str]) -> List[str]:
    t = (text or "").lower()
    return [term for term in terms if term.lower() in t]

def lookalike_score(a: str, b: str) -> float:
    # Use Jaro-Winkler similarity as a reasonable proxy
    if not a or not b:
        return 0.0
    return float(textdistance.jaro_winkler(a.lower(), b.lower()))

def regex_match(text: str, pattern: str) -> bool:
    if not text:
        return False
    return re.search(pattern, text) is not None
