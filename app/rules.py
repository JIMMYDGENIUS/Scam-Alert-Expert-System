import math, yaml, re
from typing import Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
from .feature_extractors import contains_any, lookalike_score, regex_match

# Default tier boundaries
TIERS: List[Tuple[str, int, int]] = [
    ("T0", 0, 24),
    ("T1", 25, 49),
    ("T2", 50, 79),
    ("T3", 80, 100),
]

def map_to_tier(score: float, tiers: List[Tuple[str, int, int]] = TIERS) -> str:
    for name, lo, hi in tiers:
        if lo <= score <= hi:
            return name
    return "T3" if score > 100 else "T0"

def diminishing_sum(weights: List[float], cap: bool = True) -> float:
    """Combine rule weights with diminishing returns, capped at 100 by default."""
    total = sum(weights)
    score = 100.0 * (1.0 - math.exp(-total / 100.0))
    return min(score, 100.0) if cap else score

@dataclass
class RuleHit:
    rule_id: str
    weight: float
    evidence: Dict[str, Any]

class RuleEngine:
    def __init__(self, rules_path: str):
        self.rules_path = rules_path
        self.rules: List[Dict[str, Any]] = []
        self.condition_handlers: Dict[str, Callable[[Any, Dict[str, Any]], Tuple[bool, Dict[str, Any]]]] = {
            "text.contains_any": self._cond_contains_any,
            "text.regex": self._cond_regex,
            "url.display_domain_neq_final": self._cond_domain_mismatch,
            "url.lookalike_threshold": self._cond_lookalike,
            "sender.domain_age_lt_days": self._cond_domain_age,
            "reputation.reports_last_90d_gte": self._cond_reports,
            "reputation.global_blacklist": self._cond_blacklist,
            "sender.confirmed_mule": self._cond_mule,
        }
        self.load_rules()

    def load_rules(self):
        try:
            with open(self.rules_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            # ✅ Only keep the rules list
            self.rules = data.get("rules", [])
        except Exception as e:
            raise RuntimeError(f"Failed to load rules from {self.rules_path}: {e}")

    # ---- Condition primitives ----
    def _cond_contains_any(self, text: str, values: List[str]):
        hits = contains_any(text, set(values))
        return (bool(hits), {"matched_terms": hits} if hits else {})

    def _cond_regex(self, text: str, pattern: str):
        ok = re.search(pattern, text or "") is not None
        return (ok, {"regex": pattern} if ok else {})

    def _cond_domain_mismatch(self, event: Dict[str, Any], _):
        display, final = event.get("display_domain"), event.get("final_domain")
        ok = bool(display and final and display != final)
        return (ok, {"display_domain": display, "final_domain": final} if ok else {})

    def _cond_lookalike(self, event: Dict[str, Any], threshold: float):
        display, final = event.get("display_domain"), event.get("final_domain")
        score = lookalike_score(display, final)
        ok = score >= float(threshold)
        return (ok, {"lookalike_score": round(score, 2)} if ok else {})

    def _cond_domain_age(self, event: Dict[str, Any], max_days: int):
        days = (event.get("sender") or {}).get("domain_age_days")
        ok = days is not None and days < int(max_days)
        return (ok, {"domain_age_days": days} if ok else {})

    def _cond_reports(self, event: Dict[str, Any], threshold: int):
        rep = event.get("reputation", {}) or {}
        count = rep.get("reports_last_90d", 0)
        ok = count >= int(threshold)
        return (ok, {"reports_last_90d": count} if ok else {})

    def _cond_blacklist(self, event: Dict[str, Any], expected: bool):
        rep = event.get("reputation", {}) or {}
        actual = rep.get("global_blacklist", False)
        ok = actual == bool(expected)
        return (ok, {"global_blacklist": actual} if ok else {})

    def _cond_mule(self, event: Dict[str, Any], expected: bool):
        sender = event.get("sender", {}) or {}
        actual = sender.get("confirmed_mule", False)
        ok = actual == bool(expected)
        return (ok, {"confirmed_mule": actual} if ok else {})

    # ---- Evaluation ----
    def eval_conditions(self, event: Dict[str, Any], conds: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        if "any" in conds:
            for c in conds["any"]:
                ok, ev = self.eval_condition(event, c)
                if ok:
                    return True, ev
            return False, {}
        if "all" in conds:
            combined = {}
            for c in conds["all"]:
                ok, ev = self.eval_condition(event, c)
                if not ok:
                    return False, {}
                combined.update(ev)
            return True, combined
        return self.eval_condition(event, conds)

    def eval_condition(self, event: Dict[str, Any], cond: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        text = event.get("text") or ""
        for key, val in cond.items():
            if key in self.condition_handlers:
                handler = self.condition_handlers[key]
                # some handlers need text, others need event
                if key.startswith("text."):
                    return handler(text, val)
                return handler(event, val)
        return False, {}

    def apply(self, event: Dict[str, Any]) -> Dict[str, Any]:
        hits: List[RuleHit] = []
        hard_stop = False
        for r in self.rules:
            if not isinstance(r, dict):  # ✅ skip bad entries
                continue
            ok, ev = self.eval_conditions(event, r.get("conditions", {}))
            if ok:
                rh = RuleHit(
                    rule_id=r["id"],
                    weight=float(r.get("weight", 0)),
                    evidence=ev
                )
                hits.append(rh)
                if r.get("hard_stop", False):
                    hard_stop = True

        weights = [h.weight for h in hits]
        expert_score = diminishing_sum(weights)
        tier = map_to_tier(expert_score)

        return {
            "hits": [h.__dict__ for h in hits],
            "hard_stop": hard_stop,
            "score": expert_score,
            "tier": tier,
        }

def blend_scores(expert: float, ml: float, alpha: float = 0.7) -> float:
    return alpha * expert + (1 - alpha) * ml
