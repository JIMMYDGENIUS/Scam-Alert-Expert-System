import math, yaml
from typing import Dict, Any, List, Tuple
from .feature_extractors import contains_any, lookalike_score, regex_match

# Tier boundaries
TIERS: List[Tuple[str, int, int]] = [
    ("T0", 0, 24),
    ("T1", 25, 49),
    ("T2", 50, 79),
    ("T3", 80, 100),
]

def map_to_tier(score: float) -> str:
    for name, lo, hi in TIERS:
        if lo <= score <= hi:
            return name
    return "T3" if score > 100 else "T0"

def diminishing_sum(weights: List[float]) -> float:
    total = sum(weights)
    return 100.0 * (1.0 - math.exp(-total / 100.0))

class RuleEngine:
    def __init__(self, rules_path: str):
        self.rules_path = rules_path
        self.rules = []
        self.load_rules()

    def load_rules(self):
        with open(self.rules_path, "r") as f:
            self.rules = yaml.safe_load(f) or []

    def eval_conditions(self, event: Dict[str, Any], conds: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        evidence = {}
        if "any" in conds:
            for c in conds["any"]:
                ok, ev = self.eval_condition(event, c)
                if ok:
                    evidence.update(ev)
                    return True, evidence
            return False, {}
        if "all" in conds:
            ev_all = {}
            for c in conds["all"]:
                ok, ev = self.eval_condition(event, c)
                if not ok:
                    return False, {}
                ev_all.update(ev)
            return True, ev_all
        # single condition object
        return self.eval_condition(event, conds)

    def eval_condition(self, event: Dict[str, Any], cond: Dict[str, Any]):
        # Supported primitives (mapped to YAML keys)
        text = (event.get("text") or "")
        display = (event.get("display_domain") or "")
        final = (event.get("final_domain") or "")
        sender = event.get("sender", {}) or {}
        reputation = event.get("reputation", {}) or {}

        if "text.contains_any" in cond:
            hits = contains_any(text, set(cond["text.contains_any"]))
            return (len(hits) > 0, {"matched_terms": hits} if hits else {})

        if "text.regex" in cond:
            pat = cond["text.regex"]
            import re
            ok = re.search(pat, text or "") is not None
            return ok, {"regex": pat} if ok else ({}, {})

        if "url.display_domain_neq_final" in cond:
            ok = bool(display and final and (display != final))
            return ok, {"display_domain": display, "final_domain": final} if ok else ({}, {})

        if "url.lookalike_threshold" in cond:
            thr = float(cond["url.lookalike_threshold"])
            score = lookalike_score(display, final)
            ok = score >= thr
            return ok, {"lookalike_score": round(score, 2)} if ok else ({}, {})

        if "sender.domain_age_lt_days" in cond:
            days = sender.get("domain_age_days")
            ok = days is not None and days < int(cond["sender.domain_age_lt_days"])
            return ok, {"domain_age_days": days} if ok else ({}, {})

        if "reputation.reports_last_90d_gte" in cond:
            ok = (reputation.get("reports_last_90d") or 0) >= int(cond["reputation.reports_last_90d_gte"])
            return ok, {"reports_last_90d": reputation.get("reports_last_90d", 0)} if ok else ({}, {})

        if "reputation.global_blacklist" in cond:
            val = bool(cond["reputation.global_blacklist"])
            ok = bool(reputation.get("global_blacklist", False)) == val
            return ok, {"global_blacklist": reputation.get("global_blacklist", False)} if ok else ({}, {})

        if "sender.confirmed_mule" in cond:
            val = bool(cond["sender.confirmed_mule"])
            ok = bool(sender.get("confirmed_mule", False)) == val
            return ok, {"confirmed_mule": sender.get("confirmed_mule", False)} if ok else ({}, {})

        return False, {}

    def apply(self, event: Dict[str, Any]):
        hits = []
        hard_stop = False
        for r in self.rules:
            ok, ev = self.eval_conditions(event, r.get("conditions", {}))
            if ok:
                rh = {"rule_id": r["id"], "weight": float(r.get("weight", 0)), "evidence": ev}
                hits.append(rh)
                if r.get("hard_stop", False):
                    hard_stop = True
        hits = sorted(hits, key=lambda x: x["weight"], reverse=True)
        return hits, hard_stop

def blend_scores(expert: float, ml: float, alpha: float = 0.7) -> float:
    return alpha * expert + (1 - alpha) * ml
