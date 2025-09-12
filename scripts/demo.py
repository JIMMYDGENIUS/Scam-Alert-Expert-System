import json, os
from app.rules import RuleEngine, diminishing_sum, map_to_tier
from app.main import featurize_for_ml

BASE = os.path.dirname(__file__)
rules_path = os.path.join(BASE, "rules", "rules.yaml")
engine = RuleEngine(rules_path)

sample = {
    "text": "URGENT: send your OTP within 5 minutes and keep this confidential.",
    "display_domain": "support.paypai.com",
    "final_domain": "support.paypal.com",
    "sender": {"domain_age_days": 12},
    "reputation": {"reports_last_90d": 4, "global_blacklist": False}
}

hits, hard_stop = engine.apply(sample)
expert = diminishing_sum([h["weight"] for h in hits])
print("HITS:", hits)
print("EXPERT SCORE:", expert, "TIER:", map_to_tier(expert))
print("ML FEATURES:", featurize_for_ml(sample))
