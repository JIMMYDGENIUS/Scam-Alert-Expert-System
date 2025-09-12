from app.rules import RuleEngine, diminishing_sum, map_to_tier
import os

def test_rule_load():
    engine = RuleEngine(os.path.join(os.path.dirname(__file__), "..", "rules", "rules.yaml"))
    assert len(engine.rules) > 0

def test_diminishing_sum():
    s1 = diminishing_sum([45])
    s2 = diminishing_sum([45, 30])
    assert s2 > s1 and s2 <= 100.0

def test_tier_mapping():
    assert map_to_tier(10) == "T0"
    assert map_to_tier(30) == "T1"
    assert map_to_tier(60) == "T2"
    assert map_to_tier(90) == "T3"
