import os
import pickle
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from .models import Event, DetectResponse, RuleHit
from .rules import RuleEngine, diminishing_sum, map_to_tier, blend_scores

# ------------------------------------------------------------------
#  Configuration
# ------------------------------------------------------------------
RULES_PATH = os.getenv(
    "RULES_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "rules", "rules.yaml")
)
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "model.pkl")
)
ALPHA = float(os.getenv("BLEND_ALPHA", "0.7"))  # expert weight

app = FastAPI(title="Scam Expert System", version="1.0.0")

# Mount /static for optional images/js/css
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.parent / "static"),
    name="static",
)

# ------------------------------------------------------------------
#  Templates
# ------------------------------------------------------------------
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")

# ------------------------------------------------------------------
#  Friendly action descriptions
# ------------------------------------------------------------------
ACTION_DESCRIPTIONS = {
    "allow": "Allow the message — No risk detected.",
    "warn_user": "Show a simple warning to the user.",
    "log": "Log the event for monitoring.",
    "strong_warn": "Display a strong fraud warning to the user.",
    "limit_actions": "Limit account actions until the user is verified.",
    "request_verification": "Ask the user to verify their identity.",
    "block": "Block the message or transaction completely.",
    "escalate_manual_review": "Send the case to a human reviewer for manual inspection.",
}

# ------------------------------------------------------------------
#  Core Engine + ML Model
# ------------------------------------------------------------------
engine = RuleEngine(RULES_PATH)
_ml_model = None
_ml_ready = False

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            _ml_model = pickle.load(f)
            _ml_ready = True
    except Exception:
        _ml_ready = False

# ------------------------------------------------------------------
#  UI Routes
# ------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Render the landing page with a textarea form
    for message analysis.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_message(
    request: Request,
    message: str = Form(...)
):
    """
    Handle form submissions from the UI.
    Uses the same detect logic as the /detect API,
    but returns an HTML page with the result.
    """
    event = {"text": message}
    hits, hard_stop = engine.apply(event)
    expert_score = diminishing_sum([h["weight"] for h in hits])

    score = 100.0 if hard_stop else expert_score
    ml = ml_score(event) if not hard_stop else 100.0
    if _ml_ready and not hard_stop:
        score = blend_scores(expert_score, ml, ALPHA)

    tier = "T3" if hard_stop else map_to_tier(score)

    # internal action codes
    action_codes = {
        "T0": ["allow"],
        "T1": ["warn_user", "log"],
        "T2": ["strong_warn", "limit_actions", "request_verification"],
        "T3": ["block", "escalate_manual_review"],
    }[tier]

    # convert to human-friendly descriptions
    friendly_actions = [ACTION_DESCRIPTIONS.get(a, a) for a in action_codes]

    summary = f"Score {score:.1f} → {tier}. Expert={expert_score:.1f}" + (
        f", ML={ml:.1f}" if _ml_ready else ""
    )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": {
                "score": round(score, 1),
                "tier": tier,
                "actions": friendly_actions,
                "summary": summary,
                "hits": hits[:5],
            },
        },
    )

# ------------------------------------------------------------------
#  API Routes
# ------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True, "ml_loaded": _ml_ready}


@app.get("/rules")
def get_rules():
    return engine.rules


@app.post("/rules/reload")
def reload_rules():
    engine.load_rules()
    return {"reloaded": True, "count": len(engine.rules)}


@app.post("/detect", response_model=DetectResponse)
def detect(event: Event):
    e = event.model_dump()
    hits, hard_stop = engine.apply(e)
    expert_score = diminishing_sum([h["weight"] for h in hits])

    score = 100.0 if hard_stop else expert_score
    ml = ml_score(e) if not hard_stop else 100.0
    if _ml_ready and not hard_stop:
        score = blend_scores(expert_score, ml, ALPHA)

    tier = "T3" if hard_stop else map_to_tier(score)
    action_codes = {
        "T0": ["allow"],
        "T1": ["warn_user", "log"],
        "T2": ["strong_warn", "limit_actions", "request_verification"],
        "T3": ["block", "escalate_manual_review"],
    }[tier]

    # keep API response with internal codes (not the friendly text)
    summary = f"Score {score:.1f} → {tier}. Expert={expert_score:.1f}" + (
        f", ML={ml:.1f}" if _ml_ready else ""
    )

    return DetectResponse(
        score=round(score, 1),
        tier=tier,
        rule_hits=[RuleHit(**h) for h in hits[:5]],
        actions=action_codes,
        summary=summary,
    )

# ------------------------------------------------------------------
#  Helper Functions
# ------------------------------------------------------------------
def featurize_for_ml(event: Dict[str, Any]) -> Dict[str, Any]:
    text = (event.get("text") or "").lower()
    return {
        "len_text": len(text),
        "has_otp": int("otp" in text or "one-time password" in text),
        "has_seed": int("seed phrase" in text or "private key" in text or "recovery phrase" in text),
        "has_urgent": int("urgent" in text or "immediately" in text),
        "url_mismatch": int((event.get("display_domain") or "") != (event.get("final_domain") or "")),
        "domain_age": int(event.get("sender", {}).get("domain_age_days") or 9999),
        "reports": int(event.get("reputation", {}).get("reports_last_90d") or 0),
        "blacklisted": int(bool(event.get("reputation", {}).get("global_blacklist", False))),
    }


def ml_score(event: Dict[str, Any]) -> float:
    if not _ml_ready or _ml_model is None:
        return 0.0
    feats = featurize_for_ml(event)
    xs = [[feats[k] for k in _ml_model["feature_order"]]]
    p = _ml_model["clf"].predict_proba(xs)[0][1]
    return float(p) * 100.0
