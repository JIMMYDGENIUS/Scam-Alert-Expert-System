from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class SenderInfo(BaseModel):
    domain_age_days: Optional[int] = Field(default=None)
    confirmed_mule: Optional[bool] = Field(default=False)

class Reputation(BaseModel):
    reports_last_90d: Optional[int] = 0
    global_blacklist: Optional[bool] = False

class Event(BaseModel):
    text: Optional[str] = ""
    display_domain: Optional[str] = ""
    final_domain: Optional[str] = ""
    channel: Optional[str] = "unknown"  # sms, email, call, web, txn
    sender: Optional[SenderInfo] = SenderInfo()
    reputation: Optional[Reputation] = Reputation()
    metadata: Optional[Dict[str, Any]] = {}

class RuleHit(BaseModel):
    rule_id: str
    weight: float
    evidence: Dict[str, Any]

class DetectResponse(BaseModel):
    score: float
    tier: str
    rule_hits: List[RuleHit]
    actions: List[str]
    summary: str
