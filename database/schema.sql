-- PostgreSQL schema for Scam Expert System
CREATE TABLE IF NOT EXISTS events (
  event_id BIGSERIAL PRIMARY KEY,
  channel TEXT,
  text TEXT,
  display_domain TEXT,
  final_domain TEXT,
  sender JSONB,
  reputation JSONB,
  metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS verdicts (
  verdict_id BIGSERIAL PRIMARY KEY,
  event_id BIGINT REFERENCES events(event_id) ON DELETE CASCADE,
  score NUMERIC(5,2),
  tier TEXT,
  explanation_json JSONB,
  actions JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ruleset_versions (
  version_id BIGSERIAL PRIMARY KEY,
  rules_yaml TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS feedback (
  feedback_id BIGSERIAL PRIMARY KEY,
  event_id BIGINT REFERENCES events(event_id) ON DELETE CASCADE,
  label TEXT CHECK (label IN ('scam','legit')),
  reviewer TEXT,
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at);
CREATE INDEX IF NOT EXISTS idx_verdicts_tier ON verdicts(tier);
CREATE INDEX IF NOT EXISTS idx_feedback_label ON feedback(label);
