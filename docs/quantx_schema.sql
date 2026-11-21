-- =====================================================================
-- QUANT-X research database - PostgreSQL 16 + TimescaleDB (target schema)
-- Phase 1 runs the same core tables on SQLite (quantx/data/store.py).
-- Conventions: all timestamps TIMESTAMPTZ in UTC = bar OPEN time; prices DOUBLE PRECISION;
-- every derived row points back to the ingestion/analysis run that produced it.
-- =====================================================================
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ---------- reference data -------------------------------------------
CREATE TABLE data_source (
  source_id      SMALLSERIAL PRIMARY KEY,
  name           TEXT UNIQUE NOT NULL,          -- 'mt5:MetaQuotes-Demo', 'binance', 'fred', 'sec_edgar'
  kind           TEXT NOT NULL,                 -- broker | exchange | aggregator | official
  terms_url      TEXT,                          -- licence / terms of use reviewed
  realtime       BOOLEAN NOT NULL DEFAULT FALSE,
  notes          TEXT
);

CREATE TABLE instrument (
  instrument_id  SERIAL PRIMARY KEY,
  symbol         TEXT NOT NULL,                 -- as quoted by the source (EURUSD, BTCUSDT, AAPL)
  source_id      SMALLINT NOT NULL REFERENCES data_source,
  asset_class    TEXT NOT NULL,                 -- forex | crypto | equity | etf | commodity | metal | index | rates
  base_ccy       TEXT, quote_ccy TEXT,
  calendar       TEXT NOT NULL,                 -- fx (24x5) | 24x7 | exchange:<MIC>
  price_type     TEXT NOT NULL,                 -- bid | mid | last | settlement
  volume_type    TEXT,                          -- 'broker tick count' | 'exchange base volume' | ...
  tick_size      DOUBLE PRECISION, contract_size DOUBLE PRECISION,
  listed_from    DATE, delisted_on DATE,        -- survivorship: keep delisted instruments
  UNIQUE (symbol, source_id)
);

-- ---------- provenance ------------------------------------------------
CREATE TABLE ingestion_run (
  run_id         UUID PRIMARY KEY,
  source_id      SMALLINT REFERENCES data_source,
  instrument_id  INT REFERENCES instrument,
  timeframe      TEXT NOT NULL,
  retrieved_at   TIMESTAMPTZ NOT NULL,
  server_utc_offset_h REAL,                     -- broker server time conversion applied
  rows           INT, first_ts TIMESTAMPTZ, last_ts TIMESTAMPTZ,
  data_sha256    CHAR(16),
  quality        REAL CHECK (quality BETWEEN 0 AND 1),
  warnings       JSONB,
  code_version   TEXT
);

-- ---------- market data (hypertables) ---------------------------------
CREATE TABLE ohlcv (
  instrument_id  INT NOT NULL REFERENCES instrument,
  timeframe      TEXT NOT NULL,                 -- M1..W1
  ts             TIMESTAMPTZ NOT NULL,
  open DOUBLE PRECISION NOT NULL, high DOUBLE PRECISION NOT NULL,
  low  DOUBLE PRECISION NOT NULL, close DOUBLE PRECISION NOT NULL,
  volume         DOUBLE PRECISION,
  spread_points  REAL,
  ingestion_run  UUID REFERENCES ingestion_run,
  PRIMARY KEY (instrument_id, timeframe, ts),
  CHECK (high >= GREATEST(open, close) AND low <= LEAST(open, close) AND low > 0)
);
SELECT create_hypertable('ohlcv', 'ts', chunk_time_interval => INTERVAL '30 days');

CREATE TABLE quote_tick (                       -- Phase 2+: microstructure research
  instrument_id  INT NOT NULL REFERENCES instrument,
  ts             TIMESTAMPTZ NOT NULL,
  bid DOUBLE PRECISION, ask DOUBLE PRECISION, bid_size DOUBLE PRECISION, ask_size DOUBLE PRECISION,
  ingestion_run  UUID REFERENCES ingestion_run
);
SELECT create_hypertable('quote_tick', 'ts', chunk_time_interval => INTERVAL '1 day');

CREATE TABLE derivatives_metric (               -- crypto perps / futures
  instrument_id  INT NOT NULL REFERENCES instrument,
  ts             TIMESTAMPTZ NOT NULL,
  funding_rate DOUBLE PRECISION, open_interest DOUBLE PRECISION, mark_price DOUBLE PRECISION,
  long_liquidations DOUBLE PRECISION, short_liquidations DOUBLE PRECISION,
  PRIMARY KEY (instrument_id, ts)
);
SELECT create_hypertable('derivatives_metric', 'ts');

CREATE TABLE macro_series (                     -- FRED / statistics offices; vintage-aware (no look-ahead)
  series_code    TEXT NOT NULL,                 -- e.g. FRED:CPIAUCSL
  period_end     DATE NOT NULL,
  value          DOUBLE PRECISION,
  released_at    TIMESTAMPTZ NOT NULL,          -- when this value became public
  vintage        DATE NOT NULL,                 -- revisions stored as separate vintages
  PRIMARY KEY (series_code, period_end, vintage)
);

CREATE TABLE economic_event (
  event_id       BIGSERIAL PRIMARY KEY,
  scheduled_at   TIMESTAMPTZ NOT NULL, currency TEXT, title TEXT, impact TEXT,
  forecast TEXT, previous TEXT, actual TEXT, actual_published_at TIMESTAMPTZ, source TEXT
);

-- ---------- research artefacts ----------------------------------------
CREATE TABLE feature_value (                    -- point-in-time feature store
  instrument_id  INT NOT NULL REFERENCES instrument,
  timeframe      TEXT NOT NULL,
  ts             TIMESTAMPTZ NOT NULL,          -- value known at ts (computed from data <= ts)
  feature        TEXT NOT NULL,                 -- 'rsi14', 'garch_vol', ...
  value          DOUBLE PRECISION,
  calc_version   TEXT NOT NULL,
  PRIMARY KEY (instrument_id, timeframe, ts, feature, calc_version)
);
SELECT create_hypertable('feature_value', 'ts');

CREATE TABLE experiment (                       -- every hypothesis / model / strategy trial is logged
  experiment_id  UUID PRIMARY KEY,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  kind           TEXT NOT NULL,                 -- analysis | model | strategy | portfolio
  hypothesis     TEXT,                          -- pre-registered before seeing test data
  config         JSONB NOT NULL,                -- params, data ids, splits, costs, seeds
  code_version   TEXT NOT NULL,
  parent_id      UUID REFERENCES experiment,    -- variants of the same idea (trial counting)
  status         TEXT NOT NULL                  -- planned | running | done | failed
);

CREATE TABLE experiment_result (
  experiment_id  UUID REFERENCES experiment,
  split          TEXT NOT NULL,                 -- train | validation | test | walk_forward_k
  metric         TEXT NOT NULL,                 -- sharpe, max_dd, mae, ...
  value          DOUBLE PRECISION,
  ci_low DOUBLE PRECISION, ci_high DOUBLE PRECISION,
  PRIMARY KEY (experiment_id, split, metric)
);

CREATE TABLE analysis_run (
  run_id         UUID PRIMARY KEY,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  kind           TEXT, request JSONB, data_ids JSONB, code_version TEXT, result JSONB
);

CREATE TABLE paper_order (                      -- Phase 8: paper trading only
  order_id       UUID PRIMARY KEY,
  experiment_id  UUID REFERENCES experiment,
  instrument_id  INT REFERENCES instrument,
  created_at     TIMESTAMPTZ NOT NULL, side TEXT, qty DOUBLE PRECISION,
  intended_price DOUBLE PRECISION, fill_price DOUBLE PRECISION, filled_at TIMESTAMPTZ,
  costs DOUBLE PRECISION, approved_by TEXT       -- human approver, required
);
