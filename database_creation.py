import duckdb
from pathlib import Path


def create_database(DB_PATH):
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(DB_PATH)
    #completion_time
    con.execute("""
    CREATE TABLE IF NOT EXISTS runs (
    run_id                              VARCHAR(26) PRIMARY KEY,
    experiment_name                     VARCHAR(64) NOT NULL,
    experiment_type                     VARCHAR,
    creation_time                       TIMESTAMP NOT NULL,
    completion_time                     TIMESTAMP,
    run_notes                           VARCHAR,
    n_rounds                            INTEGER NOT NULL,
    fundamental_source                  VARCHAR NOT NULL,
    run_status                          VARCHAR NOT NULL,
    run_progress                        DOUBLE NOT NULL DEFAULT 0.0,
    py_vers                             VARCHAR NOT NULL,
    code_vers                           VARCHAR NOT NULL,
    n_agents                            INTEGER NOT NULL,
    total_starting_cash                 DOUBLE NOT NULL,
    total_starting_shares               BIGINT NOT NULL,
    market_mechanism                    VARCHAR NOT NULL,
    pricing_rule                        VARCHAR NOT NULL,
    rationing_rule                      VARCHAR NOT NULL,
    tie_break_rule                      VARCHAR NOT NULL,
    transaction_cost_rate               DOUBLE NOT NULL,
    noise_parameter_distribution_type   VARCHAR NOT NULL,
    distribution_data                   JSON,
    signal_generator_noise_distribution VARCHAR NOT NULL,
    bias                                FLOAT NOT NULL                
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS gbm_config (
    run_id                        VARCHAR(26) PRIMARY KEY,
    s0                            DOUBLE NOT NULL,
    drift                         DOUBLE NOT NULL,
    volatility                    DOUBLE NOT NULL,
    seed                          VARCHAR(64),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS fundamental_series (
    run_id       VARCHAR(26) NOT NULL,
    round_number INTEGER NOT NULL,
    price        DOUBLE NOT NULL,

    PRIMARY KEY (run_id, round_number),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS market_round (
    run_id           VARCHAR(26) NOT NULL,
    round_number     INTEGER NOT NULL,

    p_t              DOUBLE,
    best_bid         DOUBLE,
    best_ask         DOUBLE,
    volume           DOUBLE NOT NULL DEFAULT 0,
    n_trades         BIGINT NOT NULL DEFAULT 0,

    demand_at_p      DOUBLE NOT NULL DEFAULT 0,
    supply_at_p      DOUBLE NOT NULL DEFAULT 0,

    n_active_buyers  INTEGER NOT NULL DEFAULT 0,
    n_active_sellers INTEGER NOT NULL DEFAULT 0,
    n_active_total   INTEGER NOT NULL DEFAULT 0,

    bid_depth_total  DOUBLE NOT NULL DEFAULT 0,
    ask_depth_total  DOUBLE NOT NULL DEFAULT 0,

    price_levels_bid INTEGER NOT NULL DEFAULT 0,
    price_levels_ask INTEGER NOT NULL DEFAULT 0,

    PRIMARY KEY (run_id, round_number),

    FOREIGN KEY (run_id) REFERENCES runs(run_id),

    FOREIGN KEY (run_id, round_number)
        REFERENCES fundamental_series(run_id, round_number)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS agent_population(
    run_id                          VARCHAR(26) NOT NULL,
    agent_id                        INTEGER NOT NULL,
    strategy_type                   VARCHAR NOT NULL,
    noise_parameter                 DOUBLE,
    group_label                     VARCHAR,
    initial_cash                    DOUBLE,
    initial_shares                  DOUBLE,

    PRIMARY KEY (run_id, agent_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS agent_round (
    run_id                          VARCHAR(26) NOT NULL,
    round_number                    INTEGER NOT NULL,
    agent_id                        INTEGER NOT NULL,
    signal                          DOUBLE,
    signal_error                    DOUBLE,
    action                          VARCHAR NOT NULL,
    limit_price                     DOUBLE,
    order_qty                       DOUBLE,
    aggressiveness                  DOUBLE,
    executed_qty                    DOUBLE NOT NULL DEFAULT 0.0,
    executed_price_avg              DOUBLE,
    fill_ratio                      DOUBLE,
    is_filled                       BOOLEAN NOT NULL DEFAULT FALSE,
    is_partial                      BOOLEAN NOT NULL DEFAULT FALSE,
    cash_start                      DOUBLE NOT NULL,
    inventory_start                 DOUBLE NOT NULL,
    cash_end                        DOUBLE NOT NULL,
    inventory_end                   DOUBLE NOT NULL,

    PRIMARY KEY (run_id, round_number, agent_id),

    FOREIGN KEY (run_id, agent_id) REFERENCES agent_population(run_id, agent_id),

    FOREIGN KEY (run_id, round_number) REFERENCES market_round(run_id, round_number)
    );
    """)
    con.close()
