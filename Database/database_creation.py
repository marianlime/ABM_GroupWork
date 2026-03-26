import duckdb
from pathlib import Path


def create_database(DB_PATH):
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(DB_PATH)

    con.execute("""
    CREATE TABLE IF NOT EXISTS experiments (
    experiment_id                       VARCHAR(26) PRIMARY KEY,
    experiment_name                     VARCHAR(64) NOT NULL,
    experiment_type                     VARCHAR,
    creation_time                       TIMESTAMP NOT NULL,
    completion_time                     TIMESTAMP,
    run_notes                           VARCHAR,
    n_generations                       INTEGER NOT NULL,
    n_rounds                            INTEGER NOT NULL,
    fundamental_source                  VARCHAR NOT NULL,
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
    info_param_distribution_type        VARCHAR NOT NULL,
    distribution_data                   JSON,
    signal_generator_noise_distribution VARCHAR NOT NULL,
    algorithm_name                      VARCHAR NOT NULL,
    algorithm_params                    JSON,
    bias                                FLOAT NOT NULL
    );
    """)
    try:
        experiment_columns = {
            row[1] for row in con.execute("PRAGMA table_info('experiments')").fetchall()
        }
        if "info_param_distribution_type" not in experiment_columns:
            con.execute("ALTER TABLE experiments ADD COLUMN info_param_distribution_type VARCHAR")
            if "noise_parameter_distribution_type" in experiment_columns:
                con.execute("""
                    UPDATE experiments
                    SET info_param_distribution_type = noise_parameter_distribution_type
                    WHERE info_param_distribution_type IS NULL
                """)
                con.execute("ALTER TABLE experiments DROP COLUMN noise_parameter_distribution_type")
    except Exception:
        pass

    con.execute("""
    CREATE TABLE IF NOT EXISTS generations (
    experiment_id                       VARCHAR(26) NOT NULL,
    generation_id                       INTEGER NOT NULL,
    creation_time                       TIMESTAMP NOT NULL,
    completion_time                     TIMESTAMP,
    generation_status                   VARCHAR NOT NULL,
    generation_progress                 DOUBLE NOT NULL DEFAULT 0.0,
    mean_qty_aggression                 DOUBLE,
    mean_signal_aggression              DOUBLE,
    mean_threshold                      DOUBLE,
    mean_signal_clip                    DOUBLE,

    PRIMARY KEY (experiment_id, generation_id),
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
    );
    """)

    con.execute("ALTER TABLE generations ADD COLUMN IF NOT EXISTS mean_qty_aggression DOUBLE")
    con.execute("ALTER TABLE generations ADD COLUMN IF NOT EXISTS mean_signal_aggression DOUBLE")
    con.execute("ALTER TABLE generations ADD COLUMN IF NOT EXISTS mean_threshold DOUBLE")
    con.execute("ALTER TABLE generations ADD COLUMN IF NOT EXISTS mean_signal_clip DOUBLE")

    con.execute("""
    CREATE TABLE IF NOT EXISTS gbm_config (
    experiment_id                 VARCHAR(26) NOT NULL,
    generation_id                 INTEGER NOT NULL,
    s0                            DOUBLE NOT NULL,
    drift                         DOUBLE NOT NULL,
    volatility                    DOUBLE NOT NULL,
    seed                          VARCHAR(64),
    PRIMARY KEY (experiment_id, generation_id),
    FOREIGN KEY (experiment_id, generation_id) REFERENCES generations(experiment_id, generation_id)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS fundamental_series (
    experiment_id VARCHAR(26) NOT NULL,
    generation_id INTEGER NOT NULL,
    round_number  INTEGER NOT NULL,
    price         DOUBLE NOT NULL,

    PRIMARY KEY (experiment_id, generation_id, round_number),
    FOREIGN KEY (experiment_id, generation_id) REFERENCES generations(experiment_id, generation_id)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS market_round (
    experiment_id     VARCHAR(26) NOT NULL,
    generation_id     INTEGER NOT NULL,
    round_number      INTEGER NOT NULL,

    p_t               DOUBLE,
    best_bid          DOUBLE,
    best_ask          DOUBLE,
    volume            DOUBLE NOT NULL DEFAULT 0,
    n_trades          BIGINT NOT NULL DEFAULT 0,

    demand_at_p       DOUBLE NOT NULL DEFAULT 0,
    supply_at_p       DOUBLE NOT NULL DEFAULT 0,

    n_active_buyers   INTEGER NOT NULL DEFAULT 0,
    n_active_sellers  INTEGER NOT NULL DEFAULT 0,
    n_active_total    INTEGER NOT NULL DEFAULT 0,

    bid_depth_total   DOUBLE NOT NULL DEFAULT 0,
    ask_depth_total   DOUBLE NOT NULL DEFAULT 0,

    price_levels_bid  INTEGER NOT NULL DEFAULT 0,
    price_levels_ask  INTEGER NOT NULL DEFAULT 0,

    PRIMARY KEY (experiment_id, generation_id, round_number),
    FOREIGN KEY (experiment_id, generation_id) REFERENCES generations(experiment_id, generation_id),
    FOREIGN KEY (experiment_id, generation_id, round_number)
        REFERENCES fundamental_series(experiment_id, generation_id, round_number)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS agent_population(
    experiment_id                   VARCHAR(26) NOT NULL,
    generation_id                   INTEGER NOT NULL,
    agent_id                        INTEGER NOT NULL,
    strategy_type                   VARCHAR NOT NULL,
    info_param                      DOUBLE,
    qty_aggression                  DOUBLE,
    signal_aggression               DOUBLE,
    threshold                       DOUBLE,
    signal_clip                     DOUBLE,
    group_label                     VARCHAR,
    initial_cash                    DOUBLE,
    initial_shares                  DOUBLE,

    PRIMARY KEY (experiment_id, generation_id, agent_id),
    FOREIGN KEY (experiment_id, generation_id) REFERENCES generations(experiment_id, generation_id)
    );
    """)
    con.execute("ALTER TABLE agent_population ADD COLUMN IF NOT EXISTS info_param DOUBLE")
    try:
        existing_columns = {
            row[1] for row in con.execute("PRAGMA table_info('agent_population')").fetchall()
        }
        if "noise_parameter" in existing_columns:
            con.execute("""
                UPDATE agent_population
                SET info_param = noise_parameter
                WHERE info_param IS NULL AND noise_parameter IS NOT NULL
            """)
            con.execute("ALTER TABLE agent_population DROP COLUMN noise_parameter")
    except Exception:
        pass

    con.execute("""
    CREATE TABLE IF NOT EXISTS agent_round (
    experiment_id                   VARCHAR(26) NOT NULL,
    generation_id                   INTEGER NOT NULL,
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

    PRIMARY KEY (experiment_id, generation_id, round_number, agent_id),
    FOREIGN KEY (experiment_id, generation_id, agent_id)
        REFERENCES agent_population(experiment_id, generation_id, agent_id),
    FOREIGN KEY (experiment_id, generation_id, round_number)
        REFERENCES market_round(experiment_id, generation_id, round_number)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS trade_execution (
    experiment_id                   VARCHAR(26) NOT NULL,
    generation_id                   INTEGER NOT NULL,
    round_number                    INTEGER NOT NULL,
    trade_id                        INTEGER NOT NULL,
    buyer_agent_id                  INTEGER NOT NULL,
    seller_agent_id                 INTEGER NOT NULL,
    price                           DOUBLE NOT NULL,
    quantity                        DOUBLE NOT NULL,
    notional                        DOUBLE NOT NULL,

    PRIMARY KEY (experiment_id, generation_id, round_number, trade_id),
    FOREIGN KEY (experiment_id, generation_id, round_number)
        REFERENCES market_round(experiment_id, generation_id, round_number)
    );
    """)
    # --- Agent-Round-Level Meta-Analysis Views ---

    # 1. Agent Profit/Loss per Round
    con.execute("""
    CREATE OR REPLACE VIEW agent_profit_loss_per_round AS
    SELECT
        ar.experiment_id,
        ar.generation_id,
        ar.round_number,
        ar.agent_id,
        (ar.cash_end + ar.inventory_end * COALESCE(mr.p_t, fs.price, 0)) -
        (ar.cash_start + ar.inventory_start * COALESCE(mr.p_t, fs.price, 0)) AS profit_loss
    FROM agent_round ar
    LEFT JOIN market_round mr
      ON ar.experiment_id = mr.experiment_id
     AND ar.generation_id = mr.generation_id
     AND ar.round_number = mr.round_number
    LEFT JOIN fundamental_series fs
      ON ar.experiment_id = fs.experiment_id
     AND ar.generation_id = fs.generation_id
     AND ar.round_number = fs.round_number;
    """)

    # 2. Agent Order Fill Rate per Round
    con.execute("""
    CREATE VIEW IF NOT EXISTS agent_fill_rate_per_round AS
    SELECT
        experiment_id,
        generation_id,
        round_number,
        agent_id,
        CASE WHEN order_qty > 0 THEN executed_qty * 1.0 / order_qty ELSE NULL END AS fill_rate
    FROM agent_round;
    """)

    # 3. Agent Aggressiveness vs. Market Spread
    con.execute("""
    CREATE VIEW IF NOT EXISTS agent_aggressiveness_vs_spread AS
    SELECT
        ar.experiment_id,
        ar.generation_id,
        ar.round_number,
        ar.agent_id,
        ar.aggressiveness,
        ABS(mr.best_ask - mr.best_bid) AS market_spread
    FROM agent_round ar
    JOIN market_round mr
      ON ar.experiment_id = mr.experiment_id
     AND ar.generation_id = mr.generation_id
     AND ar.round_number = mr.round_number;
    """)

    con.execute("""
    CREATE VIEW IF NOT EXISTS agent_aggressiveness_vs_spread_derived AS
    SELECT
        experiment_id,
        generation_id,
        round_number,
        agent_id,
        aggressiveness,
        market_spread,
        CASE WHEN market_spread > 0 THEN aggressiveness / market_spread ELSE NULL END AS aggressiveness_spread_ratio,
        aggressiveness - market_spread AS aggressiveness_spread_gap
    FROM agent_aggressiveness_vs_spread;
    """)

    # 4. Signal Accuracy per Agent per Round
    con.execute("""
    CREATE VIEW IF NOT EXISTS agent_signal_accuracy_per_round AS
    SELECT
        ar.experiment_id,
        ar.generation_id,
        ar.round_number,
        ar.agent_id,
        ABS(ar.signal - fs.price) AS signal_accuracy
    FROM agent_round ar
    JOIN fundamental_series fs
      ON ar.experiment_id = fs.experiment_id
     AND ar.generation_id = fs.generation_id
     AND ar.round_number = fs.round_number;
    """)

    # 5. Inventory Turnover per Round
    con.execute("""
    CREATE OR REPLACE VIEW agent_inventory_turnover_per_round AS
    SELECT
        experiment_id,
        generation_id,
        round_number,
        agent_id,
        CASE
            WHEN (ABS(inventory_start) + ABS(inventory_end)) / 2.0 != 0
            THEN ABS(inventory_end - inventory_start) * 1.0 /
                 ((ABS(inventory_start) + ABS(inventory_end)) / 2.0)
            ELSE NULL
        END AS inventory_turnover
    FROM agent_round;
    """)

    # 6. Execution Price Deviation per Agent per Round
    con.execute("""
    CREATE OR REPLACE VIEW agent_execution_price_deviation_per_round AS
    SELECT
        ar.experiment_id,
        ar.generation_id,
        ar.round_number,
        ar.agent_id,
        ar.executed_price_avg,
        mr.p_t AS market_price,
        CASE
            WHEN mr.p_t IS NOT NULL AND ABS(mr.p_t) > 1e-12
            THEN (ar.executed_price_avg - mr.p_t) / ABS(mr.p_t)
            ELSE NULL
        END AS execution_price_deviation
    FROM agent_round ar
    JOIN market_round mr
      ON ar.experiment_id = mr.experiment_id
     AND ar.generation_id = mr.generation_id
     AND ar.round_number = mr.round_number;
    """)

    # 7. Order Type Distribution per Agent per Round
    con.execute("""
    CREATE VIEW IF NOT EXISTS agent_order_type_distribution_per_round AS
    SELECT
        experiment_id,
        generation_id,
        round_number,
        agent_id,
        action,
        COUNT(*) AS order_count
    FROM agent_round
    GROUP BY experiment_id, generation_id, round_number, agent_id, action;
    """)

    # Additional Agent-Round-Level Views

    # 1. Agent’s Share of Market Volume per Round
    con.execute("""
    CREATE VIEW IF NOT EXISTS agent_volume_share_per_round AS
    SELECT
        ar.experiment_id,
        ar.generation_id,
        ar.round_number,
        ar.agent_id,
        CASE WHEN mr.volume > 0 THEN ar.executed_qty / mr.volume ELSE 0 END AS volume_share
    FROM agent_round ar
    JOIN market_round mr
      ON ar.experiment_id = mr.experiment_id
     AND ar.generation_id = mr.generation_id
     AND ar.round_number = mr.round_number;
    """)

    # 3. Round-to-Round Change in Agent Behaviour
    con.execute("""
    CREATE VIEW IF NOT EXISTS agent_behavior_change_per_round AS
    SELECT
        experiment_id,
        generation_id,
        round_number,
        agent_id,
        aggressiveness - LAG(aggressiveness) OVER (PARTITION BY experiment_id, generation_id, agent_id ORDER BY round_number) AS aggressiveness_change,
        order_qty - LAG(order_qty) OVER (PARTITION BY experiment_id, generation_id, agent_id ORDER BY round_number) AS order_qty_change,
        inventory_end - LAG(inventory_end) OVER (PARTITION BY experiment_id, generation_id, agent_id ORDER BY round_number) AS inventory_change
    FROM agent_round;
    """)

    # 4. Average Trade Size per Agent per Round
    con.execute("""
    CREATE VIEW IF NOT EXISTS agent_avg_trade_size_per_round AS
    SELECT
        experiment_id,
        generation_id,
        round_number,
        agent_id,
        executed_qty AS avg_trade_size  -- Since agents execute once per round
    FROM agent_round;
    """)

    # 5. Agent’s Relative Performance to Market
    con.execute("""
    CREATE OR REPLACE VIEW agent_relative_performance_per_round AS
    WITH round_avg_profit AS (
        SELECT
            ar.experiment_id,
            ar.generation_id,
            ar.round_number,
            AVG((ar.cash_end + ar.inventory_end * COALESCE(mr.p_t, fs.price, 0)) -
                (ar.cash_start + ar.inventory_start * COALESCE(mr.p_t, fs.price, 0))) AS avg_profit_loss
        FROM agent_round ar
        LEFT JOIN market_round mr
          ON ar.experiment_id = mr.experiment_id
         AND ar.generation_id = mr.generation_id
         AND ar.round_number = mr.round_number
        LEFT JOIN fundamental_series fs
          ON ar.experiment_id = fs.experiment_id
         AND ar.generation_id = fs.generation_id
         AND ar.round_number = fs.round_number
        GROUP BY ar.experiment_id, ar.generation_id, ar.round_number
    )
    SELECT
        ar.experiment_id,
        ar.generation_id,
        ar.round_number,
        ar.agent_id,
        (ar.cash_end + ar.inventory_end * COALESCE(mr.p_t, fs.price, 0)) -
        (ar.cash_start + ar.inventory_start * COALESCE(mr.p_t, fs.price, 0)) - rap.avg_profit_loss AS relative_profit_loss
    FROM agent_round ar
    JOIN round_avg_profit rap
      ON ar.experiment_id = rap.experiment_id
     AND ar.generation_id = rap.generation_id
     AND ar.round_number = rap.round_number
    LEFT JOIN market_round mr
      ON ar.experiment_id = mr.experiment_id
     AND ar.generation_id = mr.generation_id
     AND ar.round_number = mr.round_number
    LEFT JOIN fundamental_series fs
      ON ar.experiment_id = fs.experiment_id
     AND ar.generation_id = fs.generation_id
     AND ar.round_number = fs.round_number;
    """)

    # 6. Agent’s Inventory Risk per Round
    con.execute("""
    CREATE VIEW IF NOT EXISTS agent_inventory_risk_per_round AS
    SELECT
        experiment_id,
        generation_id,
        round_number,
        agent_id,
        ABS(inventory_end) AS inventory_risk
    FROM agent_round;
    """)

    # Market-Round-Level Summary View
    con.execute("""
    CREATE VIEW IF NOT EXISTS market_round_summary AS
    SELECT
        mr.experiment_id,
        mr.generation_id,
        mr.round_number,
        MAX(CASE WHEN ar.action = 'buy' THEN ar.limit_price END) AS max_bid,
        MAX(CASE WHEN ar.action = 'sell' THEN ar.limit_price END) AS max_sell,
        MIN(CASE WHEN ar.action = 'buy' THEN ar.limit_price END) AS min_bid,
        MIN(CASE WHEN ar.action = 'sell' THEN ar.limit_price END) AS min_sell,
        QUANTILE(CASE WHEN ar.action = 'buy' THEN ar.limit_price END, 0.5) AS bid_price_q2,
        QUANTILE(CASE WHEN ar.action = 'sell' THEN ar.limit_price END, 0.75) AS ask_price_q3,
        fs.price AS fundamental_price,
        (mr.best_bid + mr.best_ask) / 2 AS mid_price
    FROM market_round mr
    JOIN fundamental_series fs
      ON mr.experiment_id = fs.experiment_id
     AND mr.generation_id = fs.generation_id
     AND mr.round_number = fs.round_number
    LEFT JOIN agent_round ar
      ON mr.experiment_id = ar.experiment_id
     AND mr.generation_id = ar.generation_id
     AND mr.round_number = ar.round_number
    GROUP BY mr.experiment_id, mr.generation_id, mr.round_number, fs.price, mr.best_bid, mr.best_ask;
    """)

    # Strategy Type Performance Views

    # 1. Strategy Type Performance per Round
    con.execute("""
    CREATE OR REPLACE VIEW strategy_performance_per_round AS
    SELECT
        ar.experiment_id,
        ar.generation_id,
        ar.round_number,
        ap.strategy_type,
        COUNT(*) AS num_agents,
        AVG(ar.cash_end + ar.inventory_end * COALESCE(mr.p_t, fs.price, 0)) AS avg_wealth,
        AVG((ar.cash_end + ar.inventory_end * COALESCE(mr.p_t, fs.price, 0)) -
            (ar.cash_start + ar.inventory_start * COALESCE(mr.p_t, fs.price, 0))) AS avg_profit_loss,
        AVG(CASE WHEN ar.order_qty > 0 THEN ar.executed_qty * 1.0 / ar.order_qty ELSE NULL END) AS avg_fill_rate,
        AVG(ar.aggressiveness) AS avg_aggressiveness,
        AVG(ABS(ar.signal - fs.price)) AS avg_signal_accuracy,
        AVG(
            CASE
                WHEN (ABS(ar.inventory_start) + ABS(ar.inventory_end)) / 2.0 != 0
                THEN ABS(ar.inventory_end - ar.inventory_start) * 1.0 /
                     ((ABS(ar.inventory_start) + ABS(ar.inventory_end)) / 2.0)
                ELSE NULL
            END
        ) AS avg_inventory_turnover,
        AVG(
            CASE
                WHEN mr.p_t IS NOT NULL AND ABS(mr.p_t) > 1e-12
                THEN (ar.executed_price_avg - mr.p_t) / ABS(mr.p_t)
                ELSE NULL
            END
        ) AS avg_execution_price_deviation,
        AVG(CASE WHEN mr.volume > 0 THEN ar.executed_qty / mr.volume ELSE 0 END) AS avg_volume_share,
        AVG(ar.executed_qty) AS avg_trade_size,
        AVG(ABS(ar.inventory_end)) AS avg_inventory_risk
    FROM agent_round ar
    JOIN agent_population ap
      ON ar.experiment_id = ap.experiment_id
     AND ar.generation_id = ap.generation_id
     AND ar.agent_id = ap.agent_id
    LEFT JOIN fundamental_series fs
      ON ar.experiment_id = fs.experiment_id
     AND ar.generation_id = fs.generation_id
     AND ar.round_number = fs.round_number
    LEFT JOIN market_round mr
      ON ar.experiment_id = mr.experiment_id
     AND ar.generation_id = mr.generation_id
     AND ar.round_number = mr.round_number
    GROUP BY ar.experiment_id, ar.generation_id, ar.round_number, ap.strategy_type;
    """)

    # 2. Strategy Type Performance per Generation
    con.execute("""
    CREATE OR REPLACE VIEW strategy_performance_per_generation AS
    SELECT
        experiment_id,
        generation_id,
        strategy_type,
        COUNT(*) AS total_agent_rounds,
        AVG(avg_wealth) AS avg_wealth_per_gen,
        AVG(avg_profit_loss) AS avg_profit_loss_per_gen,
        AVG(avg_fill_rate) AS avg_fill_rate_per_gen,
        AVG(avg_aggressiveness) AS avg_aggressiveness_per_gen,
        AVG(avg_signal_accuracy) AS avg_signal_accuracy_per_gen,
        AVG(avg_inventory_turnover) AS avg_inventory_turnover_per_gen,
        AVG(avg_execution_price_deviation) AS avg_execution_price_deviation_per_gen,
        AVG(avg_volume_share) AS avg_volume_share_per_gen,
        AVG(avg_trade_size) AS avg_trade_size_per_gen,
        AVG(avg_inventory_risk) AS avg_inventory_risk_per_gen
    FROM strategy_performance_per_round
    GROUP BY experiment_id, generation_id, strategy_type;
    """)

    # 3. Strategy Type Evolution Across Generations
    con.execute("""
    CREATE OR REPLACE VIEW strategy_evolution_across_generations AS
    SELECT
        experiment_id,
        strategy_type,
        generation_id,
        avg_profit_loss_per_gen,
        avg_profit_loss_per_gen - LAG(avg_profit_loss_per_gen) OVER (PARTITION BY experiment_id, strategy_type ORDER BY generation_id) AS profit_change_from_prev_gen
    FROM strategy_performance_per_generation
    ORDER BY experiment_id, strategy_type, generation_id;
    """)

    con.close()
