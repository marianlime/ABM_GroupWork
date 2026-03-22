# Database And Views Documentation

This document describes the DuckDB schema used by the ABM simulation, including:

- core SQL tables
- derived SQL views
- metric equations
- plain-English explanations of what each field or view means

The database has two layers:

- `TABLE`s store raw simulation state and experiment metadata
- `VIEW`s derive analysis metrics from the raw tables

In general:

- `experiments` is the top-level object
- each experiment contains many `generations`
- each generation contains many `rounds`
- each round contains market-level and agent-level records

## Core Tables

### `experiments`
Top-level experiment metadata.

Purpose:
- one row per evolutionary experiment
- stores configuration, algorithm settings, and total initial endowment

Important fields:
- `experiment_id`: unique experiment identifier
- `experiment_name`: human-readable label
- `experiment_type`: scenario / run type label
- `n_generations`: number of generations in the experiment
- `n_rounds`: number of rounds per generation
- `n_agents`: total number of agents in each generation
- `total_initial_cash`: total cash distributed across all agents
- `total_initial_shares`: total shares distributed across all agents
- `algorithm_name`: evolutionary algorithm name
- `algorithm_params`: JSON algorithm settings

Note:
- `total_initial_cash` and `total_initial_shares` are experiment totals, not per-agent values

### `generations`
Metadata for each generation inside an experiment.

Purpose:
- one row per generation
- stores lifecycle status and generation-level evolved-parameter means

Important fields:
- `generation_id`: integer generation index within the experiment
- `generation_status`: `STARTED`, `COMPLETED`, `FAILED`, etc.
- `generation_progress`: generation completion percent
- `mean_qty_aggression`
- `mean_signal_aggression`
- `mean_threshold`
- `mean_signal_clip`

### `gbm_config`
Configuration used to generate the fundamental path for a generation.

Important fields:
- `S0`
- `volatility`
- `drift`
- `seed`

### `fundamental_series`
Fundamental value path for each generation.

Purpose:
- one row per round
- stores the exogenous fundamental price used in the simulation

Important fields:
- `round_number`
- `price`

### `market_round`
Market-clearing result for each round.

Purpose:
- one row per round
- stores clearing price, volume, activity, and depth statistics

Important fields:
- `p_t`: round clearing price
- `best_bid`
- `best_ask`
- `volume`
- `n_trades`
- `demand_at_p`
- `supply_at_p`

### `agent_population`
Per-generation agent population snapshot.

Purpose:
- one row per agent per generation
- stores strategy type, information parameter, and evolved strategy parameters

Important fields:
- `agent_id`
- `strategy_type`
- `info_param`
- `qty_aggression`
- `signal_aggression`
- `threshold`
- `signal_clip`
- `initial_cash`
- `initial_shares`

Important note:
- `initial_cash` and `initial_shares` are per-agent starting allocations
- they are not the experiment totals from `experiments`

### `agent_round`
Per-agent per-round behavior and outcome record.

Purpose:
- one row per agent per round
- stores what the agent tried to do and what actually happened

Important fields:
- `signal`
- `signal_error`
- `action`
- `limit_price`
- `order_qty`
- `aggressiveness`
- `executed_qty`
- `executed_price_avg`
- `fill_ratio`
- `cash_start`
- `inventory_start`
- `cash_end`
- `inventory_end`

## View Conventions

Most views are grouped into:

- agent-level per-round analysis views
- market-level per-round summary views
- strategy-level per-round and per-generation views

Where relevant:

- lower `signal_accuracy` is better in this codebase, because it is actually absolute signal error
- `execution_price_deviation` is a relative deviation, not a raw price difference

## Agent-Level Views

### `agent_profit_loss_per_round`
Per-agent mark-to-market profit/loss within a round.

Equation:

\[
\text{profit\_loss}
=
(\text{cash\_end} + \text{inventory\_end} \cdot P_t)
-
(\text{cash\_start} + \text{inventory\_start} \cdot P_t)
\]

where:

\[
P_t = \text{COALESCE}(mr.p_t, fs.price, 0)
\]

Meaning:
- compares start-of-round and end-of-round wealth using the same round price
- positive means wealth increased during the round
- negative means wealth decreased

### `agent_fill_rate_per_round`
Fraction of requested order quantity that was executed.

Equation:

\[
\text{fill\_rate}
=
\begin{cases}
\frac{\text{executed\_qty}}{\text{order\_qty}} & \text{if } \text{order\_qty} > 0 \\
\text{NULL} & \text{otherwise}
\end{cases}
\]

Meaning:
- `1.0` means fully filled
- `0.0` means submitted but not filled
- `NULL` means no order submitted

### `agent_aggressiveness_vs_spread`
Pairs agent aggressiveness with market spread in the same round.

Equation:

\[
\text{market\_spread} = \text{best\_ask} - \text{best\_bid}
\]

Meaning:
- useful for comparing how aggressive the agent is relative to current market tightness

### `agent_aggressiveness_vs_spread_derived`
Derived comparisons between aggressiveness and market spread.

Equations:

\[
\text{aggressiveness\_spread\_ratio}
=
\begin{cases}
\frac{\text{aggressiveness}}{\text{market\_spread}} & \text{if } \text{market\_spread} > 0 \\
\text{NULL} & \text{otherwise}
\end{cases}
\]

\[
\text{aggressiveness\_spread\_gap}
=
\text{aggressiveness} - \text{market\_spread}
\]

### `agent_signal_accuracy_per_round`
Absolute error between the agent signal and the fundamental price.

Equation:

\[
\text{signal\_accuracy} = |\text{signal} - \text{fundamental\_price}|
\]

Meaning:
- smaller is better
- despite the name, this is an absolute error metric, not a bounded accuracy score

### `agent_inventory_turnover_per_round`
Relative inventory change using a symmetric denominator.

Equation:

\[
\text{inventory\_turnover}
=
\begin{cases}
\frac{|\text{inventory\_end} - \text{inventory\_start}|}
{\left(|\text{inventory\_start}| + |\text{inventory\_end}|\right)/2}
&
\text{if } \left(|\text{inventory\_start}| + |\text{inventory\_end}|\right)/2 \neq 0
\\
\text{NULL}
&
\text{otherwise}
\end{cases}
\]

Meaning:
- measures how much the position changed relative to average position size
- safer than dividing by `inventory_start` directly

### `agent_execution_price_deviation_per_round`
Relative execution-price deviation from the round market price.

Equation:

\[
\text{execution\_price\_deviation}
=
\begin{cases}
\frac{\text{executed\_price\_avg} - p_t}{|p_t|}
&
\text{if } |p_t| > 10^{-12}
\\
\text{NULL}
&
\text{otherwise}
\end{cases}
\]

Meaning:
- positive means execution happened above the round market price
- negative means execution happened below it
- this is a relative slippage-style measure, not true causal market impact

### `agent_order_type_distribution_per_round`
Count of agent orders by action type per round.

Equation:

\[
\text{order\_count} = \text{COUNT(*)}
\]

grouped by:
- `experiment_id`
- `generation_id`
- `round_number`
- `agent_id`
- `action`

### `agent_volume_share_per_round`
Agent share of total market volume in that round.

Equation:

\[
\text{volume\_share}
=
\begin{cases}
\frac{\text{executed\_qty}}{\text{market\_volume}} & \text{if } \text{market\_volume} > 0 \\
0 & \text{otherwise}
\end{cases}
\]

Meaning:
- share of round turnover attributable to that agent

### `agent_behavior_change_per_round`
Round-to-round change in agent behavior variables.

Equations:

\[
\text{aggressiveness\_change}
=
\text{aggressiveness}_t - \text{aggressiveness}_{t-1}
\]

\[
\text{order\_qty\_change}
=
\text{order\_qty}_t - \text{order\_qty}_{t-1}
\]

\[
\text{inventory\_change}
=
\text{inventory\_end,t} - \text{inventory\_end,t-1}
\]

Implementation note:
- previous values are taken with SQL `LAG(...) OVER (...)`
- first round for each agent yields `NULL`

### `agent_avg_trade_size_per_round`
Executed quantity per agent per round.

Equation:

\[
\text{avg\_trade\_size} = \text{executed\_qty}
\]

Meaning:
- agents have at most one round-level aggregate execution record here
- so “average trade size” is the executed quantity for that round

### `agent_relative_performance_per_round`
Agent profit/loss relative to the round average.

Equation:

\[
\text{relative\_profit\_loss}
=
\text{profit\_loss} - \overline{\text{profit\_loss}}_{\text{round}}
\]

where round profit/loss uses the same mark-to-market definition as `agent_profit_loss_per_round`.

Meaning:
- positive means the agent outperformed the average agent that round
- negative means underperformed

### `agent_inventory_risk_per_round`
Absolute end-of-round inventory exposure.

Equation:

\[
\text{inventory\_risk} = |\text{inventory\_end}|
\]

Meaning:
- simple exposure metric
- not a full financial risk model

## Market-Level View

### `market_round_summary`
Round summary of order-book price statistics and market context.

Fields and equations:

\[
\text{max\_bid} = \max(\text{limit\_price} \mid \text{action} = \text{buy})
\]

\[
\text{max\_sell} = \max(\text{limit\_price} \mid \text{action} = \text{sell})
\]

\[
\text{min\_bid} = \min(\text{limit\_price} \mid \text{action} = \text{buy})
\]

\[
\text{min\_sell} = \min(\text{limit\_price} \mid \text{action} = \text{sell})
\]

\[
\text{bid\_price\_q2} = Q_{0.5}(\text{buy limit prices})
\]

\[
\text{ask\_price\_q3} = Q_{0.75}(\text{sell limit prices})
\]

\[
\text{fundamental\_price} = \text{fundamental\_series.price}
\]

\[
\text{mid\_price} = \frac{\text{best\_bid} + \text{best\_ask}}{2}
\]

Meaning:
- summarizes where orders were posted around the market-clearing outcome

## Strategy-Level Views

### `strategy_performance_per_round`
Per-round metrics aggregated by `strategy_type`.

Grouping:
- `experiment_id`
- `generation_id`
- `round_number`
- `strategy_type`

Metrics:

\[
\text{avg\_wealth} = \text{mean of agent round wealth}
\]

\[
\text{avg\_profit\_loss} = \text{mean of agent mark-to-market profit/loss}
\]

\[
\text{avg\_fill\_rate} = \text{mean of fill rate}
\]

\[
\text{avg\_aggressiveness} = \text{mean of aggressiveness}
\]

\[
\text{avg\_signal\_accuracy} = \text{mean of absolute signal error}
\]

\[
\text{avg\_inventory\_turnover} = \text{mean of symmetric inventory turnover}
\]

\[
\text{avg\_execution\_price\_deviation} = \text{mean of relative execution-price deviation}
\]

\[
\text{avg\_volume\_share} = \text{mean of per-agent volume share}
\]

\[
\text{avg\_trade\_size} = \text{mean of executed quantity}
\]

\[
\text{avg\_inventory\_risk} = \text{mean of } |\text{inventory\_end}|
\]

Important note:
- `avg_volume_share` is the average of per-agent shares, not the summed strategy share

### `strategy_performance_per_generation`
Generation-level averages of the strategy per-round metrics.

Grouping:
- `experiment_id`
- `generation_id`
- `strategy_type`

Metrics:
- `total_agent_rounds`
- `avg_wealth_per_gen`
- `avg_profit_loss_per_gen`
- `avg_fill_rate_per_gen`
- `avg_aggressiveness_per_gen`
- `avg_signal_accuracy_per_gen`
- `avg_inventory_turnover_per_gen`
- `avg_execution_price_deviation_per_gen`
- `avg_volume_share_per_gen`
- `avg_trade_size_per_gen`
- `avg_inventory_risk_per_gen`

Meaning:
- each metric is the mean of the strategy’s per-round metric across the generation

### `strategy_evolution_across_generations`
Tracks how strategy-level performance evolves over generations.

Fields:
- `avg_profit_loss_per_gen`
- `profit_change_from_prev_gen`

Equation:

\[
\text{profit\_change\_from\_prev\_gen}
=
\text{avg\_profit\_loss\_per\_gen,t}
-
\text{avg\_profit\_loss\_per\_gen,t-1}
\]

Meaning:
- positive means the strategy improved versus the previous generation
- negative means it deteriorated

## SQL Database Notes

### Why use tables and views?
The raw simulation produces many low-level records:
- experiment metadata
- generation metadata
- market-round records
- agent-round records

Those raw tables are flexible but noisy to query directly.

Views are used to:
- centralize metric definitions
- keep GUI queries simpler
- ensure SQL analysis and GUI analysis use the same formulas
- make downstream analysis reproducible

### Why some views use `CREATE OR REPLACE VIEW`
Some metrics were revised over time:
- profit/loss
- relative profit/loss
- inventory turnover
- execution-price deviation

`CREATE OR REPLACE VIEW` ensures the metric definition is updated when `create_database(...)` runs against an existing database.

### Important interpretation cautions

Some metric names are historical and not perfect finance terminology.

Examples:
- `signal_accuracy` is actually absolute error, so lower is better
- `execution_price_deviation` is slippage-like deviation, not true causal market impact
- `inventory_risk` is absolute inventory exposure, not a full risk model
- `avg_volume_share` at strategy level is an average of agent shares, not an aggregate strategy market-share ratio

### Recommended workflow
If you are unsure what a chart means:

1. Check whether it comes from a raw table or derived view.
2. Check the exact equation in this document.
3. Check whether the metric is:
   - absolute or relative
   - per-agent, per-round, or per-generation
   - averaged or summed

That distinction matters a lot for interpretation.
