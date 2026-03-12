from collections import Counter
import random


# Keep one canonical strategy ordering for consistency in:
# - population creation
# - counting
# - plotting later
STRATEGY_ORDER = [
    "zi",
    "signal_following",
    "utility_maximiser",
    "contrarian",
    "adapt_sig",
    "threshold_signal",
    "inventory_aware_utility",
    "patient_signal",
]


VALID_STRATEGIES = set(STRATEGY_ORDER)


def initial_population_from_counts(strategy_counts: dict[str, int]) -> list[dict]:
    """
    Build a population_spec from a strategy-count dictionary.

    Example input:
        {
            "zi": 10,
            "signal_following": 10,
            "utility_maximiser": 10,
            "contrarian": 10,
            "adapt_sig": 10,
        }

    Example output:
        [
            {"trader_type": "zi"},
            {"trader_type": "zi"},
            ...
        ]
    """
    _validate_strategy_counts(strategy_counts)

    population_spec = []
    for trader_type in STRATEGY_ORDER:
        count = strategy_counts.get(trader_type, 0)
        for _ in range(count):
            population_spec.append({"trader_type": trader_type})

    return population_spec


def count_strategies(population_spec: list[dict]) -> dict[str, int]:
    """
    Count how many agents of each strategy type are present in a population_spec.
    """
    _validate_population_spec(population_spec)

    counts = Counter(agent["trader_type"] for agent in population_spec)
    return {strategy: counts.get(strategy, 0) for strategy in STRATEGY_ORDER}


def rank_agents_by_fitness(final_score, agents) -> list[dict]:
    """
    Combine final wealth scores with trader types and sort descending by wealth.

    final_score format:
        [(agent_id, terminal_wealth), ...]

    agents format:
        game.agents dict where agents[agent_id].trader_type exists
    """
    ranked = []

    for agent_id, wealth in final_score:
        if agent_id not in agents:
            raise ValueError(f"agent_id {agent_id} found in final_score but not in agents")

        ranked.append({
            "agent_id": agent_id,
            "wealth": float(wealth),
            "trader_type": agents[agent_id].trader_type,
        })

    ranked.sort(key=lambda x: x["wealth"], reverse=True)
    return ranked


def evolve_truncation(
    final_score,
    agents,
    top_n: int,
    bottom_k: int,
    rng: random.Random | None = None,
) -> list[dict]:
    """
    Truncation selection:
    - rank all agents by terminal wealth
    - remove the bottom_k agents
    - keep the remaining survivors
    - generate bottom_k children by sampling parents uniformly from the top_n agents

    Children inherit strategy_type only.
    """
    if rng is None:
        rng = random.Random()

    ranked = rank_agents_by_fitness(final_score, agents)
    population_size = len(ranked)

    if population_size == 0:
        raise ValueError("Cannot evolve an empty population")

    if not (1 <= top_n <= population_size):
        raise ValueError(
            f"top_n must be between 1 and population size ({population_size}), got {top_n}"
        )

    if not (0 <= bottom_k <= population_size):
        raise ValueError(
            f"bottom_k must be between 0 and population size ({population_size}), got {bottom_k}"
        )

    survivors = ranked[: population_size - bottom_k]
    parent_pool = ranked[:top_n]

    children = []
    for _ in range(bottom_k):
        parent = rng.choice(parent_pool)
        children.append({"trader_type": parent["trader_type"]})

    next_population = (
        [{"trader_type": agent["trader_type"]} for agent in survivors]
        + children
    )

    return next_population


def evolve_tournament(
    final_score,
    agents,
    bottom_k: int,
    tournament_size: int,
    rng: random.Random | None = None,
) -> list[dict]:
    """
    Tournament selection:
    - remove the bottom_k agents by fitness
    - keep the remaining survivors
    - for each child, sample 'tournament_size' candidates from the full ranked population
      and select the best among them as the parent

    Children inherit strategy_type only.
    """
    if rng is None:
        rng = random.Random()

    ranked = rank_agents_by_fitness(final_score, agents)
    population_size = len(ranked)

    if population_size == 0:
        raise ValueError("Cannot evolve an empty population")

    if not (0 <= bottom_k <= population_size):
        raise ValueError(
            f"bottom_k must be between 0 and population size ({population_size}), got {bottom_k}"
        )

    if not (1 <= tournament_size <= population_size):
        raise ValueError(
            f"tournament_size must be between 1 and population size ({population_size}), got {tournament_size}"
        )

    survivors = ranked[: population_size - bottom_k]

    children = []
    for _ in range(bottom_k):
        tournament = rng.sample(ranked, k=tournament_size)
        winner = max(tournament, key=lambda x: x["wealth"])
        children.append({"trader_type": winner["trader_type"]})

    next_population = (
        [{"trader_type": agent["trader_type"]} for agent in survivors]
        + children
    )

    return next_population


def evolve_fitness_proportionate(
    final_score,
    agents,
    bottom_k: int,
    rng: random.Random | None = None,
    epsilon: float = 1e-9,
) -> list[dict]:
    """
    Fitness-proportionate (roulette wheel) selection:
    - remove the bottom_k agents by fitness
    - keep the remaining survivors
    - create bottom_k children by sampling parents with probability proportional
      to adjusted wealth

    Because wealth may be negative, fitness is shifted to be strictly positive:
        adjusted = wealth - min_wealth + epsilon
    """
    if rng is None:
        rng = random.Random()

    ranked = rank_agents_by_fitness(final_score, agents)
    population_size = len(ranked)

    if population_size == 0:
        raise ValueError("Cannot evolve an empty population")

    if not (0 <= bottom_k <= population_size):
        raise ValueError(
            f"bottom_k must be between 0 and population size ({population_size}), got {bottom_k}"
        )

    survivors = ranked[: population_size - bottom_k]

    min_wealth = min(agent["wealth"] for agent in ranked)
    adjusted_weights = [(agent["wealth"] - min_wealth + epsilon) for agent in ranked]

    if sum(adjusted_weights) <= 0:
        raise ValueError("Adjusted fitness weights must sum to a positive value")

    children = []
    for _ in range(bottom_k):
        parent = rng.choices(ranked, weights=adjusted_weights, k=1)[0]
        children.append({"trader_type": parent["trader_type"]})

    next_population = (
        [{"trader_type": agent["trader_type"]} for agent in survivors]
        + children
    )

    return next_population


def evolve_population(
    algorithm_name: str,
    final_score,
    agents,
    algorithm_params: dict,
    rng: random.Random | None = None,
) -> list[dict]:
    """
    Dispatch function for evolutionary update rules.
    """
    if algorithm_name == "truncation":
        return evolve_truncation(
            final_score=final_score,
            agents=agents,
            top_n=algorithm_params["top_n"],
            bottom_k=algorithm_params["bottom_k"],
            rng=rng,
        )

    if algorithm_name == "tournament":
        return evolve_tournament(
            final_score=final_score,
            agents=agents,
            bottom_k=algorithm_params["bottom_k"],
            tournament_size=algorithm_params["tournament_size"],
            rng=rng,
        )

    if algorithm_name == "fitness_proportionate":
        return evolve_fitness_proportionate(
            final_score=final_score,
            agents=agents,
            bottom_k=algorithm_params["bottom_k"],
            rng=rng,
            epsilon=algorithm_params.get("epsilon", 1e-9),
        )

    raise ValueError(f"Unknown algorithm_name: {algorithm_name}")


def _validate_strategy_counts(strategy_counts: dict[str, int]) -> None:
    if not isinstance(strategy_counts, dict):
        raise TypeError("strategy_counts must be a dict[str, int]")

    for trader_type, count in strategy_counts.items():
        if trader_type not in VALID_STRATEGIES:
            raise ValueError(f"Unknown strategy type: {trader_type}")
        if not isinstance(count, int):
            raise TypeError(f"Count for strategy '{trader_type}' must be an int")
        if count < 0:
            raise ValueError(f"Count for strategy '{trader_type}' cannot be negative")


def _validate_population_spec(population_spec: list[dict]) -> None:
    if not isinstance(population_spec, list):
        raise TypeError("population_spec must be a list[dict]")

    for i, agent in enumerate(population_spec):
        if not isinstance(agent, dict):
            raise TypeError(f"population_spec[{i}] must be a dict")
        if "trader_type" not in agent:
            raise ValueError(f"population_spec[{i}] is missing 'trader_type'")
        if agent["trader_type"] not in VALID_STRATEGIES:
            raise ValueError(
                f"population_spec[{i}] has unknown trader_type: {agent['trader_type']}"
            )