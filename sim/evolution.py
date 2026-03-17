from collections import Counter
import random


STRATEGY_ORDER   = ["zi", "parameterised_informed"]
VALID_STRATEGIES = set(STRATEGY_ORDER)


def initial_population_from_counts(
    strategy_counts: dict[str, int],
    default_strategy_params: dict,
    param_bounds: dict | None = None,
    rng: random.Random | None = None,
) -> list[dict]:
    """
    Build a population_spec from {"zi": N, "parameterised_informed": M}.

    If param_bounds is provided, parameterised_informed agents are initialised
    with evenly spaced values across each parameter's [lo, hi] range.  The
    assignment order is shuffled independently per parameter so that parameters
    are not correlated in the initial population.

    If param_bounds is None, all agents are seeded with default_strategy_params.
    """
    _validate_strategy_counts(strategy_counts)

    n_informed = strategy_counts.get("parameterised_informed", 0)

    if param_bounds is not None and n_informed > 0:
        _rng = rng if rng is not None else random.Random()
        informed_params = _make_evenly_spaced_params(n_informed, param_bounds, _rng)
    else:
        informed_params = [dict(default_strategy_params) for _ in range(n_informed)]

    population_spec = []
    informed_idx = 0
    for trader_type in STRATEGY_ORDER:
        count = strategy_counts.get(trader_type, 0)
        for _ in range(count):
            agent: dict = {"trader_type": trader_type}
            if trader_type == "parameterised_informed":
                agent["strategy_params"] = informed_params[informed_idx]
                informed_idx += 1
            population_spec.append(agent)

    return population_spec


def _make_evenly_spaced_params(n: int, param_bounds: dict, rng: random.Random) -> list[dict]:
    """
    For each parameter, generate n evenly spaced values across [lo, hi], then
    shuffle the assignment order independently so parameters are uncorrelated.
    Returns a list of n strategy_param dicts.
    """
    if n == 1:
        return [{param: (lo + hi) / 2.0 for param, (lo, hi, _) in param_bounds.items()}]

    param_columns = {}
    for param, (lo, hi, _) in param_bounds.items():
        step = (hi - lo) / (n - 1)
        values = [lo + i * step for i in range(n)]
        rng.shuffle(values)
        param_columns[param] = values

    return [{param: param_columns[param][i] for param in param_bounds} for i in range(n)]


def count_strategies(population_spec: list[dict]) -> dict[str, int]:
    """Count agents of each strategy type present in population_spec."""
    _validate_population_spec(population_spec)
    counts = Counter(agent["trader_type"] for agent in population_spec)
    return {strategy: counts.get(strategy, 0) for strategy in STRATEGY_ORDER}


def rank_agents_by_fitness(final_score, agents) -> list[dict]:
    """Combine final wealth scores with trader metadata, sorted descending by wealth."""
    ranked = []
    for agent_id, wealth in final_score:
        if agent_id not in agents:
            raise ValueError(f"agent_id {agent_id} found in final_score but not in agents")
        ranked.append({
            "agent_id":        agent_id,
            "wealth":          float(wealth),
            "trader_type":     agents[agent_id].trader_type,
            "info_param":      float(agents[agent_id].info_param),
            "strategy_params": dict(agents[agent_id].strategy_params),
        })

    ranked.sort(key=lambda x: x["wealth"], reverse=True)
    return ranked


def _make_child(
    parent: dict,
    rng: random.Random,
    mutation_rate: float,
    info_param_mutation_std: float,
    info_param_bounds: tuple[float, float],
    param_bounds: dict,
    default_strategy_params: dict,
    frozen_params: set = frozenset(),
) -> dict:
    """
    Create a child from a parent agent dict.

    - info_param is perturbed by Gaussian noise, clipped to info_param_bounds
    - each strategy_param is perturbed by Gaussian noise with its own std
    - with probability mutation_rate a param instead jumps to a random value
      within its bounds (exploration)
    - any param named in frozen_params is inherited unchanged from the parent
    """
    if "info_param" in frozen_params:
        info_param = parent["info_param"]
    else:
        lo, hi = info_param_bounds
        info_param = parent["info_param"] + rng.gauss(0.0, info_param_mutation_std)
        info_param = max(lo, min(hi, info_param))

    child: dict = {
        "trader_type": parent["trader_type"],
        "info_param":  info_param,
    }

    if parent["trader_type"] == "parameterised_informed":
        parent_params = parent.get("strategy_params", default_strategy_params)
        mutated_params = {}
        for param, (p_lo, p_hi, p_std) in param_bounds.items():
            val = parent_params.get(param, default_strategy_params[param])
            if rng.random() < mutation_rate:
                val = rng.uniform(p_lo, p_hi)
            else:
                val = max(p_lo, min(p_hi, val + rng.gauss(0.0, p_std)))
            mutated_params[param] = val
        child["strategy_params"] = mutated_params

    return child


def evolve_truncation(
    final_score,
    agents,
    top_n: int,
    bottom_k: int,
    param_bounds: dict,
    default_strategy_params: dict,
    rng: random.Random | None = None,
    mutation_rate: float = 0.02,
    info_param_mutation_std: float = 0.01,
    info_param_bounds: tuple[float, float] = (0.0, 1.0),
    frozen_params: set = frozenset(),
) -> list[dict]:
    """
    Truncation selection:
    - rank all agents by wealth
    - remove the bottom_k agents
    - keep the remaining survivors
    - generate bottom_k children by sampling parents uniformly from the top_n agents
    """
    if rng is None:
        rng = random.Random()

    ranked = rank_agents_by_fitness(final_score, agents)
    population_size = len(ranked)

    if population_size == 0:
        raise ValueError("Cannot evolve an empty population")
    if not (1 <= top_n <= population_size):
        raise ValueError(f"top_n must be between 1 and {population_size}, got {top_n}")
    if not (0 <= bottom_k <= population_size):
        raise ValueError(f"bottom_k must be between 0 and {population_size}, got {bottom_k}")

    survivors   = ranked[: population_size - bottom_k]
    parent_pool = ranked[:top_n]

    kwargs = dict(rng=rng, mutation_rate=mutation_rate,
                  info_param_mutation_std=info_param_mutation_std,
                  info_param_bounds=info_param_bounds,
                  param_bounds=param_bounds,
                  default_strategy_params=default_strategy_params,
                  frozen_params=frozen_params)

    children = [_make_child(rng.choice(parent_pool), **kwargs) for _ in range(bottom_k)]
    return [_make_child(a, **kwargs) for a in survivors] + children


def evolve_tournament(
    final_score,
    agents,
    bottom_k: int,
    tournament_size: int,
    param_bounds: dict,
    default_strategy_params: dict,
    rng: random.Random | None = None,
    mutation_rate: float = 0.02,
    info_param_mutation_std: float = 0.01,
    info_param_bounds: tuple[float, float] = (0.0, 1.0),
    frozen_params: set = frozenset(),
) -> list[dict]:
    """
    Tournament selection:
    - remove the bottom_k agents by fitness
    - keep the remaining survivors
    - for each child, sample tournament_size candidates and select the best as parent
    """
    if rng is None:
        rng = random.Random()

    ranked = rank_agents_by_fitness(final_score, agents)
    population_size = len(ranked)

    if population_size == 0:
        raise ValueError("Cannot evolve an empty population")
    if not (0 <= bottom_k <= population_size):
        raise ValueError(f"bottom_k must be between 0 and {population_size}, got {bottom_k}")
    if not (1 <= tournament_size <= population_size):
        raise ValueError(f"tournament_size must be between 1 and {population_size}, got {tournament_size}")

    survivors = ranked[: population_size - bottom_k]

    kwargs = dict(rng=rng, mutation_rate=mutation_rate,
                  info_param_mutation_std=info_param_mutation_std,
                  info_param_bounds=info_param_bounds,
                  param_bounds=param_bounds,
                  default_strategy_params=default_strategy_params,
                  frozen_params=frozen_params)

    children = []
    for _ in range(bottom_k):
        winner = max(rng.sample(ranked, k=tournament_size), key=lambda x: x["wealth"])
        children.append(_make_child(winner, **kwargs))

    return [_make_child(a, **kwargs) for a in survivors] + children


def evolve_fitness_proportionate(
    final_score,
    agents,
    bottom_k: int,
    param_bounds: dict,
    default_strategy_params: dict,
    rng: random.Random | None = None,
    epsilon: float = 1e-9,
    mutation_rate: float = 0.02,
    info_param_mutation_std: float = 0.01,
    info_param_bounds: tuple[float, float] = (0.0, 1.0),
    frozen_params: set = frozenset(),
) -> list[dict]:
    """
    Fitness-proportionate (roulette wheel) selection:
    - remove the bottom_k agents by fitness
    - keep the remaining survivors
    - create bottom_k children by sampling parents weighted by adjusted wealth
    """
    if rng is None:
        rng = random.Random()

    ranked = rank_agents_by_fitness(final_score, agents)
    population_size = len(ranked)

    if population_size == 0:
        raise ValueError("Cannot evolve an empty population")
    if not (0 <= bottom_k <= population_size):
        raise ValueError(f"bottom_k must be between 0 and {population_size}, got {bottom_k}")

    survivors = ranked[: population_size - bottom_k]

    min_wealth = min(a["wealth"] for a in ranked)
    weights = [(a["wealth"] - min_wealth + epsilon) for a in ranked]
    if sum(weights) <= 0:
        raise ValueError("Adjusted fitness weights must sum to a positive value")

    kwargs = dict(rng=rng, mutation_rate=mutation_rate,
                  info_param_mutation_std=info_param_mutation_std,
                  info_param_bounds=info_param_bounds,
                  param_bounds=param_bounds,
                  default_strategy_params=default_strategy_params,
                  frozen_params=frozen_params)

    children = [_make_child(rng.choices(ranked, weights=weights, k=1)[0], **kwargs)
                for _ in range(bottom_k)]
    return [_make_child(a, **kwargs) for a in survivors] + children


def evolve_population(
    algorithm_name: str,
    final_score,
    agents,
    algorithm_params: dict,
    rng: random.Random | None = None,
) -> list[dict]:
    """Dispatch function for evolutionary update rules."""
    param_bounds            = algorithm_params["param_bounds"]
    default_strategy_params = algorithm_params["default_strategy_params"]

    mutation_kwargs = {
        "mutation_rate":           algorithm_params.get("mutation_rate", 0.02),
        "info_param_mutation_std": algorithm_params.get("info_param_mutation_std", 0.01),
        "info_param_bounds":       algorithm_params.get("info_param_bounds", (0.0, 1.0)),
        "param_bounds":            param_bounds,
        "default_strategy_params": default_strategy_params,
        "frozen_params":           algorithm_params.get("frozen_params", frozenset()),
    }

    if algorithm_name == "truncation":
        return evolve_truncation(
            final_score=final_score, agents=agents,
            top_n=algorithm_params["top_n"],
            bottom_k=algorithm_params["bottom_k"],
            rng=rng, **mutation_kwargs,
        )

    if algorithm_name == "tournament":
        return evolve_tournament(
            final_score=final_score, agents=agents,
            bottom_k=algorithm_params["bottom_k"],
            tournament_size=algorithm_params["tournament_size"],
            rng=rng, **mutation_kwargs,
        )

    if algorithm_name == "fitness_proportionate":
        return evolve_fitness_proportionate(
            final_score=final_score, agents=agents,
            bottom_k=algorithm_params["bottom_k"],
            rng=rng,
            epsilon=algorithm_params.get("epsilon", 1e-9),
            **mutation_kwargs,
        )

    raise ValueError(f"Unknown algorithm_name: {algorithm_name}")


def _validate_strategy_counts(strategy_counts: dict[str, int]) -> None:
    if not isinstance(strategy_counts, dict):
        raise TypeError("strategy_counts must be a dict[str, int]")
    for trader_type, count in strategy_counts.items():
        if trader_type not in VALID_STRATEGIES:
            raise ValueError(f"Unknown strategy type: {trader_type!r}")
        if not isinstance(count, int):
            raise TypeError(f"Count for '{trader_type}' must be an int")
        if count < 0:
            raise ValueError(f"Count for '{trader_type}' cannot be negative")


def _validate_population_spec(population_spec: list[dict]) -> None:
    if not isinstance(population_spec, list):
        raise TypeError("population_spec must be a list[dict]")
    for i, agent in enumerate(population_spec):
        if not isinstance(agent, dict):
            raise TypeError(f"population_spec[{i}] must be a dict")
        if "trader_type" not in agent:
            raise ValueError(f"population_spec[{i}] is missing 'trader_type'")
        if agent["trader_type"] not in VALID_STRATEGIES:
            raise ValueError(f"population_spec[{i}] has unknown trader_type: {agent['trader_type']!r}")
