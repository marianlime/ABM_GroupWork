"""
Evolutionary algorithms (truncation, tournament, fitness-proportionate) and
population-management helpers for the agent-based market simulation.
"""

#--- Imports for evolution.py ---
from collections import Counter
import random
import numpy as np
# --- Imports for evolution.py ---

STRATEGY_ORDER = ("zi", "parameterised_informed") # Stable display and iteration order for strategy types
VALID_STRATEGIES = set(STRATEGY_ORDER) #Trader types for the population spec and strategy_count

def initial_population_from_counts(strategy_counts: dict[str, int], default_strategy_params: dict, param_bounds: dict | None = None, rng: random.Random | None = None) -> list[dict]:

    _validate_strategy_counts(strategy_counts)
    if param_bounds is not None and strategy_counts.get("parameterised_informed", 0) > 0:
        _rng = rng if rng is not None else random.Random()
        informed_params = _make_evenly_spaced_params(strategy_counts["parameterised_informed"], param_bounds, _rng)
    else:
        informed_params = [dict(default_strategy_params) for _ in range(strategy_counts.get("parameterised_informed", 0))]

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

    #Builds list of n strategy_params dicts with evenly spaced values for each parameter within bounds
    #n = number of informed agents to generate params for
    #param_bounds = dict of parameter: (low, high, mutation_std) for each

    if n == 1:
        # If there is only one informed agent, place it in the middle of the parameter space
        return [{param: (low + high) / 2.0 for param, (low, high, mutation_std) in param_bounds.items()}]

    param_columns = {} #Builds dict of parameters
    for param, (low, high, mutation_std) in param_bounds.items(): #For each parameter, generate n evenly spaced values (low -> high)
        values = np.linspace(low, high, n).tolist() #Evenly spaced values for this parameter
        rng.shuffle(values) #Shuffle to avoid correlated parameter values across agents.
        param_columns[param] = values #Store column of values for the parameter

    result = [] #Builds a list of strategy params_dicts for each informed agent.

    for i in range(n): 
        row = {}
        for param in param_bounds:
            row[param] = param_columns[param][i]
        result.append(row)
    return result

def count_strategies(population_spec: list[dict]) -> dict[str, int]:
    _validate_population_spec(population_spec)
    counts = Counter(agent["trader_type"] for agent in population_spec)
    return {strategy: counts.get(strategy, 0) for strategy in STRATEGY_ORDER}


def rank_agents_by_fitness(final_score, agents) -> list[dict]:
    ranked = []
    for agent_id, wealth in final_score:
        if agent_id not in agents:
            raise ValueError(f"agent_id {agent_id} found in final_score but not in agents")
        ranked.append({"agent_id":agent_id,"wealth":float(wealth),"trader_type":agents[agent_id].trader_type,"info_param":float(agents[agent_id].info_param),"strategy_params": dict(agents[agent_id].strategy_params)})
        
    ranked.sort(key=lambda x: x["wealth"], reverse=True)
    return ranked


def _make_child(parent: dict, rng: random.Random, mutation_rate: float, info_param_mutation_std: float, info_param_bounds: tuple[float, float], param_bounds: dict, default_strategy_params: dict, frozen_params: set = frozenset(), second_parent: dict | None = None) -> dict:

    if "info_param" in frozen_params:
        info_param = parent["info_param"]
    else:
        low, high = info_param_bounds
        base_ip = (
            second_parent["info_param"]
            if second_parent is not None and rng.random() < 0.5
            else parent["info_param"]
        )
        info_param = base_ip + rng.gauss(0.0, info_param_mutation_std)
        info_param = max(low, min(high, info_param))

    child: dict = {
        "trader_type": parent["trader_type"],
        "info_param":  info_param,
    }

    if parent["trader_type"] == "parameterised_informed":
        parent_params = parent.get("strategy_params", default_strategy_params)
        second_params = (
            second_parent.get("strategy_params", default_strategy_params)
            if second_parent is not None else None
        )
        mutated_params = {}
        for param, (p_low, p_high, p_std) in param_bounds.items():
            base = (
                second_params.get(param, default_strategy_params[param])
                if second_params is not None and rng.random() < 0.5
                else parent_params.get(param, default_strategy_params[param])
            )
            if rng.random() < mutation_rate:
                val = rng.uniform(p_low, p_high)
            else:
                val = max(p_low, min(p_high, base + rng.gauss(0.0, p_std)))
            mutated_params[param] = val

        child["strategy_params"] = mutated_params

    return child


def evolve_truncation(final_score, agents, top_n: int, bottom_k: int, param_bounds: dict, default_strategy_params: dict, rng: random.Random | None = None, mutation_rate: float = 0.02, info_param_mutation_std: float = 0.01, info_param_bounds: tuple[float, float] = (0.0, 1.0), frozen_params: set = frozenset(), crossover_rate: float = 0.0) -> list[dict]:
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

    kwargs = dict(rng=rng, mutation_rate=mutation_rate, info_param_mutation_std=info_param_mutation_std, info_param_bounds=info_param_bounds, param_bounds=param_bounds, default_strategy_params=default_strategy_params, frozen_params=frozen_params)
    children = []
    
    for _ in range(bottom_k):
        primary = rng.choice(parent_pool)
        second  = None
        if crossover_rate > 0 and rng.random() < crossover_rate and len(parent_pool) > 1:
            others = [p for p in parent_pool if p is not primary]
            second = rng.choice(others) if others else None
        children.append(_make_child(primary, second_parent=second, **kwargs))
    return list(survivors) + children


def evolve_tournament(final_score, agents, bottom_k: int, tournament_size: int, param_bounds: dict, default_strategy_params: dict, rng: random.Random | None = None, mutation_rate: float = 0.02, info_param_mutation_std: float = 0.01, info_param_bounds: tuple[float, float] = (0.0, 1.0), frozen_params: set = frozenset(), crossover_rate: float = 0.0) -> list[dict]:
    
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
        primary = max(rng.sample(ranked, k=tournament_size), key=lambda x: x["wealth"])
        second  = None
        if crossover_rate > 0 and rng.random() < crossover_rate and population_size > 1:
            second = max(rng.sample(ranked, k=tournament_size), key=lambda x: x["wealth"])
        children.append(_make_child(primary, second_parent=second, **kwargs))

    return list(survivors) + children


def evolve_fitness_proportionate(final_score, agents, bottom_k: int, param_bounds: dict, default_strategy_params: dict, rng: random.Random | None = None, epsilon: float = 1e-9, mutation_rate: float = 0.02, info_param_mutation_std: float = 0.01, info_param_bounds: tuple[float, float] = (0.0, 1.0), frozen_params: set = frozenset(), crossover_rate: float = 0.0) -> list[dict]:
    
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

    kwargs = dict(rng=rng, mutation_rate=mutation_rate,info_param_mutation_std=info_param_mutation_std,info_param_bounds=info_param_bounds,param_bounds=param_bounds,default_strategy_params=default_strategy_params, frozen_params=frozen_params)
    children = []

    for _ in range(bottom_k):
        primary = rng.choices(ranked, weights=weights, k=1)[0]
        second  = None
        if crossover_rate > 0 and rng.random() < crossover_rate and population_size > 1:
            second = rng.choices(ranked, weights=weights, k=1)[0]
        children.append(_make_child(primary, second_parent=second, **kwargs))
    return list(survivors) + children


def evolve_population(algorithm_name: str, final_score, agents, algorithm_params: dict, rng: random.Random | None = None) -> list[dict]:

    param_bounds = algorithm_params["param_bounds"]
    default_strategy_params = algorithm_params["default_strategy_params"]

    # Count informed agents to support population-scaled bottom_k / top_n.
    n_informed = sum(1 for a in agents.values() if a.trader_type == "parameterised_informed")

    bottom_k_fraction = algorithm_params.get("bottom_k_fraction")
    
    if bottom_k_fraction is not None:
        bottom_k = max(1, round(n_informed * bottom_k_fraction))
    else:
        bottom_k = algorithm_params["bottom_k"]

    top_n_fraction = algorithm_params.get("top_n_fraction")
    
    if top_n_fraction is not None:
        top_n = max(1, round(n_informed * top_n_fraction))
    else:
        top_n = algorithm_params.get("top_n", bottom_k)

    mutation_kwargs = {
        "mutation_rate":           algorithm_params.get("mutation_rate", 0.02),
        "info_param_mutation_std": algorithm_params.get("info_param_mutation_std", 0.01),
        "info_param_bounds":       algorithm_params.get("info_param_bounds", (0.0, 1.0)),
        "param_bounds":            param_bounds,
        "default_strategy_params": default_strategy_params,
        "frozen_params":           algorithm_params.get("frozen_params", frozenset()),
        "crossover_rate":          algorithm_params.get("crossover_rate", 0.0),
    }

    if algorithm_name == "truncation":
        return evolve_truncation(final_score=final_score, agents=agents, top_n=top_n, bottom_k=bottom_k, rng=rng, **mutation_kwargs)
    if algorithm_name == "tournament":
        return evolve_tournament(final_score=final_score, agents=agents, bottom_k=bottom_k, tournament_size=algorithm_params["tournament_size"], rng=rng, **mutation_kwargs)
    if algorithm_name == "fitness_proportionate":
        return evolve_fitness_proportionate(final_score=final_score, agents=agents, bottom_k=bottom_k, rng=rng, epsilon=algorithm_params.get("epsilon", 1e-9), **mutation_kwargs)
    
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
