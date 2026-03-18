# Run via pytest in terminal

import pytest
import random
from collections import Counter

from sim.evolution import (
    initial_population_from_counts,
    count_strategies,
    rank_agents_by_fitness,
    evolve_truncation,
    evolve_tournament,
    evolve_fitness_proportionate,
    evolve_population,
    STRATEGY_ORDER,
)


PARAM_BOUNDS = {
    "qty_aggression":    (0.1,  5.0, 0.20),
    "signal_aggression": (0.0,  1.0, 0.05),
    "threshold":         (0.0,  0.50, 0.03),
}
DEFAULTS = {"qty_aggression": 1.0, "signal_aggression": 1.0, "threshold": 0.0}


class FakeAgent:
    """Stand-in for Trader — only needs the fields rank_agents_by_fitness reads."""
    def __init__(self, ttype, ip=0.3, sparams=None):
        self.trader_type = ttype
        self.info_param = ip
        self.strategy_params = sparams or dict(DEFAULTS)


def _make_scores_and_agents(data):
    """Build (final_score, agents) from a list of (id, type, wealth) tuples."""
    agents, scores = {}, []
    for aid, ttype, w in data:
        agents[aid] = FakeAgent(ttype)
        scores.append((aid, w))
    return scores, agents


# Population creation

class TestInitialPopulation:

    def test_total_count(self):
        pop = initial_population_from_counts({"zi": 3, "parameterised_informed": 2}, DEFAULTS)
        assert len(pop) == 5

    def test_type_counts_match(self):
        pop = initial_population_from_counts({"zi": 4, "parameterised_informed": 6}, DEFAULTS)
        types = Counter(a["trader_type"] for a in pop)
        assert types["zi"] == 4 and types["parameterised_informed"] == 6

    def test_informed_agents_get_strategy_params(self):
        pop = initial_population_from_counts(
            {"parameterised_informed": 3}, DEFAULTS,
            param_bounds=PARAM_BOUNDS, rng=random.Random(42),
        )
        for a in pop:
            assert "strategy_params" in a

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            initial_population_from_counts({"fake_strat": 5}, DEFAULTS)

    def test_negative_count_raises(self):
        with pytest.raises(ValueError):
            initial_population_from_counts({"zi": -1}, DEFAULTS)


# Counting

class TestCountStrategies:

    def test_basic(self):
        pop = [{"trader_type": "zi"}, {"trader_type": "zi"}, {"trader_type": "parameterised_informed"}]
        c = count_strategies(pop)
        assert c["zi"] == 2 and c["parameterised_informed"] == 1

    def test_empty(self):
        c = count_strategies([])
        assert all(v == 0 for v in c.values())


# Ranking

class TestRankAgents:

    def test_descending_order(self):
        sc, ag = _make_scores_and_agents([(1, "zi", 10), (2, "zi", 30), (3, "zi", 20)])
        ranked = rank_agents_by_fitness(sc, ag)
        wealths = [r["wealth"] for r in ranked]
        assert wealths == sorted(wealths, reverse=True)

    def test_missing_agent_raises(self):
        agents = {1: FakeAgent("zi")}
        with pytest.raises(ValueError):
            rank_agents_by_fitness([(1, 10), (99, 5)], agents)


# Truncation selection

FIVE_INFORMED = [
    (1, "parameterised_informed", 50),
    (2, "parameterised_informed", 40),
    (3, "parameterised_informed", 30),
    (4, "parameterised_informed", 20),
    (5, "parameterised_informed", 10),
]

class TestTruncation:

    def test_population_size_preserved(self):
        sc, ag = _make_scores_and_agents(FIVE_INFORMED)
        out = evolve_truncation(sc, ag, top_n=3, bottom_k=2,
                                param_bounds=PARAM_BOUNDS, default_strategy_params=DEFAULTS,
                                rng=random.Random(42))
        assert len(out) == 5

    def test_children_have_required_keys(self):
        sc, ag = _make_scores_and_agents(FIVE_INFORMED)
        out = evolve_truncation(sc, ag, top_n=3, bottom_k=2,
                                param_bounds=PARAM_BOUNDS, default_strategy_params=DEFAULTS,
                                rng=random.Random(42))
        for child in out:
            assert "trader_type" in child and "info_param" in child

    def test_reproducible_with_same_seed(self):
        sc, ag = _make_scores_and_agents(FIVE_INFORMED)
        r1 = evolve_truncation(sc, ag, top_n=3, bottom_k=2,
                               param_bounds=PARAM_BOUNDS, default_strategy_params=DEFAULTS,
                               rng=random.Random(99))
        r2 = evolve_truncation(sc, ag, top_n=3, bottom_k=2,
                               param_bounds=PARAM_BOUNDS, default_strategy_params=DEFAULTS,
                               rng=random.Random(99))
        assert r1 == r2

    def test_empty_population_raises(self):
        with pytest.raises(ValueError):
            evolve_truncation([], {}, top_n=1, bottom_k=0,
                              param_bounds=PARAM_BOUNDS, default_strategy_params=DEFAULTS)


# Tournament selection

FOUR_INFORMED = [
    (1, "parameterised_informed", 50),
    (2, "parameterised_informed", 40),
    (3, "parameterised_informed", 30),
    (4, "parameterised_informed", 20),
]

class TestTournament:

    def test_population_size_preserved(self):
        sc, ag = _make_scores_and_agents(FOUR_INFORMED)
        out = evolve_tournament(sc, ag, bottom_k=2, tournament_size=2,
                                param_bounds=PARAM_BOUNDS, default_strategy_params=DEFAULTS,
                                rng=random.Random(42))
        assert len(out) == 4

    def test_invalid_tournament_size_raises(self):
        sc, ag = _make_scores_and_agents(FOUR_INFORMED)
        with pytest.raises(ValueError):
            evolve_tournament(sc, ag, bottom_k=2, tournament_size=0,
                              param_bounds=PARAM_BOUNDS, default_strategy_params=DEFAULTS)


# Fitness-proportionate (roulette wheel) selection

class TestFitnessProportionate:

    def test_population_size_preserved(self):
        sc, ag = _make_scores_and_agents(FOUR_INFORMED)
        out = evolve_fitness_proportionate(sc, ag, bottom_k=2,
                                           param_bounds=PARAM_BOUNDS,
                                           default_strategy_params=DEFAULTS,
                                           rng=random.Random(42))
        assert len(out) == 4

    def test_handles_negative_wealth(self):
        sc, ag = _make_scores_and_agents([
            (1, "parameterised_informed", -10),
            (2, "parameterised_informed", 5),
            (3, "parameterised_informed", 20),
        ])
        out = evolve_fitness_proportionate(sc, ag, bottom_k=1,
                                           param_bounds=PARAM_BOUNDS,
                                           default_strategy_params=DEFAULTS,
                                           rng=random.Random(42))
        assert len(out) == 3


# Dispatch via evolve_population

ALG_PARAMS = {
    "top_n": 2, "bottom_k": 1, "tournament_size": 2,
    "param_bounds": PARAM_BOUNDS, "default_strategy_params": DEFAULTS,
}

THREE_INFORMED = [
    (1, "parameterised_informed", 50),
    (2, "parameterised_informed", 40),
    (3, "parameterised_informed", 30),
]

class TestDispatch:

    def test_truncation(self):
        sc, ag = _make_scores_and_agents(THREE_INFORMED)
        out = evolve_population("truncation", sc, ag, ALG_PARAMS, rng=random.Random(1))
        assert len(out) == 3

    def test_tournament(self):
        sc, ag = _make_scores_and_agents(THREE_INFORMED)
        out = evolve_population("tournament", sc, ag, ALG_PARAMS, rng=random.Random(1))
        assert len(out) == 3

    def test_unknown_algorithm_raises(self):
        sc, ag = _make_scores_and_agents(THREE_INFORMED)
        with pytest.raises(ValueError):
            evolve_population("made_up", sc, ag, ALG_PARAMS)
