"""
Tests for engine.evolution's generational GA over MarkovBrainAgent populations.

Deliberately does not touch metrics.phi_autonomy (Phi/autonomy are computed only on-demand after
evolution finishes, never inside the loop) -- these tests are fast and check structural
correctness and reproducibility, not the evolutionary outcome itself (that is validated
separately, empirically, in experiments/run_evolution.py: fitness converges toward the
pygambit-confirmed Nash equilibrium of universal defection).
"""

import numpy as np

from gamebrains.agents.markov_brain import crossover as mb_crossover, spawn as mb_spawn
from gamebrains.engine.evolution import EvolutionConfig, evolve
from gamebrains.games.public_goods import PublicGoodsGame


def _small_game_factory():
    return PublicGoodsGame(n_agents=4, rounds=40)


def test_rejects_population_not_divisible_by_n_agents():
    config = EvolutionConfig(population_size=10, generations=1)  # 10 % 4 != 0
    try:
        evolve(_small_game_factory, config, spawn=mb_spawn, crossover=mb_crossover)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_runs_and_produces_well_shaped_result():
    config = EvolutionConfig(population_size=8, generations=4, match_rounds=40,
                             matches_per_generation=2, seed=0)
    result = evolve(_small_game_factory, config, spawn=mb_spawn, crossover=mb_crossover)

    assert len(result.population) == config.population_size
    assert result.fitness.shape == (config.population_size,)
    assert np.all(np.isfinite(result.fitness))
    assert len(result.fitness_history) == config.generations
    assert result.best is result.population[int(np.argmax(result.fitness))]


def test_reproducible_given_seed():
    config = EvolutionConfig(population_size=8, generations=4, match_rounds=40,
                             matches_per_generation=2, seed=7)
    r1 = evolve(_small_game_factory, config, spawn=mb_spawn, crossover=mb_crossover)
    r2 = evolve(_small_game_factory, config, spawn=mb_spawn, crossover=mb_crossover)

    assert np.allclose(r1.fitness, r2.fitness)
    for h1, h2 in zip(r1.fitness_history, r2.fitness_history):
        assert h1 == h2


def test_elite_genomes_carry_over_unchanged():
    config = EvolutionConfig(population_size=8, generations=1, match_rounds=40,
                             matches_per_generation=2, elite_count=2, seed=0)
    # Run generation 0 manually via the same internals evolve() uses, to check elitism directly.
    from gamebrains.engine.evolution import _evaluate_population, _next_generation

    rng = np.random.default_rng(config.seed)
    game0 = _small_game_factory()
    population = [
        mb_spawn(f"{config.name_prefix}{i}", game0, config.seed * 10_000 + i, config)
        for i in range(config.population_size)
    ]
    fitness = _evaluate_population(population, _small_game_factory, config, rng)
    order = np.argsort(fitness)[::-1]
    best_genome_W = population[order[0]].W.copy()

    next_gen = _next_generation(population, fitness, config, rng, mb_crossover)
    assert np.array_equal(next_gen[0].W, best_genome_W)  # top elite preserved exactly


if __name__ == "__main__":
    test_rejects_population_not_divisible_by_n_agents()
    print("OK: rejects mismatched population/n_agents")
    test_runs_and_produces_well_shaped_result()
    print("OK: runs and produces well-shaped EvolutionResult")
    test_reproducible_given_seed()
    print("OK: reproducible given the same seed")
    test_elite_genomes_carry_over_unchanged()
    print("OK: elite genomes carry over unchanged")
