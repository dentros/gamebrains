"""
Generational genetic algorithm for evolving MarkovBrainAgent populations.

Fitness = average payoff earned when a genome plays a full match of the given game against other
members of the population (several random groupings per generation, self-play, for a less noisy
estimate). Selection: tournament selection. Mutation: Gaussian perturbation of weights/bias.
Crossover: uniform per-element mixing. Elitism: top performers carry over unchanged.

Deliberately does NOT compute Φ/autonomy during evolution -- PyPhi's major-complex search is far
too slow to run every generation for every individual. Call `metrics.phi_autonomy` directly on
`EvolutionResult.best` (or any other returned genome) after evolution finishes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..agents.markov_brain import MarkovBrainAgent, crossover
from .console import LiveConsole
from .eventlog import EventLog
from .game import Game
from .runner import run_match


@dataclass
class EvolutionConfig:
    population_size: int = 20
    generations: int = 30
    match_rounds: int = 200
    matches_per_generation: int = 3
    n_hidden: int = 2
    tournament_size: int = 3
    mutation_rate: float = 0.15
    mutation_scale: float = 0.4
    elite_count: int = 2
    seed: int = 0


@dataclass
class EvolutionResult:
    population: list[MarkovBrainAgent]
    fitness: np.ndarray
    best: MarkovBrainAgent
    fitness_history: list[dict] = field(default_factory=list)


def _evaluate_population(
    population: list[MarkovBrainAgent],
    game_factory: Callable[[], Game],
    config: EvolutionConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(population)
    n_agents = game_factory().n_agents

    payoff_sum = np.zeros(n)
    match_count = np.zeros(n)

    for _ in range(config.matches_per_generation):
        order = rng.permutation(n)
        for start in range(0, n - n_agents + 1, n_agents):
            group_idx = order[start:start + n_agents]
            group = [population[i] for i in group_idx]
            for agent in group:
                agent.reset_state()

            game = game_factory()
            match_seed = int(rng.integers(1 << 30))
            records = run_match(game, group, rounds=config.match_rounds, seed=match_seed)
            payoffs = records["rewards"].sum(axis=0)

            for i, idx in enumerate(group_idx):
                payoff_sum[idx] += payoffs[i]
                match_count[idx] += 1

    match_count[match_count == 0] = 1  # guard; shouldn't happen given the divisibility check
    return payoff_sum / match_count


def _next_generation(
    population: list[MarkovBrainAgent],
    fitness: np.ndarray,
    config: EvolutionConfig,
    rng: np.random.Generator,
) -> list[MarkovBrainAgent]:
    order = np.argsort(fitness)[::-1]
    next_gen: list[MarkovBrainAgent] = []

    for rank in range(config.elite_count):
        parent = population[order[rank]]
        next_gen.append(parent.clone(name=f"MB{len(next_gen)}", seed=int(rng.integers(1 << 30))))

    def tournament_pick() -> MarkovBrainAgent:
        contenders = rng.choice(len(population), size=config.tournament_size, replace=False)
        best_idx = contenders[np.argmax(fitness[contenders])]
        return population[best_idx]

    while len(next_gen) < config.population_size:
        parent_a, parent_b = tournament_pick(), tournament_pick()
        child = crossover(parent_a, parent_b, rng, name=f"MB{len(next_gen)}",
                          seed=int(rng.integers(1 << 30)))
        child = child.mutate(rng, rate=config.mutation_rate, scale=config.mutation_scale,
                             name=child.name, seed=int(rng.integers(1 << 30)))
        next_gen.append(child)

    return next_gen


def evolve(
    game_factory: Callable[[], Game],
    config: EvolutionConfig,
    console: Optional[LiveConsole] = None,
    eventlog: Optional[EventLog] = None,
) -> EvolutionResult:
    """Run the GA. `game_factory()` must return a fresh Game whose `n_agents` evenly divides
    `config.population_size` (the population is grouped into matches of `game.n_agents` seats).
    """
    rng = np.random.default_rng(config.seed)
    game0 = game_factory()
    if config.population_size % game0.n_agents != 0:
        raise ValueError(
            f"population_size ({config.population_size}) must be a multiple of "
            f"game.n_agents ({game0.n_agents})"
        )

    population = [
        MarkovBrainAgent.random(f"MB{i}", game0.n_states, game0.n_actions,
                                n_hidden=config.n_hidden, seed=config.seed * 10_000 + i,
                                start_state=game0.start_state)
        for i in range(config.population_size)
    ]

    history: list[dict] = []
    fitness = np.zeros(config.population_size)

    for gen in range(config.generations):
        fitness = _evaluate_population(population, game_factory, config, rng)
        avg, best, worst = float(fitness.mean()), float(fitness.max()), float(fitness.min())
        history.append({"generation": gen, "avg": avg, "best": best, "worst": worst})

        if console:
            console.generation(gen, avg, best, worst)
        if eventlog:
            eventlog.log({"type": "generation", "generation": gen, "avg_fitness": avg,
                          "best_fitness": best, "worst_fitness": worst,
                          "fitness": [float(f) for f in fitness]})

        if gen < config.generations - 1:
            population = _next_generation(population, fitness, config, rng)

    order = np.argsort(fitness)[::-1]
    return EvolutionResult(population=population, fitness=fitness,
                           best=population[order[0]], fitness_history=history)
