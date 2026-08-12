"""
Generational genetic algorithm over any evolvable agent, not just Markov brains.

Fitness = average payoff earned when a genome plays a full match of the given game against other
members of the population (several random groupings per generation, self-play, for a less noisy
estimate). Selection: tournament selection. Mutation and crossover are the genome's own business.
Elitism: top performers carry over unchanged.

**This module imports no agent.** It used to `from ..agents.markov_brain import ...`, which made the
engine depend upward on the agents layer and quietly falsified the architecture's own central claim,
that the engine can be read and tested without any game, agent or metric present. Callers now supply
`spawn` and `crossover`, so the GA knows only the `Evolvable` protocol below. `agents/markov_brain.py`
provides both for its own genome, which is where that knowledge belongs.

Deliberately does NOT compute Φ/autonomy during evolution -- PyPhi's major-complex search is far
too slow to run every generation for every individual. Call `metrics.phi_autonomy` directly on
`EvolutionResult.best` (or any other returned genome) after evolution finishes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

import numpy as np

from .console import LiveConsole
from .eventlog import EventLog
from .game import Game
from .runner import run_match


class Evolvable(Protocol):
    """What the GA needs of a genome. Any agent offering these can be evolved by this module."""

    name: str

    def reset_state(self) -> None:
        """Clear per-match internal state so fitness is not inherited from the previous grouping."""

    def clone(self, name: str, seed: int) -> "Evolvable":
        """An identical genome under a new name and RNG seed. Used to carry elites over."""

    def mutate(self, rng: np.random.Generator, rate: float, scale: float,
               name: str, seed: int) -> "Evolvable":
        """A perturbed copy of this genome."""


#: Build one random individual. Takes the config too, so genome-specific knobs (`n_hidden` for a
#: Markov brain) are read by the spawner rather than interpreted by the engine.
SpawnFn = Callable[[str, Game, int, "EvolutionConfig"], Evolvable]

#: Combine two parents into a child.
CrossoverFn = Callable[..., Evolvable]


@dataclass
class EvolutionConfig:
    population_size: int = 20
    generations: int = 30
    match_rounds: int = 200
    matches_per_generation: int = 3
    n_hidden: int = 2       # genome-specific; read by the spawner, not by this module
    tournament_size: int = 3
    mutation_rate: float = 0.15
    mutation_scale: float = 0.4
    elite_count: int = 2
    seed: int = 0
    #: Prefix for individual names. Default keeps historical "MB0, MB1, ..." naming.
    name_prefix: str = "MB"


@dataclass
class EvolutionResult:
    population: list[Any]
    fitness: np.ndarray
    best: Any
    fitness_history: list[dict] = field(default_factory=list)


def _evaluate_population(
    population: list[Evolvable],
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
    population: list[Evolvable],
    fitness: np.ndarray,
    config: EvolutionConfig,
    rng: np.random.Generator,
    crossover: CrossoverFn,
) -> list[Evolvable]:
    order = np.argsort(fitness)[::-1]
    next_gen: list[Evolvable] = []

    for rank in range(config.elite_count):
        parent = population[order[rank]]
        next_gen.append(parent.clone(name=f"{config.name_prefix}{len(next_gen)}",
                                     seed=int(rng.integers(1 << 30))))

    def tournament_pick() -> Evolvable:
        contenders = rng.choice(len(population), size=config.tournament_size, replace=False)
        best_idx = contenders[np.argmax(fitness[contenders])]
        return population[best_idx]

    while len(next_gen) < config.population_size:
        parent_a, parent_b = tournament_pick(), tournament_pick()
        child = crossover(parent_a, parent_b, rng, name=f"{config.name_prefix}{len(next_gen)}",
                          seed=int(rng.integers(1 << 30)))
        child = child.mutate(rng, rate=config.mutation_rate, scale=config.mutation_scale,
                             name=child.name, seed=int(rng.integers(1 << 30)))
        next_gen.append(child)

    return next_gen


def evolve(
    game_factory: Callable[[], Game],
    config: EvolutionConfig,
    spawn: SpawnFn,
    crossover: CrossoverFn,
    console: Optional[LiveConsole] = None,
    eventlog: Optional[EventLog] = None,
) -> EvolutionResult:
    """Run the GA. `game_factory()` must return a fresh Game whose `n_agents` evenly divides
    `config.population_size` (the population is grouped into matches of `game.n_agents` seats).

    `spawn` and `crossover` are required rather than defaulted, deliberately: a default would mean
    importing a concrete agent here, which is the upward dependency this signature exists to remove.
    For Markov brains, pass `agents.markov_brain.spawn` and `agents.markov_brain.crossover`.
    """
    rng = np.random.default_rng(config.seed)
    game0 = game_factory()
    if config.population_size % game0.n_agents != 0:
        raise ValueError(
            f"population_size ({config.population_size}) must be a multiple of "
            f"game.n_agents ({game0.n_agents})"
        )

    population = [
        spawn(f"{config.name_prefix}{i}", game0, config.seed * 10_000 + i, config)
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
            population = _next_generation(population, fitness, config, rng, crossover)

    order = np.argsort(fitness)[::-1]
    return EvolutionResult(population=population, fitness=fitness,
                           best=population[order[0]], fitness_history=history)
