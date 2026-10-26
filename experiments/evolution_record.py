"""Put an evolutionary run into the repository, so the rest of the platform can see it.

Until this existed, `evolve` was the one way of producing an agent whose results reached nothing:
`/evolve` stored a genome in the content store and rendered a page, and no ledger record was
written. So lineage detection, the Smart Filter, `/compare` and the meta-analysis -- the whole
layer the platform is built around -- covered every match and no evolution. You could meta-analyse
a Q-learner's thirty seeds and not a single generation of search.

**What gets recorded, and why it is one match rather than the search.** The ledger's record is a
match: one game, one roster, one event log, one set of measures. A search is many matches over a
changing population and does not fit that shape, and bending the schema to hold it would change
what a record means for every record already written. So the search's *outcome* is recorded the
way the platform records any outcome: the fittest genome is frozen, seated against copies of
itself for one evaluation match, and that match is recorded. This is the construction the web
interface's bake-off mode and `run_bakeoff.py` already use, for the same reason -- it separates
what the genome does from how fast it got there.

Self-play for the evaluation is deliberate. Fitness was measured by grouping the population with
its own peers, so a record of the best genome among copies of itself is the same question the GA
was asking, answered once more without selection running.

**What the record carries that an ordinary match does not.** `evolved_by` on each seat names the
search: architecture, generations, population, mutation, selection and the lifetime-learning
reading. It reaches `config_hash` through `_PARAM_ATTRS`, so two genomes found by searches that
differ are two designs, and a genome the GA found never shares an identity with one gradient
descent found. It costs no existing record its hash, because `_agent_params` reads every attribute
under `hasattr` and a trained seat has no such attribute.

**What is deliberately not claimed.** The generation-by-generation history is not in the record.
It is in the event log that `run_evolution.py` writes, keyed by the same run, and a reader who
wants the trajectory reads that. Putting a summary of it in the record would invite comparing two
searches by numbers the record was not built to make comparable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..engine.eventlog import EventLog
from ..engine.evolution import EvolutionConfig, EvolutionResult
from ..engine.game import Game
from ..engine.runner import run_match
from ..metrics import social
from ..repository.record import protocol_from_run, record_experiment


def provenance(architecture: str, config: EvolutionConfig, generations_run: int) -> dict[str, Any]:
    """The search, as the fields that would change what it searches.

    `n_hidden` is left out on purpose: it is already a genome field for the one architecture that
    reads it, and restating it here would put the same number in a record twice under two names.
    """
    return {
        "architecture": architecture,
        "generations": generations_run,
        "population_size": config.population_size,
        "matches_per_generation": config.matches_per_generation,
        "match_rounds": config.match_rounds,
        "mutation_rate": config.mutation_rate,
        "mutation_scale": config.mutation_scale,
        "tournament_size": config.tournament_size,
        "elite_count": config.elite_count,
        "lifetime_learning": config.lifetime_learning,
        "ga_seed": config.seed,
    }


def record_evolution(
    repo_root: str | Path,
    result: EvolutionResult,
    config: EvolutionConfig,
    architecture: str,
    game_factory: Callable[[], Game],
    log_path: str | Path,
    eval_rounds: int | None = None,
    eval_seed: int | None = None,
) -> dict[str, Any]:
    """Seat the fittest genome against copies of itself, play once, and append the record.

    `eval_rounds` defaults to the config's own match length, so the recorded match is the same
    length as the matches fitness was measured over. `eval_seed` defaults to the GA seed offset
    past the range `evolve` draws its match seeds from, so an evaluation never silently replays a
    grouping the search already saw.
    """
    rounds = eval_rounds or config.match_rounds
    seed = config.seed + 9000 if eval_seed is None else eval_seed
    game = game_factory()

    story = provenance(architecture, config, len(result.fitness_history))
    roster: list[Any] = []
    for i in range(game.n_agents):
        twin = result.best.clone(name=f"{config.name_prefix}best{i}", seed=seed * 100 + i)
        # The evaluation measures the genome, so nothing learns during it. Under the Darwinian
        # default this is already true; under `"lamarckian"` the individual would otherwise keep
        # training here and the recorded numbers would describe neither the genome nor the search.
        if hasattr(twin, "lifetime_learning"):
            twin.lifetime_learning = False
        twin.evolved_by = story
        roster.append(twin)

    with EventLog(path=log_path) as log:
        log.meta(game=game.describe(), config=vars(config), evolved_by=story)
        run_result = run_match(game, roster, rounds=rounds, seed=seed, eventlog=log)

    metrics = social.compute_all(run_result, game.max_welfare_per_round())
    return record_experiment(
        repo_root, game, roster, log_path, rounds=rounds, seed=seed, metrics=metrics,
        protocol=protocol_from_run(run_result),
    )
