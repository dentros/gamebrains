"""The GA over genomes that are not Markov brains, and the question that only they raise.

`engine.evolution` already takes `spawn` and `crossover` as arguments, so it has been
architecture-agnostic since the day the upward import was removed. What it had never been given
is a genome that *learns while it is being evaluated*. A Markov brain does not: its `update` is
not a learning rule, so a match leaves the genome exactly as selection found it.

A Q-table does. `engine/runner.py:96` calls `update` on every agent every round, so a Q-learner
placed in the population would rewrite its own genome during the fitness match, and
`_next_generation` would then `clone` the rewritten table into the next generation. That is
Lamarckian inheritance, arrived at by nobody's decision, and it would be invisible in the output:
fitness would simply be better than the GA earned. These tests pin it down as a choice with a
name, defaulting to the Darwinian reading, and they fail loudly if the default ever drifts.

The second thing they hold is that the new knob cannot disturb what already runs. A Markov brain
is indifferent to the setting by construction, so evolving one under either value has to give the
identical answer, and that equality is asserted rather than assumed.
"""

import sys

import numpy as np

from gamebrains.agents.markov_brain import crossover as mb_crossover, spawn as mb_spawn
from gamebrains.agents.qlearning import (
    QLearningAgent,
    crossover as ql_crossover,
    spawn as ql_spawn,
)
from gamebrains.engine.evolution import EvolutionConfig, evolve
from gamebrains.engine.runner import run_match
from gamebrains.games.public_goods import PublicGoodsGame


def _game(rounds: int = 40) -> PublicGoodsGame:
    return PublicGoodsGame(n_agents=4, rounds=rounds)


def _darwinian() -> EvolutionConfig:
    return EvolutionConfig(population_size=8, generations=2, match_rounds=20,
                           matches_per_generation=1, seed=7, name_prefix="QL")


def test_the_default_is_darwinian() -> None:
    """A default nobody set is how the silent inheritance would have arrived."""
    assert EvolutionConfig().lifetime_learning == "off", (
        "EvolutionConfig.lifetime_learning must default to 'off'. The default is the whole "
        "safeguard: a genome that learns during its own fitness match inherits what it learned, "
        "and no caller would have asked for that")


def test_an_unknown_learning_mode_is_refused() -> None:
    config = EvolutionConfig(population_size=8, generations=1, match_rounds=10,
                             lifetime_learning="sometimes")
    try:
        evolve(_game, config, spawn=ql_spawn, crossover=ql_crossover)
        assert False, "expected ValueError for an unrecognised lifetime_learning value"
    except ValueError as exc:
        assert "lifetime_learning" in str(exc), f"the message should name the setting: {exc}"
    print("OK: an unrecognised learning mode is refused by name")


def test_a_spawned_population_carries_variation() -> None:
    """A population of identical genomes gives selection nothing to act on."""
    config = _darwinian()
    game = _game()
    genomes = [ql_spawn(f"QL{i}", game, seed=i, config=config).Q for i in range(4)]
    assert not all(np.array_equal(genomes[0], g) for g in genomes[1:]), (
        "every spawned Q-table is identical, so the first generation has no variation and the "
        "GA is selecting over nothing. The table is the genome here and has to be initialised "
        "at random, the way MarkovBrainAgent.random does it")
    print("OK: spawning gives a varied population")


def test_clone_is_the_same_policy() -> None:
    config = _darwinian()
    game = _game()
    parent = ql_spawn("QL0", game, seed=3, config=config)
    child = parent.clone(name="QL1", seed=99)
    assert np.array_equal(parent.Q, child.Q), "clone must carry the genome unchanged"
    for observation in range(game.n_states):
        assert parent.act(observation) == child.act(observation), (
            f"clone disagrees with its parent at observation {observation}; an elite carried "
            f"into the next generation must be the same policy, not merely a similar one")
    print("OK: a clone is the same policy as its parent")


def test_mutate_perturbs_the_table_and_keeps_its_shape() -> None:
    config = _darwinian()
    parent = ql_spawn("QL0", _game(), seed=3, config=config)
    child = parent.mutate(np.random.default_rng(0), rate=1.0, scale=0.5, name="QL1", seed=1)
    assert child.Q.shape == parent.Q.shape, "mutation must not change the genome's shape"
    assert not np.array_equal(child.Q, parent.Q), "mutation at rate 1.0 changed nothing"
    assert np.array_equal(parent.Q, ql_spawn("QL0", _game(), seed=3, config=config).Q), (
        "mutate altered the parent in place; it must return a perturbed copy")
    print("OK: mutation perturbs a copy and preserves the shape")


def test_crossover_refuses_genomes_of_different_shape() -> None:
    """The same refusal MarkovBrainAgent makes, for the same reason."""
    config = _darwinian()
    a = ql_spawn("QL0", PublicGoodsGame(n_agents=4, rounds=10), seed=1, config=config)
    b = ql_spawn("QL1", PublicGoodsGame(n_agents=6, rounds=10), seed=2, config=config)
    try:
        ql_crossover(a, b, np.random.default_rng(0), name="QL2", seed=3)
        assert False, "expected ValueError for mismatched genome shapes"
    except ValueError:
        pass
    print("OK: crossover refuses parents whose genomes are different shapes")


def test_every_crossed_cell_comes_from_one_parent_or_the_other() -> None:
    config = _darwinian()
    game = _game()
    a = ql_spawn("QL0", game, seed=1, config=config)
    b = ql_spawn("QL1", game, seed=2, config=config)
    child = ql_crossover(a, b, np.random.default_rng(0), name="QL2", seed=3)
    from_a = child.Q == a.Q
    from_b = child.Q == b.Q
    assert np.all(from_a | from_b), (
        "a crossed genome holds a value neither parent had. Uniform crossover recombines, it "
        "does not average: an averaged table is a genome that never existed in the population")
    print("OK: every cell of a crossed genome came from one of the two parents")


def test_darwinian_evaluation_does_not_teach() -> None:
    """The point of the whole module. A fitness match must not rewrite the genome."""
    config = _darwinian()
    game = _game(rounds=30)
    roster = [ql_spawn(f"QL{i}", game, seed=i, config=config) for i in range(game.n_agents)]
    before = [agent.Q.copy() for agent in roster]

    run_match(game, roster, rounds=30, seed=5)

    for agent, original in zip(roster, before):
        assert np.array_equal(agent.Q, original), (
            f"{agent.name} learned during its own fitness match. Under the default the genome is "
            f"what selection acts on, so a match that rewrites it makes fitness measure the "
            f"learning rule instead of the genome, and clone() then inherits the result")
    print("OK: under the default a fitness match leaves the genome untouched")


def test_lamarckian_is_available_and_does_teach() -> None:
    """The opposite is a legitimate experiment. It has to be asked for by name."""
    config = EvolutionConfig(population_size=8, generations=2, match_rounds=20,
                             matches_per_generation=1, seed=7, lifetime_learning="lamarckian")
    game = _game(rounds=30)
    roster = [ql_spawn(f"QL{i}", game, seed=i, config=config) for i in range(game.n_agents)]
    before = [agent.Q.copy() for agent in roster]

    run_match(game, roster, rounds=30, seed=5)

    assert any(not np.array_equal(agent.Q, original)
               for agent, original in zip(roster, before)), (
        "under lifetime_learning='lamarckian' the agents are supposed to learn during the match "
        "and pass on what they learned; none of them changed")
    print("OK: the Lamarckian mode learns during evaluation, when asked for by name")


def test_a_qlearning_run_is_reproducible() -> None:
    first = evolve(_game, _darwinian(), spawn=ql_spawn, crossover=ql_crossover)
    second = evolve(_game, _darwinian(), spawn=ql_spawn, crossover=ql_crossover)
    assert np.array_equal(first.fitness, second.fitness), (
        "two runs at the same seed disagree; the GA over Q-tables is not reproducible")
    print("OK: evolving Q-tables is reproducible at a fixed seed")


def test_the_new_knob_cannot_disturb_markov_brains() -> None:
    """A Markov brain has no learning rule, so the setting must be a no-op for it."""
    def run(mode: str) -> np.ndarray:
        config = EvolutionConfig(population_size=8, generations=2, match_rounds=20,
                                 matches_per_generation=1, seed=11, lifetime_learning=mode)
        return evolve(_game, config, spawn=mb_spawn, crossover=mb_crossover).fitness

    assert np.array_equal(run("off"), run("lamarckian")), (
        "the lifetime_learning setting changed a Markov-brain run. It has no learning rule, so "
        "either value has to give the identical answer; if it does not, the knob is reaching "
        "something it was never meant to touch and every stored Markov-brain result is in doubt")
    print("OK: the new setting leaves Markov-brain evolution exactly as it was")


def test_an_evolved_qlearner_is_not_called_q_learning() -> None:
    """Naming, because the comparison table is only honest if it says what it compared."""
    config = _darwinian()
    agent = ql_spawn("QL0", _game(), seed=1, config=config)
    assert agent.kind == "qlearning", "the architecture is still the tabular one"
    assert agent.learning_rule == "evolved", (
        "a Q-table found by a genetic algorithm was not produced by Q-learning, and a bake-off "
        "row that calls it Q-learning is comparing the substrate while naming the learning rule")
    print("OK: an evolved tabular policy says so, separately from its architecture")


def test_the_registry_names_what_can_be_evolved_and_refuses_the_rest() -> None:
    from gamebrains.agents.factory import EVOLVABLE_KINDS, evolvable

    assert EVOLVABLE_KINDS[0] == "markov_brain", (
        "markov_brain must stay first; it is the default everywhere and the ordering is what "
        "keeps adding architectures from changing an existing run")
    for kind in EVOLVABLE_KINDS:
        spawn, crossover = evolvable(kind)
        assert callable(spawn) and callable(crossover), f"{kind} returned something uncallable"
    for kind in ("fep", "classic", "llm"):
        try:
            evolvable(kind)
            assert False, f"{kind} has no genome and should have been refused"
        except ValueError as exc:
            assert "evolvable" in str(exc).lower() or "genome" in str(exc).lower(), (
                f"the refusal should say why {kind} has nothing to evolve: {exc}")
    print("OK: the registry names the three evolvable kinds and refuses the others by reason")


def test_listing_what_is_evolvable_does_not_require_torch() -> None:
    """A machine without torch must still be able to ask what can be evolved."""
    import subprocess
    import sys

    probe = (
        "import sys\n"
        "class Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        return self if name == 'torch' or name.startswith('torch.') else None\n"
        "    def load_module(self, name):\n"
        "        raise ImportError('torch blocked for this probe')\n"
        "sys.meta_path.insert(0, Block())\n"
        "from gamebrains.agents.factory import EVOLVABLE_KINDS, evolvable\n"
        "evolvable('qlearning')\n"
        "assert 'torch' not in sys.modules, 'importing the registry pulled in torch'\n"
        "print('clean')\n"
    )
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert "clean" in done.stdout, (
        f"the registry could not be used without torch:\n{done.stdout}\n{done.stderr}")
    print("OK: the registry and the tabular genome work with torch unavailable")


def test_networks_evolve_and_reproduce() -> None:
    from gamebrains.agents.dqn import crossover as dqn_crossover, spawn as dqn_spawn

    config = EvolutionConfig(population_size=8, generations=2, match_rounds=20,
                             matches_per_generation=1, seed=3, name_prefix="DQ")
    first = evolve(_game, config, spawn=dqn_spawn, crossover=dqn_crossover)
    second = evolve(_game, config, spawn=dqn_spawn, crossover=dqn_crossover)
    assert np.array_equal(first.fitness, second.fitness), (
        "two network runs at the same seed disagree. The GA's own generator drives mutation "
        "precisely so this does not depend on torch's global state")
    assert first.best.learning_rule == "evolved", (
        "a network found by a genetic algorithm took no gradient step and is not DQN")
    print("OK: networks evolve reproducibly and do not claim to be DQN")


def test_a_cloned_network_carries_its_target_too() -> None:
    """The lagged copy is part of the inheritance, not scratch space."""
    from gamebrains.agents.dqn import spawn as dqn_spawn

    config = EvolutionConfig(population_size=8, generations=1, match_rounds=10,
                             lifetime_learning="lamarckian")
    parent = dqn_spawn("DQ0", _game(), seed=4, config=config)
    child = parent.clone(name="DQ1", seed=5)
    for key, tensor in parent.policy.state_dict().items():
        assert np.allclose(tensor.detach().cpu().numpy(),
                           child.target.state_dict()[key].detach().cpu().numpy()), (
            f"the cloned agent's target net disagrees with the inherited policy at {key}. Left "
            f"at its own random initialisation it would make a cloned elite diverge from its "
            f"parent the moment it trained")
    print("OK: a cloned network inherits its target net as well as its policy")


def test_a_darwinian_network_does_not_train_during_evaluation() -> None:
    from gamebrains.agents.dqn import spawn as dqn_spawn

    config = EvolutionConfig(population_size=8, generations=1, match_rounds=10)
    game = _game(rounds=40)
    roster = [dqn_spawn(f"DQ{i}", game, seed=i, config=config) for i in range(game.n_agents)]
    before = [{k: v.detach().cpu().numpy().copy()
               for k, v in agent.policy.state_dict().items()} for agent in roster]

    run_match(game, roster, rounds=40, seed=5)

    for agent, original in zip(roster, before):
        for key, tensor in agent.policy.state_dict().items():
            assert np.array_equal(tensor.detach().cpu().numpy(), original[key]), (
                f"{agent.name} changed its weights at {key} during its own fitness match")
        assert not agent.buffer, (
            f"{agent.name} filled its replay buffer under the Darwinian default; nothing should "
            f"have reached it, since update returns first")
    print("OK: a Darwinian network leaves its weights and its buffer alone")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}\n     {exc}")
    print(f"\n{'all evolution-architecture tests passed' if not failures else f'{failures} failed'}")
    sys.exit(1 if failures else 0)
