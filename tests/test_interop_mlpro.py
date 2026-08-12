"""
MLPro consumes GameBrains through its Gymnasium bridge, and cannot through its PettingZoo bridge.

MLPro is an optional dependency, so this module SKIPS when absent. Note that its own declared
requirements are incomplete: the bridge packages do not import until `dill` and `multiprocess` are
installed by hand.

    pip install mlpro mlpro-int-gymnasium mlpro-int-pettingzoo dill multiprocess

Two findings pinned down here, and they differ in kind:

  1. `WrEnvGYM2MLPro` needs a REGISTERED environment, not merely a conformant one. It reads
     `env.env.spec.id` and later calls `gymnasium.make` on that id to rebuild the env, so a bare
     `gymnasium.Env` subclass (what Gymnasium's own `check_env` and SB3 both accept) is refused.
     `interop.wrappers.register_gym_env` covers the gap.

  2. `WrEnvPZOO2MLPro` cannot accept ANY third-party environment. It resolves the environment class
     by name inside five hardcoded `pettingzoo.*` submodules, so anything outside the PettingZoo
     distribution is rejected however well it conforms. This is a limitation of the bridge, not of
     the adapter, and no wrapper on our side can satisfy it.

Both are asserted rather than described, so that a future MLPro release relaxing either one makes a
test fail and prompts an update to the paper's interoperability section.
"""

import numpy as np

from ..agents.qlearning import QLearningAgent
from ..games.public_goods import PublicGoodsGame
from ..interop.gym_adapter import GameBrainsGymEnv
from ..interop.pettingzoo_adapter import GameBrainsParallelEnv
from ..interop.wrappers import register_gym_env

N_AGENTS = 3
ROUNDS = 200
ENV_ID = "GameBrainsPGG-v0"

_LAST_ROSTER: dict = {}


def _make_gym_env() -> GameBrainsGymEnv:
    game = PublicGoodsGame(n_agents=N_AGENTS, rounds=ROUNDS, mpcr=0.5)
    roster = [None] + [
        QLearningAgent(f"Q{i}", n_states=game.n_states, n_actions=game.n_actions, seed=i)
        for i in range(1, N_AGENTS)
    ]
    _LAST_ROSTER["roster"] = roster
    return GameBrainsGymEnv(game, roster, controlled_agent_id=0)


def test_bare_gymnasium_env_is_refused() -> None:
    """Documents why register_gym_env exists. MLPro wants a registry entry, not just conformance."""
    from mlpro.bf.various import Log
    from mlpro_int_gymnasium.wrappers.basics import WrEnvGYM2MLPro

    try:
        WrEnvGYM2MLPro(_make_gym_env(), p_logging=Log.C_LOG_NOTHING)
    except AttributeError:
        return
    raise AssertionError(
        "MLPro accepted a bare gymnasium.Env. register_gym_env may no longer be needed; "
        "re-check interop/wrappers.py and the paper's interoperability section."
    )


def test_mlpro_wraps_the_registered_gym_adapter() -> None:
    import gymnasium
    from mlpro.bf.various import Log
    from mlpro_int_gymnasium.wrappers.basics import WrEnvGYM2MLPro

    register_gym_env(ENV_ID, _make_gym_env, max_episode_steps=ROUNDS)
    env = WrEnvGYM2MLPro(gymnasium.make(ENV_ID), p_logging=Log.C_LOG_NOTHING)

    assert env.get_state_space().get_num_dim() >= 1, "MLPro state space came out empty"
    assert env.get_action_space().get_num_dim() >= 1, "MLPro action space came out empty"


def test_mlpro_drives_the_match_and_our_brains_learn() -> None:
    import gymnasium
    from mlpro.bf.math import Element
    from mlpro.bf.systems import Action
    from mlpro.bf.various import Log
    from mlpro_int_gymnasium.wrappers.basics import WrEnvGYM2MLPro

    register_gym_env(ENV_ID, _make_gym_env, max_episode_steps=ROUNDS)
    env = WrEnvGYM2MLPro(gymnasium.make(ENV_ID), p_logging=Log.C_LOG_NOTHING)
    background = [a for a in _LAST_ROSTER["roster"] if a is not None]
    updates_before = [a.updates for a in background]
    epsilon_before = [a.epsilon for a in background]

    env.reset(p_seed=0)
    action_space = env.get_action_space()
    rewards = []
    for t in range(150):
        element = Element(action_space)
        element.set_values([t % 2])
        env.process_action(Action(p_action_space=action_space, p_values=element.get_values()))
        # No arguments: MLPro's gym wrapper returns the reward the underlying step produced and
        # raises NotImplementedError if handed explicit states.
        rewards.append(float(env.compute_reward().get_overall_reward()))

    assert len(rewards) == 150
    for i, agent in enumerate(background):
        assert agent.updates > updates_before[i], f"{agent.name} never updated"
        assert agent.epsilon < epsilon_before[i], f"{agent.name} epsilon did not decay"
        assert np.abs(agent.Q).sum() > 0, f"{agent.name} Q-table all zeros"


def test_mlpro_pettingzoo_bridge_rejects_third_party_envs() -> None:
    """Asserts the hardcoded allowlist. No adapter change can pass this; that is the finding."""
    from mlpro.bf.various import Log
    from mlpro_int_pettingzoo.wrappers.basics import WrEnvPZOO2MLPro

    allowlist = WrEnvPZOO2MLPro.C_SUPPORTED_MODULES
    assert all(m.startswith("pettingzoo.") for m in allowlist), (
        f"allowlist no longer looks like PettingZoo submodules: {allowlist}")

    game = PublicGoodsGame(n_agents=N_AGENTS, rounds=ROUNDS, mpcr=0.5)
    par = GameBrainsParallelEnv(game, [None] * N_AGENTS,
                                controlled_agent_ids=list(range(N_AGENTS)))
    try:
        WrEnvPZOO2MLPro(par, p_logging=Log.C_LOG_NOTHING)
    except Exception as exc:
        assert "not supported" in str(exc), f"unexpected rejection reason: {exc}"
        return
    raise AssertionError(
        "MLPro accepted a third-party PettingZoo env. The allowlist may be gone; "
        "re-check the paper's interoperability section, which reports this route as closed."
    )


if __name__ == "__main__":
    try:
        import mlpro_int_gymnasium.wrappers.basics  # noqa: F401
        import mlpro_int_pettingzoo.wrappers.basics  # noqa: F401
    except ImportError as exc:
        print(f"SKIP: MLPro bridge packages unavailable ({exc}). Install with: "
              "pip install mlpro mlpro-int-gymnasium mlpro-int-pettingzoo dill multiprocess")
        raise SystemExit(0)

    test_bare_gymnasium_env_is_refused()
    print("OK: a bare gymnasium.Env is refused, as documented (this is why register_gym_env exists)")
    test_mlpro_wraps_the_registered_gym_adapter()
    print("OK: MLPro wraps the registered Gymnasium adapter")
    test_mlpro_drives_the_match_and_our_brains_learn()
    print("OK: MLPro drives 150 steps and GameBrains brains learn against it")
    test_mlpro_pettingzoo_bridge_rejects_third_party_envs()
    print("OK: MLPro's PettingZoo bridge rejects third-party envs, as documented")
