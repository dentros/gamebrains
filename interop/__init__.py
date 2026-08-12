"""
Interoperability adapters: expose a GameBrains `Game` under external standard APIs.

Two adapters here (`pettingzoo_adapter`, `gym_adapter`) let external tooling drive a subset
of "controlled" seats while the remaining seats stay real GameBrains `Agent` objects (classic
strategies, Q-learning, DQN, FEP, ...) acting on their own. This is what makes the adapters
useful rather than a generic pass-through: an outside RL algorithm (via RLlib, Stable-Baselines3,
MLPro) can be trained *against* GameBrains's own cognitive brains.

Because PettingZoo and Gymnasium are the de facto standard APIs, wrapping our `Game` in them also
reaches three further ecosystems. What that actually costs was measured, not assumed
(tests/test_interop_{sb3,rllib,mlpro}.py):

  - Stable-Baselines3  free. Consumes gymnasium.Env directly, nothing to add.
  - RLlib              needs `wrappers.OneHotObs`. Its API stack has no default encoder for a
                       Discrete observation space and refuses to build a model without one.
  - MLPro (Gymnasium)  needs `wrappers.register_gym_env`. WrEnvGYM2MLPro reads env.env.spec.id,
                       so the env must be in Gymnasium's registry, not merely conformant.
  - MLPro (PettingZoo) not possible. WrEnvPZOO2MLPro resolves the env class by name inside five
                       hardcoded pettingzoo.* submodules, so no third-party env can satisfy it.

The lesson worth carrying: interface conformance predicts that a consumer will accept an
environment, not that it will run it.
"""
