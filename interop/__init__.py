"""
Interoperability adapters: expose a GameBrains `Game` under external standard APIs.

Two adapters here (`pettingzoo_adapter`, `gym_adapter`) let external tooling drive a subset
of "controlled" seats while the remaining seats stay real GameBrains `Agent` objects (classic
strategies, Q-learning, DQN, FEP, ...) acting on their own. This is what makes the adapters
useful rather than a generic pass-through: an outside RL algorithm (via RLlib, Stable-Baselines3,
MLPro) can be trained *against* GameBrains's own cognitive brains.

Because PettingZoo and Gymnasium are the de facto standard APIs, wrapping our `Game` in them also
yields, at no extra code:
  - RLlib      via ray.rllib.env.wrappers.pettingzoo_env.PettingZooEnv
  - MLPro      via mlpro_int_pettingzoo.wrappers.basics.WrEnvPZOO2MLPro /
                   mlpro_int_gymnasium.wrappers.basics.WrEnvGYM2MLPro
  - Stable-Baselines3, which consumes gymnasium.Env directly.
"""
