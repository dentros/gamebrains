# CLAUDE.md — GameBrains

Guidance for Claude (and humans) working in this repository. Read this first.

---

## 1. What this is

**GameBrains** is a research platform where diverse **cognitive agents** ("brains") play
**repeated game-theoretic scenarios** (starting with the Iterated Prisoner's Dilemma) so we can
compare, visualize, and analyze how different cognitive architectures behave and what social
dynamics emerge.

Two faces, on purpose:

- **Gamified & cute on top** — agents are rendered as little "brain creatures" (Primer /
  lichess-mascot energy), with a **live console / debug log** that streams what is happening every
  round (actions, rewards, internal-state updates). This live log is a hard requirement, not a
  nice-to-have.
- **Scientifically robust underneath** — deterministic, seeded, reproducible simulations; every run
  emits a structured event-log that doubles as the reproducibility record and the data source for
  the paper figures.

Authors / context: Nikolaos Al. Papadopoulos & Konstantinos E. Psannis, Dept. of Applied
Informatics, University of Macedonia. The platform backs a paper (see `../SMC 2025 GAMEBRAINS/`).

## 2. Publication strategy (drives priorities)

1. **Conference (work-in-progress), now** — strengthen the existing SMC-style paper: keep the
   platform review + comparison table + architecture, and add **one catchy experiment/visualization
   from this beta**.
2. **Journal (full paper), target: JAAMAS** — angle **"Theory of Mind in Multi-Agent Systems."**

⇒ **Implication for the code:** the metrics we collect must be **Theory-of-Mind-relevant from day
one** (e.g. transfer entropy / predictive information between agents = a quantitative signature of
"reading the opponent's mind"). Same beta feeds both papers.

## 3. Architecture (decided)

Two decoupled pieces communicating over a structured **event-log**:

```
  Python engine  ──(event-log over WebSocket)──▶  Web frontend (Canvas + TS)
  - game loop                                     - cute brain creatures
  - Agent interface                               - live console / debug log
  - metrics                                       - charts (coop rate, payoff, ...)
  - event-log emitter                             - per-brain inspector panel
```

The event-log is the single contract between engine and viz. It is also the on-disk record used for
reproducibility and for generating paper figures. Design it well and keep it stable.

### The `Agent` interface (build this first, everything hangs off it)

Every brain implements the same contract so new brains plug in without touching the core:

- `act(observation) -> action`
- `update(reward, next_observation)` — learning / adaptation (no-op for fixed strategies)
- `inspect() -> brain_state` — the raw internal state (Q-table, network weights, beliefs, ...)
- `render_brain() -> viz_payload` — a serializable description the frontend knows how to draw

This directly realizes the paper's claims of a *"unified agent interface"* and *"transparency /
inspect internal states."* Building it **is** validating the paper.

## 4. The agent zoo & libraries

| Brain | Internal state to visualize | Library |
|---|---|---|
| Q-learning | **Q-table** (live-updating grid) | custom / numpy (baseline from CS188 labs is fine) |
| Deep Q-Network | **the network** (layers/weights/activations) | **PyTorch** (or stable-baselines3) |
| Evolutionary Markov-brain animat | population **avg / mean / max** brains; **Φ** & **autonomy** | **DEAP** / `pygad` + custom Markov-brain (à la MABE) |
| Free Energy Principle / Bayesian | beliefs / free-energy landscape | **`pymdp`** (active inference) |
| LLM agent | **natural-language rationale** (the XAI panel) | **Ollama** (local: Llama 3.x / Mistral / Qwen) |
| Classic strategies (TFT, AllD, AllC, ...) | fixed rule | **`axelrod`** (Axelrod-Python) |

**LLM notes:** easy to add via the interface; runs *few* rounds (slow vs RL's millions); pin
temperature/seed for reproducibility; the move rationale is the explainable-AI story (ties to ToM).
Deep mechanistic interpretability (activation probing) is **out of scope for the beta**.

## 5. Metrics

- **Social:** cooperation/defection rate, payoff (avg/cumulative), resource-distribution equity,
  stability of emergent norms. — **DONE**, `metrics/social.py`.
- **Integrated information Φ & autonomy (Albantakis / IIT):** via the vendored **`PyPhi`** fork.
  ⚠️ **Φ is exponential in system size** — compute it **only for the small Markov-brains**
  (~8–16 nodes), never for DQN/LLM. — **DONE**, `metrics/phi_autonomy.py`.
- **Graph-theoretic:** metrics on the interaction / who-cooperates-with-whom network via
  **`networkx`**; correlate these with the social metrics. — **TODO, not started.**
- **⭐ TODO — Transfer Entropy / Mutual Information / Predictive Information (the ToM-relevant
  family), per our own decision-framework paper** — Papadopoulos & Psannis, *"Information-Theoretic
  Measures in AI: A Practical Decision Guide"* (arXiv:2604.23716, same authors as GameBrains — a
  self-citation, not an external survey). This paper's own prescriptive framework is what
  GameBrains should implement and follow, not just cite:
  - **Transfer Entropy (TE)** — directed information flow from one agent's action/observation
    history into another's — the concrete, quantifiable **Theory-of-Mind signature** this project
    has wanted since Phase 1 planning ("does agent A's behavior reveal that it is modeling agent
    B?"). Library: **IDTxl** (multivariate; the paper's recommendation) or **JIDT**.
  - **Mutual Information (MI)** — statistical dependence between agents' decisions, or between an
    agent's internal representation (FEP belief vector, Markov-brain hidden state) and the true
    environment state. Library: IDTxl/JIDT, or **`dit`** for simple discrete cases.
  - **Predictive Information** — the "bridge" measure: how much of an agent's own past is
    informative about its own future. Same tooling as above.
  - **Data source for all three:** the existing structured event-log (per-round actions,
    observations, `brain_snapshot`s) — no new instrumentation needed, only new post-hoc metric
    functions, consistent with the "metrics are pure functions of the log" principle already
    established for `social.py`/`equilibrium.py`/`phi_autonomy.py`.
  - **⚠️ Guardrails to follow (from our own paper, non-negotiable when we implement this):**
    (1) Always distinguish an **estimator** (e.g. KSG for continuous MI/TE — a calibrated point
    estimate suitable for reporting) from a **training surrogate** (e.g. MINE/InfoNCE — a lower-bound
    training objective, NOT a calibrated estimate); never report a training-surrogate value as if
    it were a measured MI/TE. (2) Every reported TE/MI value must state its estimator parameters
    (e.g. TE lag length, $k$ for a kNN estimator) and a bootstrap confidence interval — a bare
    point number is not an acceptable report. (3) Keep Φ restricted to small Markov-brains only
    (already our practice, see above) — the same exponential-cost caution the paper raises.
  - **Delivered (2026-07-17): `metrics/information.py`.** `mutual_information_per_agent()` --
    I(observation; action) per agent, discrete plug-in + Miller-Madow correction, no engine changes
    needed (the shared PGG observation is reconstructed from the already-logged `cooperators`
    series). `transfer_entropy_pairwise()` -- T(agent_i -> agent_j) at lag 1 for every ordered
    pair, each validated against 200 time-shift surrogates with a reported p-value (never a bare
    TE point value, per the guardrails above). Wired into `experiments/run_pgg.py` and both
    `webui/app.py` match routes via `information.compute_all()`; the two headline scalars
    (`mutual_information_bits`, `transfer_entropy_bits`, the latter averaged only over
    surrogate-significant pairs) are in `_METRIC_META`; the full per-agent/per-pair breakdown
    (including every p-value) has its own results-page panel. Tests: `tests/test_information.py`.
    Predictive information and graph-theoretic metrics remain not started. Candidate home for the
    TE-as-ToM-signature result once an actual experiment is run: Paper 3 (JAAMAS); Paper 2
    (software journal) should still mention this and cite the framework paper (already added, see
    `IEEE SOFTWARE GAMEBRAINS/root.tex` Future Work).

## 6. Games

**Hard requirement: every game is n-player-native (>= n), standard, generalized.** The platform
must be genuinely multi-agent, not a 2-player tool.

- **Game #1 — N-player Prisoner's Dilemma = Public Goods Game (PGG):** the textbook generalization
  of PD to n players (reduces to classic PD at n=2). Known result (cooperation collapse /
  free-riding) = strong validation. Social metric: cooperation rate.
- **Game #2 — Honey-Jar Game (HJG), formerly "MBoE" (Multi-agent Battle of the Exes):** the
  n-player anti-coordination / congestion game from the ALT project (Hawkins & Goldstone lineage;
  `environment.py`), pairs with the ALT metrics. **Naming:** the user has since renamed this game
  to **Honey-Jar Game (HJG)** across their papers (see `../../RP for Journal/paper_main_teac.tex`,
  §"The Honey-Jar Game as Temporal Fair Division"); "MBoE" is now historical-only. Implement as
  `games/honey_jar.py`, not `mboe.py`. Formal payoff: solo winner → `r_high`; partial tie
  (2≤m<n) → `r_high/n` (ILF) or `r_high/n²` (IQF); all n simultaneously → `0` (full congestion).
  Type-A state = positions only; Type-B = positions + last-winner memory.
- Later: Battle of the Sexes proper, custom game builder.

Games expose: number of rounds, information visibility, payoff parameters.

## 7. Tech stack & conventions

- **Language:** Python (engine) + TypeScript (web viz). No low-level language — PyTorch/numpy/numba
  already exploit local CPU/GPU at this scale; the ML/GT libraries we need only exist in Python.
- **Repro is non-negotiable:** explicit seeds everywhere; log config + seed in every event-log;
  same config ⇒ same results.
- **Structure (current):**
  ```
  gamebrains/
    engine/        # game.py, agent.py, eventlog.py, registry.py, console.py, runner.py
    games/         # public_goods.py (mboe.py planned)
    agents/        # classic.py, qlearning.py, dqn.py, fep.py (markov_brain.py, llm.py planned)
    metrics/       # social.py, equilibrium.py (pygambit) (phi_autonomy.py, graph.py planned)
    interop/       # pettingzoo_adapter.py, gym_adapter.py — see section 9c
    vendor/pyphi/  # patched PyPhi fork — see FORK_NOTES.md, section 9a
    tests/         # test_pyphi_fork.py, test_pettingzoo_adapter.py, test_gym_adapter.py, test_equilibrium.py
    experiments/   # run_pgg.py (paper-figure scripts planned)
    viz/           # TS frontend (canvas creatures + live console + charts) — planned
    .venv/         # dedicated venv (numpy, torch, gymnasium, pettingzoo, pygambit)
  ```
  Current: `agents/{classic,qlearning,dqn,fep,markov_brain}.py`, `metrics/{social,equilibrium,phi_autonomy,information}.py`,
  `engine/evolution.py`, `repository/{normalize,cas,ledger,smart_filter,record}.py`, `webui/` — see 9e.
  `markov_brain.py`/`phi_autonomy.py` planned entries above are now implemented.
- **Style:** match surrounding code; type hints on public functions; keep the engine framework-free
  (agents/metrics/viz depend on the engine, never the reverse).
- **License / hosting:** open source on GitHub. The reused ALT code is **100% the author's own**
  (sole copyright holder, no co-owners) → GameBrains may take any license (e.g. MIT/Apache); the
  ALT repo's GPL does not constrain us.
- **⚠️ WORKFLOW RULE:** **ask the user for confirmation before writing or running ANY code.** Present
  the plan (files, interfaces, logic) and wait for an explicit "go". Docs/markdown and explanations
  do not count as code.

## 8. Phasing — "one basic thing, done right" first

1. **Vertical slice:** N-player PGG + Q-learning brain end-to-end → live Q-table + live console +
   cooperation/payoff + 1 info-theoretic metric. Reproduce the known free-riding/cooperation-collapse
   result as validation.
2. Add brains one at a time onto the same interface: classic strategies → DQN → Markov-brain
   (+Φ/autonomy) → FEP → LLM (+rationale XAI). Add MBoE as game #2 (+ ALT metrics).
3. Repository / decentralized (Matrix) + Smart Filter (energy-saving reuse of prior experiments).

Do **not** build four half-finished brains. A correct interface + one polished brain beats four
stubs.

## 9. Reusable assets from the ALT / MBoE project

Source: `../../2. ALT MEASURES TO MEGALO/src/` — the author's own code (Paper 3 "The Coordination
Gap"; GitHub `dentros/Alternation`). **100% author-owned → free to relicense and reuse.** This is
the "MBoE testbed" with the ALT social metrics the user wants in GameBrains.

| ALT file | Reuse | Goes to | Notes |
|---|:--:|---|---|
| `metrics.py` | 🟢 lift as-is | `metrics/social_alt.py` | ALT family (FALT, EALT, qFALT, qEALT, CALT, AALT) + RP (AWE/WPE) + traditional (Efficiency/Fairness) + PA benchmark + `calculate_coordination_score` (vs random). Pure functions over `top_agents_per_episode` & `terminal_occurrences_per_episode` → engine-agnostic, drop-in. **Highest-value, lowest-effort win.** |
| `environment.py` | 🟢 port | `games/mboe.py` | MBoE dynamics: n agents, positions (default 3), actions stay/move, rewards ILF `r/n` or IQF `r/n²`, Type-A/Type-B state. |
| `dqn_agents.py` | 🟢 lift | `agents/dqn.py` | PyTorch DQN (2×128 hidden, ReplayBuffer, target net, GPU-aware, one-hot state). Nearly wrap-and-go. |
| `qlearning_agents.py` | 🟡 refactor | `agents/qlearning.py` | Tabular Q + Type-A/B state encoding. Reuse encoding; see refactor note below. |
| `config.py` | 🟢 reuse | defaults | `COLOR_PALETTE` per-metric + hyperparams (alpha=0.3, gamma=0.999, epsilon 0.9→, full_reward=100). |
| `visualization.py` | 🟠 reference only | `paper_figures/` | 207 KB matplotlib/seaborn @300 DPI. These are the "classic plots" the user wants to *replace* with the gamified web viz. **Keep only for static paper figures**, not the live platform viz. |

**⚠️ Key refactor (learning location):** In the ALT project the learning rule lives *inside*
`Environment.step()` (the Q-update is at `environment.py:298-303`), and there is a **separate
environment per brain type** (`qlearning_environment.py`, `dqn_environment.py`,
`random_baseline_environment.py`). That works only when all agents are the same type. GameBrains
needs **heterogeneous brains in one match**, so the learning must move *out* of the game and *into*
each agent (`Agent.update()`); the game then does only game logic and is brain-agnostic. Per-agent
Q-tables / independent policies are unaffected — only *where the update code lives* changes.

## 9b. Promised features (paper contract) & delivery status

The paper + notes commit us to the features below. Build toward all of them; the PoC covers a
subset. **Do not silently drop the decentralized repository or the open license — they are headline
promises.**

- Agents ≥4 cognitive architectures (RL, evolutionary/Markov-brain, Bayesian, FEP, LLM, custom) with
  inspectable internal state — **PoC: Q-learning, DQN, FEP, classic, Markov-brain (+Φ/autonomy) done; LLM pending.**
- n-player games: PD, Battle of the Sexes, **Battle of the Exes / MBoE**, Public Goods; repeated /
  stochastic / incomplete-info; static + dynamic; custom builder; **equilibrium detection** — **PoC:
  PGG + equilibrium detection (pygambit, `metrics/equilibrium.py`) done; MBoE pending.**
- Experiment-config GUI (population, agent-mix pie, sliders, rounds, info visibility, payoffs) — **delivered (lite): `webui/` (Flask), a local form-driven demo interface — see 9e. Full "houses" web GUI (TS/Canvas per the architecture diagram) still pending.**
- **Smart Filter** — real-time reuse of prior experiments + energy/computation savings — **delivered (lite): `repository/smart_filter.py` (exact + k-NN similarity lookup over the ledger) — see 9e. Real-time notification (Matrix) layer still pending.**
- Multi-level analysis (agent + system): learning curves, Q-table/brain inspection, social-network
  emergence, heatmaps, Nash convergence, **information-theoretic + graph-theoretic** measures,
  correlations social↔cognitive — **PoC: social + equilibrium + Φ/autonomy done; transfer entropy/MI/graph-theoretic pending (see §5).**
- Visualization: interactive charts, **3D strategy space**, time-series, comparison matrices — **PoC: cooperation-rate chart + brain-state cards in `webui/` (lite) — 3D strategy space and the full "houses" UI still pending.**
- **⭐ Decentralized public Repository & Network Layer** — the paper says **Matrix network
  protocols**; the user also wants **public blockchain** storage with an **OPEN LICENSE**. **Delivered
  (lite, 2026-07-09): `repository/` — content-addressed store + signed (Ed25519) append-only ledger
  + automatic `extends`/`replicates` lineage detection — see 9e for full detail.** Still pending:
  storage nodes / federated access / Matrix notifications / public-chain anchoring / IPFS network
  layer (all require the peer-to-peer layer this "lite" version deliberately does not build yet).
- Cross-cutting: **open license** (code + data), reproducibility, community, energy efficiency.
- Applications framing: blockchain governance, market design, resource allocation.

## 9c. Interoperability layer + equilibrium detection

Two small adapters in `interop/` expose our `engine.Game`/`Agent` under external standard APIs,
**without changing the engine**: one `Agent`-controlled seat (or a subset of seats) is driven by
an outside caller; every other seat is a real GameBrains `Agent` (classic/Q-learning/DQN/FEP)
acting *and learning* on its own, invisibly, each round. This is what lets an external RL
algorithm be trained or evaluated *against GameBrains's own cognitive brains* — the actual
research value, not just "made it look like a generic env."

- **`interop/pettingzoo_adapter.py`** — `GameBrainsParallelEnv(pettingzoo.utils.env.ParallelEnv)`.
  Validated against PettingZoo's own `pettingzoo.test.parallel_api_test` conformance suite,
  plus a test that a background `QLearningAgent` actually learns (epsilon
  decays, Q-table updates) purely from being stepped through the adapter with zero direct calls
  into `engine.runner`. See `tests/test_pettingzoo_adapter.py`.
- **`interop/gym_adapter.py`** — `GameBrainsGymEnv(gymnasium.Env)`, the single-controlled-seat case,
  implemented by *composing* `GameBrainsParallelEnv` (one seat controlled, rest background) rather
  than duplicating the seat-splitting logic. Validated against Gymnasium's own
  `gymnasium.utils.env_checker.check_env` (what Stable-Baselines3 runs on a custom env before
  training) plus the same background-learning test. See `tests/test_gym_adapter.py`.
- **Because PettingZoo/Gymnasium are the de facto standards, these 2 adapters buy compatibility
  with 3 more ecosystems at zero extra code:**
  - **RLlib** via `ray.rllib.env.wrappers.pettingzoo_env.PettingZooEnv` (consumes any PettingZoo env).
  - **MLPro** (`fhswf/MLPro`, continues at `blueAIC/MLPro`, Apache 2.0) via
    `mlpro_int_pettingzoo.wrappers.basics.WrEnvPZOO2MLPro` /
    `mlpro_int_gymnasium.wrappers.basics.WrEnvGYM2MLPro`.
  - **Stable-Baselines3**, which consumes `gymnasium.Env` directly.
- **`metrics/equilibrium.py`** — Nash equilibrium detection for `PublicGoodsGame` via `pygambit`
  (Gambit's Python bindings): builds the 2ⁱ normal-form payoff table from the closed-form payoff
  function and calls `pygambit.nash.enumpure_solve` (any n) / `enummixed_solve` (n=2 exact only —
  Gambit's vertex-enumeration mixed solver is two-player-only; general n-player mixed equilibria
  would need the heavier `enumpoly_solve`/`gnm_solve`/`logit_solve`, out of scope for now).
  Validated against theory: for any config satisfying our social-dilemma condition
  (`1/n < mpcr < 1`), Defect strictly dominates, so universal defection is *always* the unique pure
  Nash equilibrium — confirmed for n=2..5 in `tests/test_equilibrium.py`, and consistent with the
  empirical free-riding collapse already observed from RL training (two independent methods,
  analytical and empirical, agreeing — worth stating explicitly in the paper).
- **Not adopted:** PettingZoo's dict-by-agent-id convention as our own *internal* `Game`/`Agent`
  contract. PettingZoo only models the environment (obs/reward per agent id, agents as black
  boxes); our `Agent` carries strictly more (`inspect`/`render_brain` transparency, `training_mode`,
  event-log emission) that has no PettingZoo equivalent, so adopting their convention natively would
  not remove that layer — it would just cost a rewrite of 3 already-validated phases for a stylistic
  convention match. The adapters are a one-line-per-call list↔dict translation instead.
- **Reused directly from PettingZoo's own code** (not just "compatible with"):
  `pettingzoo.test.parallel_api_test` (conformance testing our adapter), and — noted for later —
  `pettingzoo.utils.conversions.{aec_to_parallel_wrapper,parallel_to_aec_wrapper}` (free AEC/turn-based
  view of a Parallel env) and `pettingzoo.utils.agent_selector.AgentSelector` (turn-order management),
  relevant if/when a sequential-move game is added.

**Other platforms from the review, integration verdict:**
- 🟢 Real code integration: **Gambit/pygambit** (done, above). **Axelrod-Python** strategies target
  pairwise 2-player IPD history and don't map onto our n-player aggregate-signal PGG observation —
  would plug in directly only if/when a genuine 2-player IPD game mode is added.
- 🟡 Conceptual inspiration only, no code borrowed: MABE → `markov_brain.py` design; ABED → evolutionary
  revision protocols; Dynamo → phase-diagram visualization ideas for the future evolutionary "house".
- ⚪ Citation-only, no integration path (proprietary/GUI/non-Python/legacy): MobLab, z-Tree, GamePlan,
  TUGlab, PDToolbox (MATLAB), Didactic Web-Based Experiments, Finite Mathematics Utility (web/JS),
  GameSolver.NET (.NET), Evoplex (C++/Qt), EvoDyn-3s, ALYMPICS (research code, not a maintained
  package — possibly worth mining for LLM-agent prompting ideas for `agents/llm.py`).

**For the paper (root.tex) — APPLIED (2026-07-06), compile-verified (2 pdflatex passes, 0 undefined
refs):** (1) new comparison-table row for **PettingZoo**; (2) updated MLPro row/paragraph (continues
at `blueAIC/MLPro`, Apache 2.0, itself depends on external Gymnasium/PettingZoo interop packages —
reinforces our own point); (3) new `\subsection{Interoperability with the Broader MARL Ecosystem}`
describing the "2 adapters → 5 ecosystems" strategy; (4) 4 new bibitems (terry2021pettingzoo,
towers2024gymnasium, liang2018rllib, raffin2021stable).

## 9d. Markov-brain animats, evolution, and Φ/autonomy (Phase 4)

- **`agents/markov_brain.py`** — `MarkovBrainAgent`: a small stochastic recurrent binary network
  (sensor/hidden/motor nodes, `training_mode="evolutionary"`). Node dynamics
  `P(node=1) = sigmoid(W·state + bias)` for hidden/motor; sensors are clamped from the game
  observation each step. `W`/`bias` are the evolvable genome; `clone`/`mutate`/`crossover` are GA
  hooks; `full_tpm()` builds the state-by-node TPM (little-endian row order, matching
  `pyphi.convert.le_index2state`) that `metrics/phi_autonomy.py` consumes.
- **`metrics/phi_autonomy.py`** — Φ via `pyphi.compute.sia` on the fixed hidden+motor subsystem
  (see performance note below), and **causal autonomy**: a faithful reimplementation of Albantakis
  et al.'s `A_m_sensors_causal` (github.com/Albantakis/autonomy, `causal_agent_analysis.py`) —
  not the actual package (officially Windows-unsupported, needs the same unpatched PyPhi our fork
  fixes), but the *same formula*, rebuilt from the same `pyphi.tpm`/`pyphi.convert` utilities it
  uses internally, verified empirically shape-by-shape against those utilities before shipping.
  **Honesty carried from the source:** the full causal-autonomy measure is a difference of two
  conditional entropies; the second term is exactly 0 only for *deterministic* dynamics, and the
  reference implementation (their own comment) only computes the first term, with a `TODO: add
  the part for non-deterministic agents` left in their code. Our animats are stochastic, so — like
  the published implementation — we compute that first term only, not the complete measure.
  Validated on two analytically-known degenerate genomes: all-zero (every hidden/motor node an
  independent fair coin) → causal autonomy = exactly n_hidden+n_motor bits; strongly-negative-bias
  (near-deterministic) → causal autonomy ≈ 0. Both match exactly.
- **⚠️ Performance finding (2026-07-07/08), important for any future Φ/autonomy work:** naively
  using `pyphi.compute.major_complex` (search over *every* candidate subsystem, matching the
  reference toolbox's `average_IIT_3_0`) took **1109.86s (~18.5 min) for a single 5-node animat**.
  Switching to `pyphi.compute.sia` on the *fixed* hidden+motor subsystem — justified because the
  main complex empirically always turned out to be exactly hidden+motor anyway (sensors are
  clamped, not autonomous, so they contribute little intrinsic cause-effect power) — plus letting
  PyPhi's own default `PARALLEL_CUT_EVALUATION=True` run (we had it force-disabled repo-wide to
  dodge a Windows multiprocessing hang that only happens with `python -c` one-liners, not real
  script/module invocations) brought the *identical* computation down to **12.97s** (~85×). Do not
  reintroduce `major_complex` or blanket-disable parallel evaluation without re-checking this.
- **`engine/evolution.py`** — generational GA: fitness = average payoff from self-play matches
  (via the existing brain-agnostic `runner.run_match`, reused as-is), tournament selection, uniform
  crossover, Gaussian mutation, elitism. Deliberately never computes Φ/autonomy inside the loop
  (only on the final best genome, given the cost above).
- **Validated finding, cross-checked three independent ways:** evolving a population of
  `MarkovBrainAgent`s on the Public Goods Game converges fitness toward **0** — exactly the payoff
  of universal defection, i.e. the same Nash equilibrium `metrics/equilibrium.py` proves
  analytically (§9c) *and* the same free-riding collapse already observed from Q-learning/DQN
  training (Phase 1/2). RL convergence, game-theoretic analysis, and evolutionary dynamics all
  agree — worth stating explicitly in the paper as cross-paradigm validation.
- **`experiments/run_evolution.py`** — demo: runs the GA with live per-generation console output,
  then computes Φ/autonomy for the fittest evolved genome (evaluated at a concrete state reached
  by stepping once from a fresh reset — Φ is always a property of a system *at a state*, not of
  the transition rules alone).

## 9e. Repository-lite, Smart Filter, and the demo web UI (Tier 2, 2026-07-09)

- **`repository/normalize.py`** — canonical config normalization + `hash_config()`. `config_hash`
  is computed over `game` + `roster` + `code_version` only — **deliberately excludes `horizon.rounds`
  and `seeds.master`** so that same-design runs of different length/seed share one `config_hash`,
  which is what makes `extends`/`replicates` detection possible (see §4 of
  `docs/repository-schema.md`).
- **`repository/cas.py`** — `ContentStore`: local content-addressed store (sha256 CID over
  canonical JSON), whole-package addressing (not per-chunk Merkle-DAG — a documented
  simplification), dedup on `put()`.
- **`repository/ledger.py`** — `Ledger`: append-only JSONL, each record **really Ed25519-signed**
  (`cryptography` package, key auto-generated/persisted unencrypted as `identity_private.pem` —
  explicitly a "lite"/local scheme, not production key management), hash-chained via `parents`
  (`verify_chain()` catches tampering with an *older* record even if its signature were
  recomputed), automatic `extends`/`replicates` lineage detection (`_detect_lineage`; the two are
  checked independently and **can coexist** on one record — tested explicitly), `find_exact()` for
  exact-match dedup.
- **`repository/smart_filter.py`** — the two lookups from `docs/repository-schema.md` §6:
  `nearest()` (k-NN Euclidean over the record's `feature_vector`) and `lookup()` (exact + similar
  together). **Honesty note:** distance is unweighted/unnormalized on the raw feature vector, so
  `rounds` dominates it in practice — documented limitation, not hidden.
- **`repository/record.py`** — `record_experiment()`: glue from a completed match to a ledger
  record. **Bug found & fixed via real `run_pgg.py` testing, not just unit tests:**
  `game.describe()` embeds `rounds` alongside design fields (mpcr, cost...); passing it straight
  through into `config_hash` silently defeated `extends` detection on every real run (the unit
  tests didn't catch this because they call `ledger.append()` directly with hand-built
  `game_desc` dicts that never included `rounds`). Fixed by `_config_game_desc()`, which strips
  `rounds` before hashing. Regression test: `tests/test_record.py`.
- **`experiments/run_pgg.py`** — now calls `record_experiment()` at the end of every run (new
  `--repo` flag, default `gamebrains/repo_store`, `--repo ''` to skip), prints
  `config_hash`/`content_cid`/lineage.
- **`webui/`** (Flask, new dependency — pure-Python wheel, no C-extensions) — a **local, form-driven
  demo interface**, explicitly documented as a first, functional step toward the full gamified
  "houses" UI (§8), not that UI itself. Lets a user pick every parameter (agents, rounds, mpcr,
  seed, roster mix per kind) through a browser form, runs the match through the same
  engine/agents/metrics/repository code the CLI uses, and renders: per-agent brain-creature cards
  (Q-table/DQN heatmap, FEP belief bars, Markov-brain bit-grid, classic rule text), a server-side
  SVG cooperation-rate chart (no JS/CDN dependency), the full narrated console log (captured via
  `contextlib.redirect_stdout`), the resulting ledger record + lineage, Smart Filter hits
  (exact + similar), and optional Nash-equilibrium / Φ-autonomy panels. Run via
  `"$GB/gamebrains/.venv/Scripts/python.exe" -m gamebrains.webui.app` → `http://127.0.0.1:5000/`.
  **⚠️ Windows gotcha found & fixed:** `app.run(debug=True)` (Flask's reloader) spawned a second
  process — on Windows this resolved to a *different* Python interpreter than the one hosting the
  platform's dependencies — and PyPhi's spawn-based parallel-cut-evaluation workers (§9d) got
  confused across that process boundary, leaking orphaned `--multiprocessing-fork` processes and
  raising `MemoryError`/`std::bad_alloc` from the LP solver. Fixed by
  `app.run(debug=False, use_reloader=False)` — a single process. A compounding instance of the
  same Windows spawn-based-multiprocessing caution already noted in §9d's performance finding, not
  a new failure mode.
- Redesign in progress (2026-07-09/10, per user feedback): dynamic per-kind roster builder (no
  manual `n_agents` field — derived from the counts you add), configurable epsilon schedule
  per kind with a live "rounds until ε reaches ε_min" indicator, a real DQN architecture/weight
  visualization (not just the derived Q-table), inline theory+practice explanations for the
  FEP-AI belief panel and the Nash panel, metrics on/off checkboxes + disabled roadmap
  placeholders (Honey-Jar game, transfer entropy/MI/graph-theoretic metrics), and a separate
  `/evolve` tab for the Markov-brain GA with a **fair bake-off comparison mode**: evolve/train each
  architecture to convergence *separately*, freeze all of them, then run one evaluation match
  together — the clean way to compare a population-evolved architecture against online learners
  without the confound of simultaneous co-adaptation. Two further comparison modes (evolved genome
  dropped into the normal mixed-learning match; one kind learns while the rest are held fixed) are
  planned as thinner layers on top of the same primitives. A fourth mode (alternating
  learning-vs-frozen phases across paradigms) is deliberately deferred — noted as a candidate
  Theory-of-Mind research question for Paper 3/JAAMAS, not a demo feature.
- **New `/analytics` tab (2026-07-17), delivered.** A third webui tab (alongside Match and
  Evolutionary) that queries every run ever recorded to the repository-lite ledger, not just the
  one just executed: a "smart filter" over game fields, every scalar metric, and every per-kind
  hyperparameter actually stored in a run's `roster[i]["params"]` (reusing
  `repository/record.py`'s `_PARAM_ATTRS` as the single source of truth for what's filterable, so
  it can't drift out of sync with what a record stores), combined with AND; a per-kind-parameter
  filter matches a run if *any* agent of that kind in its roster satisfies the condition (e.g.
  "runs with a Q-learning agent whose epsilon_decay >= 0.8"). Results render as a table plus a
  user-chosen X/Y scatter chart (server-side SVG, no JS charting library) over the filtered set.
  This is the "general page with smart filters + correlations + charts across all games" the user
  asked for; the full parameter-space-map (know every possible config per game and auto-run
  what's missing) remains separate, not-yet-built future work.

## 10. References

- Albantakis et al. — Integrated Information Theory, Φ, autonomy in animats; `PyPhi` (see
  `vendor/pyphi/FORK_NOTES.md` for the compatibility fork we maintain).
- Axelrod, *The Evolution of Cooperation*; Axelrod-Python library.
- Active inference / Free Energy Principle — `pymdp` (not used directly; `agents/fep.py` implements
  the math in numpy — see that file's docstring for why).
- MABE (Modular Agent Based Evolver) — Markov-brain inspiration.
- Mayner et al. (2018), *PyPhi: A toolbox for integrated information theory*, PLOS Comp. Biol.
- Terry et al., *PettingZoo: Gym for Multi-Agent Reinforcement Learning*; Towers et al., *Gymnasium*.
- Moritz et al., *Ray/RLlib*; Raffin et al., *Stable-Baselines3*; Arend et al. (2022), *MLPro*.
- McKelvey, McLennan & Turocy, *Gambit: Software Tools for Game Theory* (`pygambit`).
- Existing paper sources: `../SMC 2025 GAMEBRAINS/death/root.tex` (final), `gamebrains_architecture.png` (Fig. 1).
