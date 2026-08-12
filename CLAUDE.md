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
Informatics, University of Macedonia. The platform backs a paper (see
`../CONFERENCE PAPER GAMEBRAINS/`, not tied to any specific venue -- see that folder's `death/`
and `arxiv/` subfolders).

## 2. Publication strategy (drives priorities)

1. **Conference (work-in-progress), now** — strengthen the existing conference paper: keep the
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

### Games declare what their numbers mean (2026-07-28)

An action index means nothing on its own, and this bit us for real. `agents/classic.py` imported
`COOPERATE` from the Public Goods Game, pinning "cooperate" to action 1. Action 1 in
`games/congestion.py` is MOVE, grabbing the contested resource, so **`AllC` played the most
aggressive strategy available there while still being labelled cooperative.** Nothing failed; the
game ran fine, only the meaning was wrong. Found only because a second, structurally different
game finally existed to expose it.

The fix: a game declares its own semantics, agents ask for meaning rather than for numbers.

- `Game.action_roles` maps a semantic role to *that game's* action index; `Game.action_for(role)`
  resolves it or refuses with a message naming what the game does declare.
- `Game.observation_kind` says what the integer means (`opaque` default, `concede_count`,
  `board_index`); `Game.require_observation_kind(...)` lets an agent refuse early. Majority-TFT
  needs this second half: counting what others did requires a count, not a board index.
- **Two vocabulary levels, deliberately.** Universal (`concede`/`claim`) covers any game with an
  individual-vs-collective tension, so an agent written today works with a game written later.
  Family aliases (`cooperate`/`defect`, `yield`/`contest`) let a game speak its own literature's
  language. A game declares only what it truly has, so a coordination game like Battle of the
  Sexes (no concede axis at all, the question is *which* option) correctly refuses an
  always-concede agent instead of silently accepting it.
- New `Agent.on_match_start(game)` hook, symmetric with the existing `on_match_end`. The runner
  calls it; `interop/pettingzoo_adapter.py` and the webui's `_Frozen` wrapper also must, since
  they drive agents directly, and both now do.
- The declarations ride in `describe()`, hence in `config_hash`: a game assigning its roles to
  different actions makes every role-aware agent behave differently, so it is a different design.
- **`docs/adding-a-game.md`** is the third-party authoring contract, written because the user
  wants outside contributors adding games. It uses this bug as the worked example of why the
  declaration matters. Tests: `tests/test_action_roles.py` (7 cases), including the headline one:
  the same strategy resolving to *opposite* indices in the two games with one consistent meaning.
- **Caveat to carry into any anti-coordination game:** no constant strategy is collectively good
  there. Everyone conceding pays zero exactly as everyone claiming does; the collectively best
  behaviour is *taking turns*, which no fixed strategy can express. AllC/AllD stay useful
  baselines in HJG but neither is "the cooperative agent" there, which is precisely the point the
  ALT metrics exist to make.

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
  **`networkx`**; correlate these with the social metrics. — **Delivered (2026-07-19):
  `metrics/graph.py`.** Two graphs per match, both pure functions of already-recorded data:
  a weighted co-cooperation graph (edge weight = fraction of rounds both agents cooperated;
  headline scalars `coop_graph_weight`, `coop_graph_clustering`) and a directed influence graph
  whose edges are exactly the surrogate-significant TE pairs from `metrics/information.py`
  (headline `influence_graph_density`) — reusing the validated TE results, never recomputing them.
  Validated end-to-end on a real match with a MajorityTFT in the roster: the influence graph
  recovered exactly the theoretically-required direction (both Q-learners → TFT significant, since
  TFT literally reads their previous actions; reverse directions correctly rejected). Tests:
  `tests/test_graph.py`.
- **Temporal fairness: ALT family + RP + the traditional measures — DELIVERED 2026-08-02,
  `metrics/social_alt.py`.** The author's own measures, ported so the platform can reproduce their
  central finding rather than only cite it. Depends on the episodic runner and Game #2; the ALT
  family is defined per *episode* and there was nothing to compute before both existed.
  - **Ported from two sources, deliberately, and two functions must NOT be used.** ALT ×6 +
    Efficiency + the 3 fairness measures + coordination score come from
    `2. ALT MEASURES TO MEGALO/src/metrics.py`. RS/WPE/RP come from
    `RP for Journal/synthetic_experiments/common.py` (extracted from
    `compute_rp_full_corrected.py`). **Do not** use `src/metrics.py`'s `compute_rp_metrics`: it
    averages WPE with AWE, RS's deprecated predecessor, which returns 0 for any agent whose
    average wait reaches twice the ideal gap `n-1`. Every collapsed run in the source data does
    that, so RP becomes WPE/2 with no rhythm term (corrected values run 1.5-2x higher). Verified
    empirically both ways while adding the warnings: healthy alternation at n=3 gives AWE 0.594
    and RP != WPE/2, a run with long idle tails gives AWE 0.0000 and RP == WPE/2 exactly. **The
    bug is therefore invisible on good data and only bites on the runs whose failure you are
    measuring** (an earlier version of this note wrongly called it "exactly 0 for n>=3", which is
    a property of that dataset, not an algebraic identity). **Do not** use
    `RP for Journal/compute_rs_all_modes.py`'s `compute_rs` either: it predates the boundary fix
    and roughly doubles RS on real data. A grep for "RS" finds the wrong file first, so this trap
    is easy to walk into; the user flagged both before the port, which is how they were avoided.
    **A third stale function found while adding those warnings: `compute_wpe` in the same file.**
    Its `t_i` is the count of waiting *periods*, where canonical WPE uses the raw win-event count
    `k_i`. The two differ by exactly 1 whenever an agent's first and last win bookend the run, and
    that **breaks the "WPE = 1 at an exactly fair share" property**: on n=2, nu=4 with A winning
    at [0,3] and B at [1,2], each agent holds exactly its fair share of 2 wins, so a
    frequency-only measure must read 1.0000; canonical does, this gives 0.5000. (That sequence is
    A,B,B,A, the papers' *clumped* example rather than perfect alternation A,B,A,B. Noted because
    an earlier version of this entry mislabelled it, and the RP session caught it.) Comparing the
    two names the defect exactly: stale gives 0.5000 on A,B,B,A but 1.0000 on A,B,A,B, though both
    hand every agent the same fair share. **The error is not a constant offset, it is order
    sensitivity leaking into a frequency-only measure**, so that WPE quietly duplicates RS instead
    of complementing it as RP's independent second dimension. This is the RP project's own "Second
    subtlety", recorded in its CLAUDE.md but never marked at the function that still has it. (`compute_awe` also uses `next - prev` where canonical uses
    `next - prev - 1`, noted for completeness since AWE is superseded outright.)
    **All three stale functions now carry an in-place `STALE / SUPERSEDED` warning block at their
    own definition and at the offending line**, in the authors' own project folders, so a future
    reader is warned at the point of use rather than only here. Comments only, no behaviour
    change: those files produced the published numbers and the checkpoint pkls, and every one was
    re-run afterwards to confirm identical output. That re-run also independently confirmed the
    ALT family is intact, all six variants scoring exactly 1.000000 on perfect alternation.
  - **Reach vs exclusive is computed for everything, never configured.** Following the papers' own
    "report the whole submetric family, don't pick a winner" principle, so it never becomes a
    convention two users could disagree on (see the protocol-tier entry in §9e for why that
    matters). For ALT the split *is* the variant axis (FALT/qFALT count reaches, EALT/qEALT/AALT
    count solo wins, CALT combines); for RP it is a caller-side choice, so every RP measure ships
    as both `_reach` and `_excl`.
  - **Validated against four published oracles, not against itself:** the Gap paper's worked
    example (CALT = 5 x 0.5625 / 6 = 0.469), ABABABAB at n=2 scoring exactly 1.00, ABBAABBA
    scoring RS 0.775 / WPE 1.00 / RP 0.8875, and ABCABC at n=3 scoring RS 0.889 rather than 1.00.
    That last one is the important one: it is the *disclosed finite-boundary edge effect*, and an
    implementation that dropped boundary waiting periods would score it a cleaner-looking 1.00
    while hiding the failure mode that matters most (an agent that stops winning partway through
    and never recovers has a long idle tail that only boundary counting sees).
  - **Reproduced the headline finding live**, 40k rounds, 3 Q-learners on the `hjg` preset against
    a matched random-policy baseline: Reward Fairness 0.92, TT-Fairness 0.98, Fairness 0.97 (all
    look healthy) while CALT 0.040 and AALT 0.025 (collapsed), 95% of 19,844 episodes ending in
    collision, and coordination scores negative on *every* measure (CALT -50.6%, inside the
    papers' own 34-74% band). Efficiency 0.22 for Q-learning against 0.87 for random: the random
    agents are four times more efficient precisely because they do not all learn to rush at once.
  - Tests: `tests/test_social_alt.py` (11 cases), including the traditional-measures-look-fine
    scenario as an explicit assertion.
- **Transfer Entropy / Mutual Information / Predictive Information (the ToM-relevant
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
    Predictive information delivered 2026-07-19 (`predictive_information_per_agent` in the same
    module: one-step PI = I(action_{t-1}; action_t) per agent, same lag-1 convention as TE so the
    two are directly comparable; validated on alternator ~1 bit / iid ~0 / constant 0); graph
    metrics also delivered 2026-07-19 (see the Graph-theoretic entry above). The webui
    `_METRIC_ROADMAP` is now empty — the whole section-5 metric family is implemented. Candidate home for the
    TE-as-ToM-signature result once an actual experiment is run: Paper 3 (JAAMAS); Paper 2
    (software journal) should still mention this and cite the framework paper (already added, see
    `SPE GAMEBRAINS/root.tex` Future Work).

## 6. Games

**Hard requirement: every game is n-player-native (>= n), standard, generalized.** The platform
must be genuinely multi-agent, not a 2-player tool.

- **Game #1 — N-player Prisoner's Dilemma = Public Goods Game (PGG):** the textbook generalization
  of PD to n players (reduces to classic PD at n=2). Known result (cooperation collapse /
  free-riding) = strong validation. Social metric: cooperation rate.
- **Game #2 — DELIVERED 2026-07-26 as `games/congestion.py`.** Built as a *parametrized family*
  rather than one game, because the Gap paper itself frames it that way ("a minimally dynamic,
  repeated threshold-congestion game" that "stands on its own footing within the congestion and
  market-entry family"), and the conference paper already promised "an anti-coordination
  congestion game in the Hawkins-Goldstone tradition". HJG is the headline preset, not the whole
  module. Depends on the episodic-runner commit above; it could not have worked before it.
  - **Axes, each one an axis the source papers actually vary:** `num_positions` (2 = one-shot,
    the papers' "ballistic"; 3+ = dynamic, with an intermediate cell where approach is observable
    before commitment, which is the whole "minimally dynamic" claim), `reward_rule`,
    `collapse_at_full`, `memory_episodes`, `episode_max_rounds`, `full_reward`.
  - **Reward rules are stored as a resolved `(base, exponent)` pair**, not just a name: ILF=(n,1),
    IQF=(n,2), KLF=(k,1), KQF=(k,2) are the four the papers ran, ICF/KCF=(·,3) are exposed but
    flagged `reward_published: False` so an exploratory run can never be mistaken for reproducing
    a published one. `reward_rule="custom"` takes an arbitrary base/exponent; because `describe()`
    reports the resolved pair, a custom rule spelling out r/n^2 correctly hashes as the same
    design as IQF rather than as something new.
  - **The zero-floor is its own parameter, deliberately.** In the original code the denominator
    choice and the "everyone arrives pays exactly zero" rule were coupled: the main version had
    the floor, the k-variant did not. Those are independent decisions, so they are separate
    parameters here, with the `hjg` / `hjg_k` presets encoding the two published combinations.
    Tested explicitly in both directions, since the asymmetry looks like an oversight and is easy
    to "tidy up" by mistake.
  - `memory_episodes` generalizes Type-A (0) / Type-B (1) to any depth; `state_type` still reports
    the papers' label. Guarded at 5M states, since the count is
    `num_positions^n * 2^(n*memory)` and n=10 with one episode of memory is already ~60 million.
  - **Validated against the papers' arithmetic, not just "it runs":** every reward case asserts
    the published formula. And a first real 20k-round Q-learning match reproduced the qualitative
    headline finding immediately: ~9.9k episodes, only ~6% solo wins against ~94% collisions, with
    those few solo wins spread evenly across agents (175/230/194) so the failure is collision, not
    monopolization, which is exactly the case where outcome-based fairness looks fine while
    coordination has collapsed. Tests: `tests/test_congestion.py` (10 cases).
  - **⚠️ Semantic trap found while testing, not yet resolved:** `agents/classic.py` imports
    `COOPERATE`/`DEFECT` straight from `games/public_goods.py`, where `COOPERATE == 1`, and `1` is
    `MOVE` here. So `AllC` *rushes the jar* and `AllD` *hangs back* in a congestion game, the
    opposite of what the names suggest, since restraint is the cooperative act when a resource
    jams. The game's own per-round `cooperators` count correctly treats STAY as cooperation, so
    the metric and the agent labels currently disagree. Fine for the fixed strategies (they are
    just constant actions) but it will read as a bug to anyone building a roster, and the classic
    agents need a game-agnostic notion of their action before more games land.
- **Game #2 background (kept for the design rationale) — Honey-Jar Game (HJG), formerly "MBoE" (Multi-agent Battle of the Exes):** the
  n-player anti-coordination / congestion game from the ALT project (Hawkins & Goldstone lineage;
  `environment.py`), pairs with the ALT metrics. **Naming:** the user has since renamed this game
  to **Honey-Jar Game (HJG)** across their papers (see `../../RP for Journal/paper_main_teac.tex`,
  §"The Honey-Jar Game as Temporal Fair Division"); "MBoE" is now historical-only. Implement as
  `games/honey_jar.py`, not `mboe.py`.

  **⚠️ Corrected 2026-07-19 (was wrong before — verified via web search + actual source code, do
  not re-assert the old claim that HJG "is just a rename of" Battle of the Exes):**
  - **The real "Battle of the Exes" (BoE)** is a published game (Hawkins & Goldstone, 2016),
    introduced by them, not by the user. It is explicitly a variant of the classical Battle of the
    Sexes with an inverted goal (avoid, not match) but the SAME asymmetric-payoff structure: 2
    actions, matching → (0,0), differing → **asymmetric** (3,2)/(2,3) (e.g. one coffee shop is
    "great," the other "average," so whoever avoids-and-lands-on-the-good-one gets more). BoE is
    close kin to Battle of the Sexes, not its opposite family, contra an earlier note here.
  - **The ALT project's own `environment.py` is a symmetric simplification, not identical to BoE.**
    Confirmed by reading the actual code: whichever single agent is a solo winner at ANY position
    gets the same `full_reward` — no per-position asymmetric payoff like BoE's 3-vs-2. There are
    two reward-formula variants, verified by reading both source files directly (do not trust this
    note alone without re-checking if it matters again):
    - **Main version** (`environment.py`, ILF/IQF): partial-tie reward divides by the fixed total
      `self.num_agents` (n), not by the actual number of agents in the tie -- so a 3-way tie out of
      5 gets the same reward as a 4-way tie out of 5. Full congestion (everyone ties) is an
      **explicit special-cased 0**.
    - **k-variant / KLF** ("per-claimant reward split," lives in the *Gap project* /
      `complexity_journal`, re-run at `../../RP for Journal/synthetic_experiments/run_kvariant_rp.py`
      via a `KEnvironment(Environment)` subclass): divides by `count_of_top_agents`, the *actual*
      tie size -- this is what makes it a genuine congestion game (payoff degrades with how crowded
      *your* resource is, not with total population size). **Edge case: no explicit zero at full
      congestion here** -- `full_reward / count_of_top_agents` when everyone ties just evaluates to
      `full_reward/n` (small, not zero), unlike the main version's explicit floor.
  - **Does HJG "become" BoE at n=2? No**, for two independent reasons, not one: (1) reward
    asymmetry -- BoE gives both parties positive-but-different reward when they differ; HJG (either
    variant) gives the loser exactly 0 when they differ (a race with one winner, not a mutual-avoidance
    payoff for both); (2) game shape -- BoE is one simultaneous 2-action choice; HJG is a multi-round
    race across `num_positions` positions converging on one shared terminal, a different game tree
    entirely, independent of n. Which variant (main ILF/IQF vs. true k-variant/KLF) GameBrains should
    actually implement for its own Game #2 is still an open decision, not yet made.
  Type-A state = positions only; Type-B = positions + last-winner memory.
  - **Why the Gap paper uses ILF/IQF as the main version, not literal Rosenthal congestion
    (k-variant) — relevant to that open decision above** (source: the Gap paper itself,
    `2. ALT MEASURES TO MEGALO/complexity_journal/main/paper_main.tex`,
    §"Differentiating HJG from Congestion, Market-Entry, and Anti-Coordination Games", submitted
    2026-07-19 alongside the RP paper — both now in their final submission phase): the paper gives
    5 distinguishing arguments, verified against the primary source directly (not a secondhand
    paraphrase). The two the paper's own text calls "more fundamental" (used that exact phrase
    twice) are: (1) the ALT/PA evaluation asks *which specific agent* gets access *and when* across
    a repeated sequence, not an aggregate statistic like El Farol/Minority-Game literature does —
    true regardless of reward rule, game dynamics, or even which specific game this is; (2) a
    robustness check using the literal per-claimant/Rosenthal k-variant shows the coordination gap
    still persists, and — the paper's stronger point here — that detecting it *still requires the
    ALT/PA framework rather than congestion-equilibrium analysis even under the most canonical
    possible congestion payoff*, reinforcing rather than undercutting the framework's necessity.
    The other 3 arguments (payoff-structure threshold-step vs. declining-per-entrant; dynamic
    observable approach vs. stateless simultaneous-move; the ILF/IQF choice being deterministic and
    keeping universal collision at exactly zero) are real but more contextual/methodological —
    defenses of *this specific implementation choice*, not of why the framework/question exists.
    **Implication for GameBrains's own Game #2, when built:** the main ILF/IQF version is the
    scientifically-motivated default (deterministic tie penalty, clean zero-floor at full
    congestion, avoids confounding the alternation signal), with the k-variant/KLF worth keeping
    available as an explicit alternative/robustness toggle rather than silently picking one -- that
    mirrors exactly how the Gap paper itself treats the two (main experiments + one robustness
    check), and lets GameBrains demonstrate the same "the finding survives the reward-rule choice"
    argument live in the tool if wanted later.
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
- **The 3 downstream ecosystems, TESTED 2026-08-12 (was asserted before, and the assertion was
  wrong).** The old claim here and in the paper was "2 adapters buy 3 more ecosystems at zero extra
  code". One of the three is free, two need a shim, and one advertised route is impossible. Tests:
  `tests/test_interop_{sb3,rllib,mlpro}.py`, each skipping cleanly if its optional library is absent.
  - **Stable-Baselines3** — genuinely free, via `gym_adapter`. Its own `check_env` passes, DQN and
    PPO train, `DummyVecEnv` wraps parallel instances, and background Q-learners really learn inside
    its training loop (600 updates, epsilon 1.0 -> 0.741 over 600 steps).
  - **RLlib** — needs `interop/wrappers.py`'s **`OneHotObs`**. The wrapper accepts our env and
    `check_multiagent_environments` passes, but building an algorithm dies with *"No default encoder
    config for obs space=Discrete(5)"*: Ray 2.57's API stack has no default encoder for a discrete
    observation space. One-hot into a `Box` and PPO trains. **The widening lives on RLlib's side
    deliberately** — `Discrete` is what the observation actually is and what PettingZoo, Gymnasium
    and SB3 all accept, so do not "fix" the adapters instead. SB3 does this same preprocessing
    internally without mentioning it, which is exactly why the cost was invisible until a second
    consumer was tried.
  - **MLPro, Gymnasium route** — works, but needs `interop/wrappers.py`'s **`register_gym_env`**.
    `WrEnvGYM2MLPro` reads `env.env.spec.id` and later calls `gymnasium.make` on it, so a bare
    `gymnasium.Env` subclass has no `.spec` and is refused. Conformance is not enough; it wants a
    registry entry. Also note `compute_reward()` takes **no arguments** there (passing states raises
    `NotImplementedError`; the reward is whatever the underlying step produced).
  - **MLPro, PettingZoo route — IMPOSSIBLE, do not spend time on it.** `WrEnvPZOO2MLPro` resolves the
    env class by name lookup inside a hardcoded `C_SUPPORTED_MODULES` list (`pettingzoo.classic` /
    `butterfly` / `atari` / `mpe` / `sisl`). Anything outside the PettingZoo distribution is rejected
    however well it conforms. No wrapper on our side can satisfy it.
  - **MLPro's declared deps are incomplete**: the bridge packages do not import until `dill` and
    `multiprocess` are installed by hand.
  - **Generalizable lesson, now in the paper**: conformance to a standard interface predicts a
    consumer will *accept* an env, not that it will *run* it, because consumers add requirements the
    standard never mentions. And a bridge that resolves envs by name against a fixed list is an
    integration with specific environments, not with an interface.
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
  asked for. **Pearson r added 2026-07-19** next to the scatter chart (guarded to n>=3 points with
  variance on both axes; below that it says why rather than printing a degenerate coefficient) —
  the page now does correlation, not just filtering-and-plotting.
- **New `/spacemap` tab (2026-07-19), delivered.** The parameter-space-map the analytics note above
  used to call "not-yet-built future work" — no longer true. Declares a grid (population size,
  MPCR, rounds, seed, and a roster-mix menu: all-Q/all-DQN/all-FEP/one-of-each), checks it against
  the ledger (`_cell_covered`: matches on every grid coordinate + exact roster kind-composition;
  hyperparameters are deliberately free — a custom-alpha run still covers its cell), and renders an
  n x MPCR coverage table per mix, invalid cells (violates `1/n < mpcr < 1`, or the mix is
  undefined at that n) shown as such rather than hidden. "Run missing cells now" runs a capped
  number of missing valid cells through the lean pipeline (match + metrics + ledger record, no
  results-page rendering) per click. `markov_brain` excluded from default mixes on purpose: an
  un-evolved random genome doesn't learn within a match, so it isn't a meaningful data point for a
  population-size/MPCR grid the way an online learner is. Tests: `tests/test_spacemap.py`.
  **Model handoff, for the record:** Claude Fable 5 implemented the PI/graph metrics, the
  Markov-brain visualization, the Pearson-correlation addition, and this space-map's entire
  app.py/template/CSS/test code, in that order — but stopped right after writing
  `tests/test_spacemap.py`, before running anything. Claude Sonnet 5 picked up from there: ran
  that test file, import-checked, restarted the dev server, did the live end-to-end verification
  above (real 2-cell grid run, ledger write confirmed, visible to /analytics), cleaned up test
  artifacts, re-ran the full regression suite, and did the commit+push for this feature and for
  this consolidated CLAUDE.md update itself.
- **2026-07-19, same session — the rest of the earlier punch-list closed out:**
  predictive-information + graph-theoretic metrics delivered (see section 5 above); the
  Markov-brain creature card now shows the evolved wiring/weight-matrix/TPM, not just state bits
  (`agents/markov_brain.py`'s `render_brain()` now exposes `W`/`bias`/`node_labels`); Q-learning/DQN
  cards gained a policy-vs-behavior explainability line + "how to read this" panels. What's left of
  that list: Honey-Jar Game #2 (design guidance now recorded, see section 6's Gap-paper addendum),
  the LLM agent, Battle of the Sexes, the custom game/metrics builder, the gamified/live-animation
  visualization, and the full decentralized network — all still genuinely not started.
- **Match-tab form rebuilt as a multi-step wizard (2026-07-22).** The five panels (Game, Roster,
  Metrics, Recording, Batch mode) are now one-at-a-time steps behind a sticky, clickable tab bar
  (`.fw-tabs`/`.fw-tab`, `mainWizardGoto(n)` in `form.html` — deliberately separate from the
  pre-existing quick-start modal's own `wizardStep(n)`, scoped by `.panel.fw-step` vs `#quickstart
  .wizard-step` so the two never collide) plus Back/Next buttons per step ("editorial-manager
  style," the user's own reference point). All fields stay in the DOM throughout (only
  `display:none` toggles), so form submission is unaffected — verified with a real end-to-end
  match run. New per-field icons (`glyph-clock`/`glyph-dice`/`glyph-percent`) on Rounds/Seed/MPCR
  and the batch sweep-field/values fields. **Scoped to the Match tab only** (explicit user
  decision) — the Evolutionary tab does not have this treatment yet. **Explicitly NOT built**:
  sliders for bounded numeric fields (0-1 reciprocity, 1-8 hidden nodes, etc.) — the user asked for
  this but said "let's think about it carefully" first (see
  [[gamebrains-webui-wizard-redesign]] in memory); don't add sliders without a fresh, explicit go.
  Also recorded but not started: evolutionary/GA optimization extended to RL/DQN/Transformer/LSTM
  substrates (not just Markov-brain animats), which the user called "the most important" of this
  batch and tied explicitly to the Bohm & Hintze/MABE lineage already cited in this file — see
  [[gamebrains-evolutionary-any-architecture-scope]] in memory for the open questions before this
  can be scoped into real work (what a "genome" means for a transformer/LSTM, compute cost).
- **Episodic games supported in the engine (2026-07-25).** Prerequisite for Game #2, done as its
  own commit ahead of it. `runner.run_match` previously called `game.reset()` exactly once and
  passed `StepResult.done` to the agents without ever acting on it, so a match was always a single
  run of `rounds` steps. It now treats `rounds` as a total round *budget*: when a game reports
  `done`, the episode is recorded and `game.reset()` starts the next one, until the budget runs
  out. A match may therefore contain many episodes, which is what the congestion family needs
  (agents race to a terminal, episode ends, next contest begins).
  - New `episode` event type in the log (documented in `eventlog.py`'s docstring alongside the
    others). It carries per-agent reward totals and the episode's round count, both computed by
    the runner, merged with whatever the game itself put in `StepResult.info["episode"]`. That
    split keeps the runner game-agnostic: for the congestion game the payload will be the
    per-agent reach vector, but the runner never needs to know that. `run_match` also returns
    the same records under a new `"episodes"` key (purely additive, existing keys untouched).
  - **This is the series ALT/RP consume.** Those metrics ask who won and when, per *episode*, and
    a per-round log cannot answer that, so without this event they cannot be computed at all.
  - **Two deliberate edge-case choices**, both tested: (1) no reset fires on the final round, since
    that would leave the game rewound behind the caller's back for no benefit (and PGG reports
    `done` exactly there, so this is the common case, not an exotic one); (2) a trailing partial
    episode, where the budget runs out mid-contest, is *not* reported, because a truncated contest
    has no winner and would corrupt the alternation series.
  - Regression risk was concentrated on `PublicGoodsGame`, which does set `done` on its last round
    (`public_goods.py:78`) rather than never. Verified: same seed still reproduces identical
    actions/rewards/cooperators, the `round` events are byte-identical, only the opening reset
    runs, and the whole match now correctly reports as exactly one episode. Tests:
    `tests/test_episodic.py` (4 cases). Full suite re-run green, including `test_evolution`, which
    drives fitness through `run_match` and is the most exposed caller.
- **`config_hash` gained a third tier: `protocol` (2026-07-25).** Found while adding
  `partial_episode` to the runner (see the episodic entry above): that flag changes the reported
  numbers but lives in neither `game.describe()` nor the roster, so it entered no hash at all. Two
  runs identical in game, roster and seed but scored under different conventions therefore shared
  a `config_hash`, which would have made the Smart Filter offer one as a reusable answer for the
  other and the ledger call them replications. Harmless locally, wrong the moment records are
  shared, which is the whole point of the repository layer.
  - **The rule, now documented in `docs/repository-schema.md` §4a and enforced in
    `normalize.build_config`:** *Design* (game, roster, code_version) is hashed. *Protocol*
    (scoring conventions that move a reported number without changing the game or the agents) is
    hashed. *Scale* (rounds, seed) is not, since the `extends`/`replicates` semantics are defined
    as "same hash, differing in exactly those". The test for the middle tier is whether it can
    change a number that gets reported; a choice that only changes what you *look at* (the metric
    checkboxes, the Nash/Phi toggles) is in no tier and stays out of the record's identity.
  - **Three copies of the hashed-config literal existed**, which is exactly how a new tier gets
    added in one place and missed in the others: `build_config`, `Ledger.append`, and
    `Ledger.find_exact`. The latter two now call `build_config`, so the write path and the reuse
    lookup cannot drift. `find_exact`/`smart_filter.lookup` take `protocol` too, or the lookup
    would contradict the way the record it finds was hashed.
  - `record.protocol_from_run(run_result)` reads the conventions back off the runner's own output
    instead of having each call site restate them, so a recorded protocol cannot drift from the
    one that produced the numbers. It is the single place to extend when the congestion game adds
    its own. All three `record_experiment` call sites (run_pgg, the webui match, the space-map
    cell runner) go through it.
  - **Migration was free and is now closed:** the local ledger held 2 records. Verified afterwards
    on a real recorded match that the chain still verifies with mixed records, old ones (no
    `protocol` field) still validating individually since a signature only covers its own payload.
    Doing this after federation would have meant migrating other people's data.
  - Tests: `tests/test_record.py` gained two cases (different protocols must not collide, and
    `find_exact` must agree with how records were hashed; plus `protocol_from_run`'s defaulting).
- **Game #2 and the temporal-fairness metrics wired into the webui (2026-08-10).** Until this,
  everything from the congestion family onward ran only from code. Now:
  - The game selector offers the five congestion presets alongside the Public Goods Game, each
    labelled with whether it is a published configuration. `_GAME_ROADMAP` no longer claims the
    Honey-Jar Game is "coming soon", since it has arrived; Battle of the Sexes and the custom
    builder took its place there.
  - A "Congestion family" fieldset appears only for those games (and MPCR hides, since it belongs
    to the Public Goods Game alone), exposing every axis: corridor length with the one-shot dial
    explained inline, reward rule with published-vs-exploratory marked in the option labels,
    memory depth, episode cap, full reward, and the zero-floor toggle. All fields blank by
    default, meaning "keep the preset", so a published configuration stays one click away.
  - New "Temporal fairness (ALT & RP)" panel on the results page, with all 16 measures, the
    episode breakdown (won alone / collision / nobody arrived), and a help drawer explaining why
    outcome fairness cannot see turn-taking. **Shown only when the run produced more than one
    episode**, since a single-stage game has no sequence to alternate over and a confident zero
    would be worse than an absent panel.
  - **Three couplings had to be broken.** `CongestionGame` gained `state_labels()` (the corridor
    as digits per agent, plus the remembered arrival bits, falling back to bare indices past 512
    states since a 60-million-row Q-table view helps nobody) and `max_welfare_per_round()` (one
    agent arriving alone, spread over the `num_positions - 1` rounds that takes, which is *not*
    the papers' per-episode Efficiency). Nash detection now says plainly that it is built on the
    Public Goods Game's closed-form payoff and has no normal form for an episodic game, rather
    than failing.
  - **A name collision worth knowing about:** both `social.py` and `social_alt.py` produce
    "efficiency", per round and per episode respectively. Different quantities, so the
    alternation one is renamed `alt_efficiency` at the merge point rather than silently
    overwriting.
  - Verified end-to-end: a real 3000-round HJG match through the form produced 1411 episodes, 487
    won alone against 924 collisions, and the panel showed the finding directly, Reward Fairness
    0.9131 and Turn-Taking Fairness 0.9595 sitting beside CALT 0.1840 and AALT 0.1701. Public
    Goods Game regression confirmed: still runs, still gets its Nash panel, correctly gets no
    alternation panel.
- **Second theme, "Horsey Lab", + a Scientific/Gamified toggle (2026-07-25).** The dark console
  theme above is no longer the only skin. A bright, deliberately goofy alternative ("Horsey Lab":
  cream background, white sticker-panels with thick plum borders and hard offset shadows, saturated
  agent-kind colors, rounded display face) now ships alongside it, chosen by a header toggle that
  persists in a `gb_theme` cookie. **Neither theme replaces the other** (explicit user decision).
  Implementation notes worth knowing before touching this:
  - The whole skin is one `[data-theme="gamified"]` block in `base.html` that **redefines the same
    `:root` variables** the console theme already used, so most existing rules repaint for free.
    The cookie is read by a tiny inline script at the very top of `<body>`, before any content
    renders, so there is no flash of the wrong theme.
  - **The gotcha, found only by looking at the real page:** roughly a dozen rules had a *literal*
    dark-navy hex (`#262c47`, `#232941`, `#171c30`, `#2a3050`, `#20263f`) rather than a variable,
    for what the console theme treats as a "recessed surface inside a panel" (secondary buttons,
    creature cards, roster rows, code chips, `.badge.todo`, belief-bar and mbrain-bit tracks,
    `details.explain`). Redefining variables did nothing for those, so they stayed dark under a
    light background: unreadable dark-on-dark. Each one now has an explicit gamified override.
    **If you add a new rule with a hardcoded dark color, it will silently break the light theme
    the same way** — prefer a variable, or add the override in the same commit.
  - `render_creature()` in `app.py` now also returns a `mascot` key: a per-kind cartoon-face SVG
    (`_mascot_svg`), rendered by `results.html` into `.creature-mascot`, which is `display:none`
    under Scientific and shown under Gamified. Shape encodes the architecture (squircle = the
    Q-table, rounded rhombus = a network node, circle = an animat, rounded inverted triangle = a
    belief funnel, and a face-free quadcopter drone for the fixed-rule classic strategies, since a
    lookup rule has no mind to depict). Its `--psy`/`--sun` accents and stroke color are fixed, not
    theme-driven, because it only ever renders on the bright background.
  - Verified end-to-end on a real 5-agent mixed-roster match (one of each kind): all 5 mascots
    render, each with the right letter marker (Q/D/M/F, drone has none), and the three webui-facing
    test modules still pass.

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
- Existing paper sources: `../CONFERENCE PAPER GAMEBRAINS/death/root.tex` (venue submission,
  co-authored), `../CONFERENCE PAPER GAMEBRAINS/arxiv/root.tex` (preprint, single-authored). Fig. 1
  (architecture) is now a TikZ diagram in-source, not the old `gamebrains_architecture.png`.

## 9f. Reproducibility and interop hardening (2026-08-12, for the SPE submission)

Driven by revising `../SPE GAMEBRAINS/root.tex` for submission. See that folder's
`REVISION-NOTES.md` for the paper-side decisions; this section records only what changed in code.

**Seven new files, no tracked file's behaviour changed.**

- **`interop/wrappers.py`** — `OneHotObs` and `register_gym_env`, the two shims the downstream
  ecosystems turned out to need. See the corrected §9c entry above for why each exists. The
  Discrete observation space stays correct in the adapters; only consumers who need widening pay
  for it.
- **`tests/test_interop_sb3.py` / `_rllib.py` / `_mlpro.py`** — the three downstream integrations,
  tested rather than asserted. Every module SKIPs with exit 0 and a clear install hint when its
  optional library is absent, so the default suite still runs on a machine without ray/sb3/mlpro
  (verified by blocking the imports via a `sys.meta_path` hook).
  - **Two of these tests assert a FAILURE on purpose**, which is unusual enough to flag:
    `test_raw_discrete_obs_is_refused` and `test_bare_gymnasium_env_is_refused`. They pin down *why*
    the two shims exist. If a future Ray or MLPro release relaxes its requirement, those tests fail,
    and that failure is the signal to simplify both `wrappers.py` and the paper's interoperability
    section. Do not "fix" them by deleting the assertion.
- **`tests/run_all.py`** — runs every `test_*.py` via `runpy` so there is one definition of what a
  module's tests are. `--skip-slow` omits the two genuinely slow modules. **The `SLOW` set is
  measured, not guessed**: a first version listed `test_pyphi_fork` (0.44s) and `test_evolution`
  (0.20s) while omitting `test_markov_brain` (15.7s), which made the flag nearly useless. Re-measure
  from the runner's own per-module timings before editing it. Current: 20/20 in ~37s, or 18/18 in
  ~12s with the flag.
- **`experiments/run_multiseed.py`** — the paper's two headline experiments over N seeds with 95%
  intervals. Regenerates Table 4 and Appendix C. Takes `--seeds` and `--lengths` so the trend can be
  confirmed cheaply before the full 30-seed, five-length run.
- **`experiments/run_bakeoff.py`** — train each architecture separately, freeze all, play one match.
  Regenerates Appendix E. The `Frozen` wrapper duplicates the webui's own bake-off construction
  rather than importing from `webui/`, since `experiments/` must not depend on the web layer.

**Why these last two exist at all, and the lesson.** The paper's tables had been generated by
throwaway scratchpad scripts, so nothing in the repository reproduced them, while the paper's own
appendix listed commands (`tests.run_all`, `experiments.run_congestion`) that **did not exist**.
A paper claiming reproducibility has to be checkable from the repository alone. When a paper table is
produced, the producing script belongs in `experiments/` in the same commit.

**Findings about third-party code, recorded so nobody re-derives them:** RLlib refuses `Discrete`
observations outright; MLPro needs a Gymnasium registry entry rather than mere conformance, its
`compute_reward()` takes no arguments, its PettingZoo bridge is closed to third-party envs by
construction, and its bridge packages do not import until `dill` and `multiprocess` are added by
hand. All four are asserted in tests, not just written down here.
