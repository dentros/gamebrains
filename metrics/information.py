"""
Mutual information and transfer entropy for the Public Goods Game (CLAUDE.md section 5's
Theory-of-Mind-relevant metric family: MI/TE quantify whether one agent's behavior is
statistically explained by, or predictive of, another signal in the match).

Operates on the same per-round `records` dict produced by engine.runner.run_match and consumed by
metrics/social.py (`actions`: (rounds, n_agents) int array; `cooperators`: (rounds,) int array).
No engine changes were needed: PublicGoodsGame.step() broadcasts the identical `k_prev` observation
to every agent (public information, see games/public_goods.py), so that shared observation series
is reconstructed here directly from `cooperators`, one round shifted.

Both metrics use a discrete plug-in (maximum-likelihood) entropy estimator with a Miller-Madow bias
correction -- appropriate because every symbol involved (observations, actions) is already a small
discrete integer in this game, so no continuous-valued estimator (KSG) or neural training surrogate
(MINE/InfoNCE) is needed. Per the project's own information-theoretic guardrails
(arXiv:2604.23716, cited in CLAUDE.md section 5): this is a calibrated *estimator*, not a training
surrogate, and every reported transfer-entropy value ships with its estimator parameters
(alphabet, sample size) and a surrogate-based significance test -- never a bare point number.

## The training transient, which this module used to ignore

**A significance test on the whole match is worthless, and we measured how worthless.** Every
learning agent in a roster typically shares an exploration schedule, so every agent's action
distribution drifts in the same direction over the same rounds whether or not any of them is
reacting to the others. A permutation surrogate cannot null that out, because it destroys the very
temporal structure the shared trend lives in. The result is directed influence detected between
agents that never met.

Measured on this platform, on a null control of agents drawn from *separate* matches, so that every
"significant" pair is a false positive by construction: **35 of 40 pairs flagged, 87.5%, against a
nominal 5%.** Cutting the first half of the series drops it to 47.5%, which is the second half of
the finding: excluding the transient is necessary and not sufficient. Both figures are consistent
with the authors' companion validation study, which reports 100% and 99.9% for the unexcluded case
across 100 seeds and two game families.

So this module now does three things it did not:

1. **Derives the burn-in from the roster rather than taking a constant.** An epsilon schedule states
   exactly when exploration stops changing, so `suggested_burn_in` solves for it. A magic number
   would be wrong for any roster but the one it was tuned on.
2. **Tests the analysed window for stationarity** (ADF and KPSS together, `stationarity_report`),
   because the residual 47.5% is precisely the part a burn-in cannot fix.
3. **Withholds the aggregate when it cannot be justified, and says why.** The per-pair detail is
   always returned, with its own verdict attached. What is withheld is the single number that
   travels into ledgers, figures and papers, since that is the one nobody re-derives.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np


def _entropy_bits_mm(symbols: np.ndarray) -> float:
    """Miller-Madow-corrected discrete plug-in entropy, in bits.

    The plug-in estimator H_hat = -sum(p_hat * log2(p_hat)) is negatively biased for finite
    samples; Miller-Madow adds the standard first-order correction (K-1)/(2*N*ln 2), K = number of
    distinct symbols actually observed, N = sample size. `symbols` must be non-negative ints
    (a `np.bincount` requirement) -- every caller below encodes its alphabet that way.
    """
    n = symbols.size
    counts = np.bincount(symbols)
    counts = counts[counts > 0]
    p = counts / n
    h_plugin = float(-np.sum(p * np.log2(p)))
    k = counts.size
    correction = (k - 1) / (2.0 * n * np.log(2))
    return h_plugin + correction


def _reconstruct_observations(cooperators: np.ndarray, n_agents: int) -> np.ndarray:
    """The shared pre-action observation every agent actually saw each round (previous round's
    cooperator count, or the `start` sentinel for round 0) -- recovered purely from the logged
    `cooperators` series, valid because the observation is public/identical for every agent (see
    module docstring), so no engine change is needed to recover it after a match has already run.
    """
    rounds = cooperators.shape[0]
    obs = np.empty(rounds, dtype=np.int64)
    obs[0] = n_agents + 1  # matches PublicGoodsGame.start_state
    if rounds > 1:
        obs[1:] = cooperators[:-1]
    return obs


def _mi_bits(x: np.ndarray, y: np.ndarray) -> float:
    """I(X;Y) in bits, via H(X) + H(Y) - H(X,Y), each term Miller-Madow-corrected. `y` must be
    non-negative ints; its observed cardinality is used to build a unique joint symbol for (x, y).
    """
    x = x.astype(np.int64)
    y = y.astype(np.int64)
    y_card = int(y.max()) + 1
    joint = x * y_card + y
    return _entropy_bits_mm(x) + _entropy_bits_mm(y) - _entropy_bits_mm(joint)


def mutual_information_per_agent(records: dict[str, np.ndarray]) -> dict[str, Any]:
    """I(observation; action) for every agent: does the public previous-round signal statistically
    explain this agent's current action? A near-zero value means the agent's action looks
    independent of that signal (e.g. a fixed strategy, or an online learner that hasn't picked up
    on it yet); a value near the action's own entropy means the action is (close to) a deterministic
    function of the observation.
    """
    actions = records["actions"]
    cooperators = records["cooperators"]
    rounds, n_agents = actions.shape
    obs = _reconstruct_observations(cooperators, n_agents)
    per_agent = [_mi_bits(obs, actions[:, i]) for i in range(n_agents)]
    return {
        "bits_by_agent": per_agent,
        "mean_bits": float(np.mean(per_agent)) if per_agent else 0.0,
        "n_samples": int(rounds),
        "estimator": "discrete plug-in + Miller-Madow correction",
    }


def predictive_information_per_agent(records: dict[str, np.ndarray]) -> dict[str, Any]:
    """One-step predictive information PI = I(action_{t-1}; action_t) for every agent: how much
    of an agent's own next action is already determined by its own previous one. The "bridge"
    measure of the MI/TE/PI family (see module docstring): a fixed strategy like AllC scores 0
    (its action carries no information because it never varies), an alternator scores ~1 bit, an
    iid random agent ~0. Uses the same lag-1 convention as `transfer_entropy_pairwise` so the two
    are directly comparable (TE measures cross-agent prediction, PI self-prediction, at the same
    timescale)."""
    actions = records["actions"]
    rounds, n_agents = actions.shape
    if rounds < 2:
        return {"bits_by_agent": [0.0] * n_agents, "mean_bits": 0.0, "n_samples": 0,
                "estimator": "discrete plug-in + Miller-Madow correction"}
    per_agent = [_mi_bits(actions[:-1, i], actions[1:, i]) for i in range(n_agents)]
    return {
        "bits_by_agent": per_agent,
        "mean_bits": float(np.mean(per_agent)) if per_agent else 0.0,
        "n_samples": int(rounds - 1),
        "estimator": "discrete plug-in + Miller-Madow correction",
    }


# --- the guardrail -------------------------------------------------------------------------------

#: Significance level for the stationarity diagnostics, and for calling a transfer-entropy pair
#: significant. Kept as one constant because a reader comparing the two should see they agree.
ALPHA = 0.05


def suggested_burn_in(roster: list[Any] | None, rounds: int) -> tuple[int, str]:
    """How many opening rounds to discard, solved from the roster's own exploration schedules.

    An epsilon-greedy schedule states exactly when exploration stops changing: multiplicative decay
    from `epsilon_start` reaches `epsilon_min` after ``log(min/start) / log(decay)`` rounds. Every
    scheduled agent must have got there, so the burn-in is the maximum over the roster.

    This is derived rather than configured on purpose. A constant is wrong for every roster but the
    one it was tuned on, and it is silently wrong, since nothing in the output says which regime the
    numbers came from. Returns the round count and a sentence explaining it, because a discarded
    window that cannot explain itself is indistinguishable from a bug.
    """
    if not roster:
        return 0, ("no roster supplied, so no exploration schedule could be read and no burn-in "
                   "was applied; the estimate covers the whole match including any training "
                   "transient")

    longest = 0
    who = ""
    for agent in roster:
        start = getattr(agent, "epsilon_start", None)
        floor = getattr(agent, "epsilon_min", None)
        decay = getattr(agent, "epsilon_decay", None)
        if start is None or floor is None or decay is None:
            continue
        if not (0.0 < decay < 1.0) or start <= floor or floor <= 0.0:
            continue                      # constant or degenerate schedule: nothing to wait for
        needed = int(np.ceil(np.log(floor / start) / np.log(decay)))
        if needed > longest:
            longest, who = needed, getattr(agent, "name", "?")

    if longest == 0:
        return 0, ("no agent in the roster has a decaying exploration schedule, so there is no "
                   "training transient of this kind to exclude")
    if longest >= rounds:
        return longest, (f"the slowest schedule ({who}) needs {longest:,} rounds to reach its "
                         f"exploration floor and the match is only {rounds:,}, so no part of this "
                         f"run is post-transient")
    return longest, (f"first {longest:,} of {rounds:,} rounds discarded, the point at which the "
                     f"slowest exploration schedule in the roster ({who}) reaches its floor")


def stationarity_report(series: np.ndarray) -> dict[str, Any]:
    """ADF and KPSS on one series, reported together because they test opposite nulls.

    ADF's null is that a unit root is present, so a small p favours stationarity. KPSS's null is
    that the series is stationary, so a small p favours non-stationarity. Running both and
    requiring agreement is stricter than either alone, and the disagreement cases are informative
    rather than embarrassing.

    Note what a pass does and does not mean. Failing to reject a null is not evidence for it, so
    "stationary (both agree)" means the window survived two tests designed to catch the failure
    that matters here, not that stationarity has been established.
    """
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
    except ImportError:
        return {"available": False, "verdict": "not tested",
                "note": "statsmodels is not installed, so stationarity could not be checked"}

    x = np.asarray(series, dtype=float)
    if x.size < 20:
        return {"available": True, "verdict": "too short",
                "note": f"{x.size} samples is too few to test"}
    if np.all(x == x[0]):
        return {"available": True, "verdict": "constant",
                "note": "the series never changes, so the question does not arise"}

    with warnings.catch_warnings():
        # KPSS warns whenever its p-value falls outside the published lookup table. That is a
        # statement about table resolution, not about this series, and it fires constantly.
        warnings.simplefilter("ignore")
        adf_p = float(adfuller(x, autolag="AIC")[1])
        kpss_p = float(kpss(x, regression="c", nlags="auto")[1])

    adf_says_stationary = adf_p < ALPHA        # rejects the unit root
    kpss_says_stationary = kpss_p >= ALPHA     # fails to reject stationarity

    if adf_says_stationary and kpss_says_stationary:
        verdict = "stationary"
    elif not adf_says_stationary and not kpss_says_stationary:
        verdict = "non-stationary"
    else:
        verdict = "inconclusive"

    return {"available": True, "verdict": verdict, "adf_p": adf_p, "kpss_p": kpss_p,
            "alpha": ALPHA, "n": int(x.size),
            "note": f"ADF p={adf_p:.4f}, KPSS p={kpss_p:.4f}"}


def window_is_trustworthy(records: dict[str, np.ndarray]) -> dict[str, Any]:
    """Is every agent's action series in this window stationary enough to test for influence?

    One non-stationary series is enough to spoil the pairs it appears in, so the window is judged
    on its worst member rather than on an average.
    """
    actions = records["actions"]
    per_agent = [stationarity_report(actions[:, i]) for i in range(actions.shape[1])]
    verdicts = [r["verdict"] for r in per_agent]

    if "non-stationary" in verdicts:
        ok, why = False, (f"{verdicts.count('non-stationary')} of {len(verdicts)} agent action "
                          f"series are non-stationary after burn-in exclusion")
    elif "inconclusive" in verdicts:
        ok, why = False, (f"{verdicts.count('inconclusive')} of {len(verdicts)} agent action "
                          f"series gave disagreeing stationarity tests")
    elif not per_agent or not per_agent[0].get("available", False):
        ok, why = False, per_agent[0]["note"] if per_agent else "no series to test"
    else:
        ok, why = True, f"all {len(verdicts)} agent action series passed ADF and KPSS"

    return {"ok": ok, "reason": why, "per_agent": per_agent}


def _te_bits(x_prev: np.ndarray, y_prev: np.ndarray, y_t: np.ndarray) -> float:
    """T(X->Y) at lag 1 = I(Y_t ; X_prev | Y_prev), via the standard conditional-MI identity
    T = H(Y_t,Y_prev) + H(X_prev,Y_prev) - H(Y_prev) - H(Y_t,X_prev,Y_prev). All three inputs are
    binary action symbols (0/1) here, so each joint alphabet is fixed and small (<=8 symbols for
    the triple) -- encoded directly rather than through a generic mixed-radix helper.
    """
    x_prev = x_prev.astype(np.int64)
    y_prev = y_prev.astype(np.int64)
    y_t = y_t.astype(np.int64)
    h_y_prev = _entropy_bits_mm(y_prev)
    h_yt_yprev = _entropy_bits_mm(y_t * 2 + y_prev)
    h_xprev_yprev = _entropy_bits_mm(x_prev * 2 + y_prev)
    h_yt_xprev_yprev = _entropy_bits_mm(y_t * 4 + x_prev * 2 + y_prev)
    return h_yt_yprev + h_xprev_yprev - h_y_prev - h_yt_xprev_yprev


def transfer_entropy_pairwise(
    records: dict[str, np.ndarray], seed: int = 0, n_surrogates: int = 200,
    burn_in: int = 0,
) -> dict[str, Any]:
    """T(agent_i -> agent_j) at lag 1 for every ordered pair, with a surrogate significance test.

    `n_surrogates` independent **random permutations** of the source's action history are
    substituted for the real one, destroying any genuine temporal link to the target while
    preserving both agents' marginal action statistics, and `p_value` is the fraction of surrogates
    whose TE meets or exceeds the observed value. Raw TE is reported alongside that test and never
    in place of it: with a finite sample, plug-in TE is never exactly zero even between agents with
    no causal link, so an unvalidated point value reads as detected influence when it is sampling
    noise or a shared confound.

    **`burn_in` is not optional in any meaningful sense**, and the default of 0 exists only so the
    function stays callable on a series a caller has already trimmed. A permutation surrogate
    cannot null out a trend the whole roster shares, so testing across a training transient reports
    influence between agents that never met. See the module docstring for what that costs measured
    (87.5% false positives against a nominal 5%). `compute_all` derives the value from the roster;
    pass it here only if you are trimming deliberately.

    `p_value < ALPHA` flags a pair as significant, uncorrected for the multiple comparisons across
    a roster's pairs. That is a documented simplification rather than an oversight: a small roster
    keeps the pair count low, and a Bonferroni or FDR correction is future work.
    """
    actions = records["actions"]
    if burn_in:
        actions = actions[burn_in:]
    rounds, n_agents = actions.shape
    rng = np.random.default_rng(seed)
    by_pair: dict[tuple[int, int], dict[str, Any]] = {}

    if rounds < 2:
        return {
            "by_pair": {}, "mean_bits": 0.0, "mean_significant_bits": 0.0,
            "n_significant_pairs": 0, "n_pairs": 0,
            "estimator": "discrete plug-in + Miller-Madow correction", "n_surrogates": n_surrogates,
        }

    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                continue
            x_prev = actions[:-1, i]
            y_prev = actions[:-1, j]
            y_t = actions[1:, j]
            observed = _te_bits(x_prev, y_prev, y_t)
            surrogates = np.empty(n_surrogates)
            for s in range(n_surrogates):
                surrogates[s] = _te_bits(rng.permutation(x_prev), y_prev, y_t)
            p_value = float((np.sum(surrogates >= observed) + 1) / (n_surrogates + 1))
            by_pair[(i, j)] = {
                "bits": observed, "p_value": p_value, "n_surrogates": n_surrogates,
                "surrogate_mean_bits": float(surrogates.mean()),
                "surrogate_std_bits": float(surrogates.std()),
            }

    all_bits = [r["bits"] for r in by_pair.values()]
    sig_bits = [r["bits"] for r in by_pair.values() if r["p_value"] < 0.05]
    return {
        "by_pair": by_pair,
        "mean_bits": float(np.mean(all_bits)) if all_bits else 0.0,
        "mean_significant_bits": float(np.mean(sig_bits)) if sig_bits else 0.0,
        "n_significant_pairs": len(sig_bits),
        "n_pairs": len(by_pair),
        "estimator": "discrete plug-in + Miller-Madow correction",
        "n_surrogates": n_surrogates,
        "burn_in": int(burn_in),
        "rounds_analysed": int(rounds),
    }


def compute_all(records: dict[str, np.ndarray], seed: int = 0, n_surrogates: int = 200,
                roster: list[Any] | None = None) -> dict[str, Any]:
    """Entry point mirroring metrics/social.py's `compute_all`.

    The headline scalars are what `_METRIC_META` displays and what a ledger record stores;
    `*_detail` carries the full per-agent and per-pair breakdown for the results page. The
    transfer-entropy headline averages only over pairs that passed the surrogate test, since
    averaging in non-significant pairs would let sampling noise set the number.

    **Pass `roster`.** It is how the burn-in gets derived, and without it the transfer-entropy
    analysis runs over the training transient and is not reported as a headline number at all.
    That is deliberate and it is the whole point of this function's guardrail: see the module
    docstring for the measured false-positive rate that justifies it.

    Three conditions must hold before `transfer_entropy_bits` is published, and each failure is
    reported by name in `transfer_entropy_bits_status` rather than by an absent field:

      1. a burn-in could be derived, so we know which regime the window is in
      2. rounds remain after discarding it
      3. every agent's action series in that window passes both stationarity tests

    Failing any of them leaves `transfer_entropy_bits` as ``None`` and fills
    `transfer_entropy_bits_status` with the reason. A number that is absent without explanation is
    indistinguishable from a bug, which is why the reason travels with the absence into the ledger.
    """
    mi = mutual_information_per_agent(records)
    pi = predictive_information_per_agent(records)

    rounds = int(records["actions"].shape[0])
    burn_in, burn_in_note = suggested_burn_in(roster, rounds)

    if burn_in >= rounds:
        te = transfer_entropy_pairwise(records, seed=seed, n_surrogates=n_surrogates)
        te["window"] = {"ok": False, "reason": burn_in_note}
        headline, status = None, burn_in_note
    else:
        te = transfer_entropy_pairwise(records, seed=seed, n_surrogates=n_surrogates,
                                       burn_in=burn_in)
        trimmed = {"actions": records["actions"][burn_in:]}
        window = window_is_trustworthy(trimmed)
        window["burn_in_note"] = burn_in_note
        te["window"] = window

        if burn_in == 0 and roster is None:
            headline, status = None, burn_in_note
        elif not window["ok"]:
            headline, status = None, window["reason"]
        else:
            headline = te["mean_significant_bits"]
            status = f"{burn_in_note}; {window['reason']}"

    return {
        "mutual_information_bits": mi["mean_bits"],
        "mutual_information_detail": mi,
        "transfer_entropy_bits": headline,
        "transfer_entropy_bits_status": status,
        "transfer_entropy_detail": te,
        "predictive_information_bits": pi["mean_bits"],
        "predictive_information_detail": pi,
    }
