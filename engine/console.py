"""
Live terminal console — the "error/debug log που φαίνεται όταν τρέχει κάτι".

Prints a readable, lightly gamified trace of a match: a header, throttled per-round lines
(each agent's move as 🟢/🔴, cooperator count, mean payoff, exploration), periodic brain
snapshots (e.g. a Q-table), evolution generation summaries, and a final metrics box.

It consumes the same information the event-log carries, so the eventual web "houses" UI can
render the exact same stream. Emoji are used when the terminal can encode them, with an ASCII
fallback so it never crashes on a legacy Windows code page.
"""

from __future__ import annotations

import sys
from typing import Any, Sequence


class LiveConsole:
    def __init__(self, use_emoji: bool = True) -> None:
        # Best-effort: switch stdout to UTF-8 so emoji render on Windows; never fatal.
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass
        self.use_emoji = use_emoji and self._can_emoji()
        if self.use_emoji:
            self.coop_sym, self.defect_sym = "🟢", "🔴"
        else:
            self.coop_sym, self.defect_sym = "C", "D"

    @staticmethod
    def _can_emoji() -> bool:
        try:
            "🟢".encode(sys.stdout.encoding or "ascii")
            return True
        except Exception:
            return False

    # --- sections -----------------------------------------------------------------

    def rule(self, char: str = "─", width: int = 72) -> None:
        print(char * width)

    def header(self, game_desc: dict[str, Any], roster: Sequence[Any], seed: int, rounds: int) -> None:
        self.rule("═")
        print(f"  GameBrains · {game_desc.get('name', '?')} · n={game_desc.get('n_agents')} "
              f"· {rounds} rounds · seed={seed}")
        if "mpcr" in game_desc:
            print(f"  MPCR={game_desc['mpcr']}  cost={game_desc.get('cost')}  "
                  f"actions={game_desc.get('action_names')}")
        self.rule("═")
        print("  Roster:")
        for i, ag in enumerate(roster):
            print(f"    [{i}] {ag.name:<22} kind={getattr(ag, 'kind', '?'):<12} "
                  f"mode={getattr(ag, 'training_mode', '?')}")
        self.rule()

    def action_glyphs(self, actions: Sequence[int]) -> str:
        return "".join(self.coop_sym if a == 1 else self.defect_sym for a in actions)

    def round_line(
        self,
        rnd: int,
        actions: Sequence[int],
        cooperators: int,
        mean_reward: float,
        extra: str = "",
    ) -> None:
        n = len(actions)
        bar = self.action_glyphs(actions)
        print(f"  r{rnd:>6} {bar}  coop {cooperators}/{n}  "
              f"x̄payoff {mean_reward:+.3f}{('  ' + extra) if extra else ''}")

    def brain_snapshot(self, agent: Any, title: str = "") -> None:
        brain = agent.render_brain()
        head = f"  🧠 {agent.name}" + (f" · {title}" if title else "")
        print(head)
        kind = brain.get("kind")
        if kind == "qlearning":
            self._print_qtable(brain)
        elif kind == "dqn":
            layers = brain.get("layers", [])
            print(f"     net {'-'.join(map(str, layers))}  loss={brain.get('last_loss', 0):.4f}  "
                  f"dev={brain.get('device', 'cpu')}")
            self._print_qtable(brain, q_key="q_values")
        elif kind == "fep":
            self._print_beliefs(brain)
        else:
            print(f"     {brain}")

    def _print_beliefs(self, brain: dict[str, Any]) -> None:
        belief = brain.get("belief", [])
        labels = brain.get("state_labels") or [f"s{i}" for i in range(len(belief))]
        probs = brain.get("action_probs", {})
        print(f"     belief P(others cooperate)   E[others]={brain.get('expected_others', 0):.2f}")
        for lab, p in zip(labels, belief):
            bar = "█" * int(round(p * 24)) if self.use_emoji else "#" * int(round(p * 24))
            print(f"     {lab:>10} {p:5.2f} {bar}")
        if probs:
            print(f"     -> P(Cooperate)={probs.get('Cooperate', 0):.2f}  "
                  f"P(Defect)={probs.get('Defect', 0):.2f}")

    def _print_qtable(self, brain: dict[str, Any], q_key: str = "q_table") -> None:
        q = brain[q_key]
        states = brain.get("state_labels") or [f"s{i}" for i in range(len(q))]
        actions = brain.get("action_labels") or [f"a{j}" for j in range(len(q[0]))]
        policy = brain.get("greedy_policy")
        w = max(len(a) for a in actions)
        print(f"     ε={brain.get('epsilon', 0):.3f}   "
              f"state -> [{'  '.join(f'{a:>{w}}' for a in actions)}]  greedy")
        for s, row in enumerate(q):
            vals = "  ".join(f"{v:>{max(w, 6)}.2f}" for v in row)
            pick = actions[policy[s]] if policy is not None else ""
            print(f"     {states[s]:>7} -> [{vals}]  {pick}")

    def leaderboard(self, roster: Sequence[Any], cumulative_payoffs: Sequence[float]) -> None:
        """Rank the roster by total payoff earned over the match — 'which agent did best'."""
        self.rule("═")
        print("  LEADERBOARD (by cumulative payoff)")
        self.rule()
        ranked = sorted(zip(roster, cumulative_payoffs), key=lambda pair: pair[1], reverse=True)
        for rank, (agent, payoff) in enumerate(ranked, start=1):
            mode = getattr(agent, "training_mode", "?")
            print(f"    {rank:>2}. {agent.name:<22} kind={agent.kind:<12} mode={mode:<12} "
                  f"total payoff {payoff:+.2f}")
        self.rule("═")

    def generation(self, gen: int, avg: float, best: float, worst: float, extra: str = "") -> None:
        print(f"  gen {gen:>4}  fitness  avg {avg:+.3f}  best {best:+.3f}  worst {worst:+.3f}"
              f"{('  ' + extra) if extra else ''}")

    def summary(self, metrics: dict[str, Any]) -> None:
        self.rule("═")
        print("  RESULTS")
        self.rule()
        order = ["cooperation_rate", "efficiency", "payoff_gini", "action_entropy_bits"]
        pretty = {
            "cooperation_rate": "Cooperation rate",
            "efficiency": "Efficiency (vs all-cooperate)",
            "payoff_gini": "Payoff Gini (0=equal)",
            "action_entropy_bits": "Action entropy (bits)",
        }
        for key in order:
            if key in metrics:
                print(f"    {pretty[key]:<32} {metrics[key]:.4f}")
        self.rule("═")
