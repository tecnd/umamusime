"""Parameter sweeps for MCTS. Does not change the search algorithm.

Writes one JSON object per game. Seeds are the environment seed passed to
`mcts.play`; reruns skip seeds already recorded for that arm.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")

from open_spiel.python.algorithms import mcts as openspiel_mcts

from .bots.mcts import PlayTrace, play

RECORD_TURNS = (29, 43)


@dataclass(frozen=True)
class Arm:
    name: str
    uct_c: float = 200.0
    max_simulations: int = 100
    n_rollouts: int = 15
    solve: bool = True
    dont_return_chance_node: bool = False
    use_puct: bool = False


def play_record(
    seed: int, arm: Arm, decision_limit: int | None = None
) -> dict[str, Any]:
    trace = PlayTrace()
    started = time.perf_counter()
    state, _actions = play(
        verbose=False,
        seed=seed,
        uct_c=arm.uct_c,
        max_simulations=arm.max_simulations,
        n_rollouts=arm.n_rollouts,
        solve=arm.solve,
        dont_return_chance_node=arm.dont_return_chance_node,
        child_selection_fn=openspiel_mcts.SearchNode.puct_value
        if arm.use_puct
        else None,
        record_turns=RECORD_TURNS,
        decision_limit=decision_limit,
        trace=trace,
    )
    seconds = time.perf_counter() - started
    race = state.failed_race()
    row: dict[str, Any] = {
        "name": arm.name,
        "seed": seed,
        "uct_c": arm.uct_c,
        "max_simulations": arm.max_simulations,
        "n_rollouts": arm.n_rollouts,
        "solve": arm.solve,
        "dont_return_chance_node": arm.dont_return_chance_node,
        "use_puct": arm.use_puct,
        "score": state.returns()[0],
        "turn": state.turn,
        "ended": "finished" if race is None else race.name,
        "terminal": state.is_terminal(),
        "speed": state.stats[0],
        "stamina": state.stats[1],
        "training_fails": len(state.training_failures),
        "decisions": trace.decisions,
        "seconds": seconds,
    }
    for turn in RECORD_TURNS:
        snap = trace.snapshots.get(turn)
        row[f"t{turn}_speed"] = None if snap is None else snap[0]
        row[f"t{turn}_stamina"] = None if snap is None else snap[1]
    return row


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def run_arm(
    arm: Arm,
    seeds: range,
    out: Path,
    *,
    jobs: int = 4,
    decision_limit: int | None = None,
) -> list[dict[str, Any]]:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {
        int(row["seed"])
        for row in _load_rows(out)
        if row["name"] == arm.name
        and row.get("decision_limit", decision_limit) == decision_limit
    }
    pending = [seed for seed in seeds if seed not in done]
    print(
        f"{arm.name}: {len(pending)} games, {jobs} workers, "
        f"sims={arm.max_simulations} rollouts={arm.n_rollouts} "
        f"uct={arm.uct_c}",
        flush=True,
    )
    fresh: list[dict[str, Any]] = []
    if pending:
        ctx = get_context("fork")
        with ctx.Pool(jobs) as pool:
            for index, row in enumerate(
                pool.imap_unordered(
                    _play_job, [(seed, arm, decision_limit) for seed in pending]
                ),
                start=1,
            ):
                row["decision_limit"] = decision_limit
                fresh.append(row)
                with out.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                if index == len(pending) or index % 8 == 0:
                    print(
                        f"  {arm.name} {index}/{len(pending)} "
                        f"last={row['seconds']:.1f}s ended={row['ended']}",
                        flush=True,
                    )
    kept = [
        row
        for row in _load_rows(out)
        if row["name"] == arm.name
        and row.get("decision_limit", decision_limit) == decision_limit
    ]
    print(_format_summary(arm.name, kept), flush=True)
    return kept


def _play_job(job: tuple[int, Arm, int | None]) -> dict[str, Any]:
    seed, arm, decision_limit = job
    return play_record(seed, arm, decision_limit)


def _mean_ci(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, 0.0, 0.0
    sd = statistics.stdev(values)
    half = 1.96 * sd / math.sqrt(len(values))
    return mean, sd, half


def _format_summary(name: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"{name}: no games"
    scores = [float(row["score"]) for row in rows]
    turns = [float(row["turn"]) for row in rows]
    secs = [float(row["seconds"]) for row in rows]
    mean, sd, half = _mean_ci(scores)
    turn_mean, _, _ = _mean_ci(turns)
    finished = sum(
        1 for row in rows if row["ended"] == "finished" and row.get("terminal", True)
    )
    ends = Counter(str(row["ended"]) for row in rows)
    end_text = ", ".join(f"{label}={count}" for label, count in ends.most_common())
    stamina = [
        float(row["t43_stamina"]) for row in rows if row.get("t43_stamina") is not None
    ]
    stamina_text = ""
    if stamina:
        stamina_text = f" t43_stamina={statistics.mean(stamina):.0f}(n={len(stamina)})"
    return (
        f"{name}: n={len(rows)} finish={finished}/{len(rows)} "
        f"turns={turn_mean:.1f} score={mean:.0f}±{half:.0f} sd={sd:.0f} "
        f"median={statistics.median(scores):.0f} "
        f"time={statistics.mean(secs):.1f}s max={max(secs):.1f}s "
        f"[{end_text}]{stamina_text}"
    )


def summarize(path: Path, baseline: str | None) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_rows(path):
        grouped[str(row["name"])].append(row)
    for name, rows in grouped.items():
        print(_format_summary(name, rows))
    if baseline is None or baseline not in grouped:
        return
    base = {int(row["seed"]): float(row["score"]) for row in grouped[baseline]}
    base_turn = {int(row["seed"]): float(row["turn"]) for row in grouped[baseline]}
    for name, rows in grouped.items():
        if name == baseline:
            continue
        diffs = [
            float(row["score"]) - base[int(row["seed"])]
            for row in rows
            if int(row["seed"]) in base
        ]
        turn_diffs = [
            float(row["turn"]) - base_turn[int(row["seed"])]
            for row in rows
            if int(row["seed"]) in base_turn
        ]
        if len(diffs) < 2:
            continue
        mean, sd, half = _mean_ci(diffs)
        turn_mean, _, turn_half = _mean_ci(turn_diffs)
        print(
            f"  paired {name} - {baseline}: score {mean:+.0f}±{half:.0f} "
            f"(sd={sd:.0f}, n={len(diffs)}) turns {turn_mean:+.1f}±{turn_half:.1f}"
        )


def _parse_seeds(text: str) -> range:
    start, stop = text.split(":")
    return range(int(start), int(stop))


def _arm_from_args(args: argparse.Namespace) -> Arm:
    return Arm(
        name=args.name,
        uct_c=args.uct_c,
        max_simulations=args.sims,
        n_rollouts=args.rollouts,
        solve=not args.no_solve,
        dont_return_chance_node=args.dont_return_chance_node,
        use_puct=args.puct,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "summarize"))
    parser.add_argument("--name", default="arm")
    parser.add_argument("--uct-c", type=float, default=200.0)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--rollouts", type=int, default=15)
    parser.add_argument("--seeds", default="1000:1001")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--decision-limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("/tmp/mcts_sweep.jsonl"))
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--no-solve", action="store_true")
    parser.add_argument("--dont-return-chance-node", action="store_true")
    parser.add_argument("--puct", action="store_true")
    args = parser.parse_args()
    if args.command == "summarize":
        summarize(args.out, args.baseline)
        return
    run_arm(
        _arm_from_args(args),
        _parse_seeds(args.seeds),
        args.out,
        jobs=args.jobs,
        decision_limit=args.decision_limit,
    )


if __name__ == "__main__":
    main()
