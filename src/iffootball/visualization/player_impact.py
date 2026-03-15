"""Player impact ranking from Branch A/B comparison.

Identifies the players most affected by the trigger by computing the
mean absolute difference in dynamic state between Branch A and Branch B
across N simulation runs.

Impact score per player per run:
    |form_a - form_b| + |fatigue_a - fatigue_b|
    + |understanding_a - understanding_b| + |trust_a - trust_b|

The per-run scores are averaged to produce a stable ranking. This avoids
cancellation that would occur if averaging states first, then diffing.

Ranking:
    Players are ranked by descending impact score, with player_id as
    ascending tiebreaker for deterministic ordering.
"""

from __future__ import annotations

from dataclasses import dataclass

from iffootball.agents.player import PlayerAgent
from iffootball.simulation.comparison import ComparisonResult
from iffootball.simulation.engine import SimulationResult


@dataclass(frozen=True)
class PlayerImpact:
    """Impact summary for a single player.

    Attributes:
        player_id:   Unique player identifier (used for matching across runs).
        player_name: Display name.
        impact_score: Mean absolute dynamic-state difference across runs.
        mean_form_a:  Mean final form in Branch A.
        mean_form_b:  Mean final form in Branch B.
        mean_fatigue_a:  Mean final fatigue in Branch A.
        mean_fatigue_b:  Mean final fatigue in Branch B.
        mean_understanding_a: Mean final tactical understanding in Branch A.
        mean_understanding_b: Mean final tactical understanding in Branch B.
        mean_trust_a: Mean final manager trust in Branch A.
        mean_trust_b: Mean final manager trust in Branch B.
    """

    player_id: int
    player_name: str
    impact_score: float
    mean_form_a: float
    mean_form_b: float
    mean_fatigue_a: float
    mean_fatigue_b: float
    mean_understanding_a: float
    mean_understanding_b: float
    mean_trust_a: float
    mean_trust_b: float


def _player_impact_per_run(
    squad_a: list[PlayerAgent],
    squad_b: list[PlayerAgent],
) -> dict[int, float]:
    """Compute per-player impact score for a single run pair.

    Returns a mapping of player_id -> absolute dynamic state difference.
    Only players present in both squads are included.
    """
    index_b = {p.player_id: p for p in squad_b}
    scores: dict[int, float] = {}

    for pa in squad_a:
        pb = index_b.get(pa.player_id)
        if pb is None:
            continue

        score = (
            abs(pa.current_form - pb.current_form)
            + abs(pa.fatigue - pb.fatigue)
            + abs(pa.tactical_understanding - pb.tactical_understanding)
            + abs(pa.manager_trust - pb.manager_trust)
        )
        scores[pa.player_id] = score

    return scores


def _mean_state(
    runs: tuple[SimulationResult, ...],
    player_id: int,
    attr: str,
) -> float:
    """Compute mean of a player attribute across runs."""
    total = 0.0
    count = 0
    for run_result in runs:
        for p in run_result.final_squad:
            if p.player_id == player_id:
                total += getattr(p, attr)
                count += 1
                break
    return total / count if count > 0 else 0.0


def rank_player_impact(
    comparison: ComparisonResult,
    *,
    top_n: int = 5,
) -> list[PlayerImpact]:
    """Rank players by impact score from A/B comparison.

    Computes per-run absolute dynamic state differences, averages them,
    and returns the top N most affected players.

    Args:
        comparison: A/B comparison result with run_results retained.
        top_n:      Number of top players to return.

    Returns:
        List of PlayerImpact sorted by descending impact_score,
        then ascending player_id as tiebreaker.
    """
    runs_a = comparison.no_change.run_results
    runs_b = comparison.with_change.run_results
    n_runs = min(len(runs_a), len(runs_b))

    if n_runs == 0:
        return []

    # Accumulate per-player impact scores across runs.
    accumulated: dict[int, float] = {}
    for i in range(n_runs):
        per_run = _player_impact_per_run(
            runs_a[i].final_squad, runs_b[i].final_squad
        )
        for pid, score in per_run.items():
            accumulated[pid] = accumulated.get(pid, 0.0) + score

    # Average and collect player names.
    name_map: dict[int, str] = {
        p.player_id: p.player_name for p in runs_a[0].final_squad
    }

    scored: list[tuple[int, float]] = [
        (pid, total / n_runs) for pid, total in accumulated.items()
    ]
    # Sort: descending impact_score, ascending player_id as tiebreaker.
    scored.sort(key=lambda x: (-x[1], x[0]))

    results: list[PlayerImpact] = []
    for pid, impact in scored[:top_n]:
        results.append(
            PlayerImpact(
                player_id=pid,
                player_name=name_map.get(pid, f"Player {pid}"),
                impact_score=impact,
                mean_form_a=_mean_state(runs_a, pid, "current_form"),
                mean_form_b=_mean_state(runs_b, pid, "current_form"),
                mean_fatigue_a=_mean_state(runs_a, pid, "fatigue"),
                mean_fatigue_b=_mean_state(runs_b, pid, "fatigue"),
                mean_understanding_a=_mean_state(
                    runs_a, pid, "tactical_understanding"
                ),
                mean_understanding_b=_mean_state(
                    runs_b, pid, "tactical_understanding"
                ),
                mean_trust_a=_mean_state(runs_a, pid, "manager_trust"),
                mean_trust_b=_mean_state(runs_b, pid, "manager_trust"),
            )
        )

    return results
