# Simulation Rules

This document defines all configurable parameters used by the IfFootball simulation engine. Each parameter includes its value, definition, rationale, and source.

Configuration files are located in `config/simulation_rules/`.

---

## Adaptation Parameters

**File:** `adaptation.toml`

These parameters control weekly state updates for players during the simulation.

| Parameter                     | Value | Scale   | Definition                                                                  |
| ----------------------------- | ----- | ------- | --------------------------------------------------------------------------- |
| `base_fatigue_increase`       | 0.05  | 0.0–1.0 | Fatigue added to each player who played in a match                          |
| `base_fatigue_recovery`       | 0.03  | 0.0–1.0 | Fatigue recovered by each player who did not play                           |
| `tactical_understanding_gain` | 0.04  | 0.0–1.0 | Base weekly gain in tactical understanding (before adaptation_rate scaling) |
| `fatigue_penalty_weight`      | 0.5   | 0.0–1.0 | How much fatigue reduces the agent state factor in match result calculation |
| `trust_increase_on_start`     | 0.02  | 0.0–1.0 | Manager trust increase when selected as starter                             |
| `trust_decrease_on_bench`     | 0.01  | 0.0–1.0 | Manager trust decrease when benched                                         |

### Rationale

- **base_fatigue_increase / recovery:** A starter accumulates ~0.05 fatigue per match and recovers ~0.03 when rested. Over a 3-match cycle (play-play-rest), fatigue peaks around 0.07. This keeps fatigue meaningful without dominating lineup decisions. [Developer judgement, no empirical calibration yet]
- **tactical_understanding_gain:** At 0.04 per week with neutral adaptation rate, a player reaches ~0.75 understanding after ~6 matches post-appointment. This reflects a realistic mid-season adaptation window. [Developer judgement]
- **fatigue_penalty_weight:** At 0.5, a fully fatigued player (1.0) reduces the agent state factor by 50%. This makes squad rotation important without making fatigue catastrophic. [Developer judgement]
- **trust_increase/decrease:** Asymmetric (0.02 vs 0.01) to reflect that trust builds faster through selection than it erodes through benching. A starter for 10 consecutive matches gains ~0.2 trust. [Developer judgement]

---

## Match Parameters

**File:** `match.toml`

| Parameter               | Value | Definition                                         |
| ----------------------- | ----- | -------------------------------------------------- |
| `home_advantage_factor` | 1.1   | Multiplier for expected goals when playing at home |

### Rationale

- **home_advantage_factor:** Home teams in the Premier League historically score ~10-15% more goals than away. The 1.1 multiplier applies to expected_goals_for when is_home=True, or to expected_goals_against when is_home=False. [Based on aggregate Premier League home/away goal ratios; see Pollard & Pollard (2005) for reference ranges]

---

## Turning Point Thresholds

**File:** `turning_points.toml`

### Player Thresholds

| Parameter                    | Value | Scale   | Definition                                                                    |
| ---------------------------- | ----- | ------- | ----------------------------------------------------------------------------- |
| `bench_streak_threshold`     | 3     | integer | Consecutive benched matches before a turning point fires                      |
| `tactical_understanding_low` | 0.40  | 0.0–1.0 | Understanding below this triggers confusion TP (within short_term_window)     |
| `short_term_window`          | 4     | integer | Matches after appointment during which low_understanding TP can fire          |
| `trust_low`                  | 0.40  | 0.0–1.0 | Trust below this combined with bench_streak TP triggers resist-heavy response |

### Manager Thresholds

| Parameter                      | Value | Scale   | Definition                                                |
| ------------------------------ | ----- | ------- | --------------------------------------------------------- |
| `job_security_warning`         | 0.30  | 0.0–1.0 | Job security below this causes a defensive tactical shift |
| `job_security_critical`        | 0.10  | 0.0–1.0 | Job security below this generates a dismissal event       |
| `style_stubbornness_threshold` | 80    | 0–100   | Stubbornness at or above this prevents tactical shifts    |

### Rationale

- **bench_streak_threshold (3):** Three consecutive benchings is a clear signal of exclusion. In practice, most squad players tolerate 1-2 missed matches before frustration becomes visible. [Domain knowledge, common threshold in FM-style models]
- **tactical_understanding_low (0.40):** Players start at 0.25 after a manager change and gain ~0.04/week. At 0.40, confusion fires for roughly the first 4 weeks, matching the short_term_window. [Aligned with adaptation curve]
- **short_term_window (4):** Four matches represents approximately one month of fixtures, a commonly cited adaptation period for new managerial appointments. [Domain knowledge]
- **trust_low (0.40):** Combined with bench_streak, this threshold identifies players who are both excluded and losing confidence. Starting at 0.5 after a manager change, a player benched every week drops to ~0.4 after 10 weeks. [Calibrated against trust update rates]
- **job_security_warning (0.30):** Corresponds to ~1.0 points per match over the rolling 5-match window (5 points from 15). This is relegation-level form. [Based on points-per-game thresholds]
- **job_security_critical (0.10):** Corresponds to ~0.3 points per match (near-zero wins). Extreme enough that dismissal is the expected outcome. [Developer judgement]
- **style_stubbornness_threshold (80):** Only very stubborn managers (top 20th percentile) refuse to adapt tactically under pressure. [LLM-derived hypothesis attribute; threshold is provisional]

---

## Action Distributions

**File:** `turning_points.toml` — `[action_distribution]` section

These define the probability weights for player actions (adapt / resist / transfer) returned by the RuleBasedHandler. Distributions are normalised to sum to 1.0 at runtime.

### Priority Order

1. **bench_streak_low_trust** — bench_streak TP + trust below trust_low
2. **low_understanding** — low_understanding TP (early confusion)
3. **default** — no TP or unmatched condition

| Condition              | adapt | resist | transfer |
| ---------------------- | ----- | ------ | -------- |
| bench_streak_low_trust | 0.3   | 0.6    | 0.1      |
| low_understanding      | 0.5   | 0.4    | 0.1      |
| default                | 0.8   | 0.2    | 0.0      |

### Rationale

- **bench_streak_low_trust:** A player who is repeatedly benched AND has low trust is most likely to resist. 60% resist reflects strong disaffection. 10% transfer signal represents the minority who would push for a move. [Developer judgement]
- **low_understanding:** Early confusion after a manager change favours adaptation (50%) since most players try to learn the new system. However, 40% resist reflects that confusion often causes performance drops regardless of intent. [Developer judgement]
- **default:** Stable conditions strongly favour adaptation (80%). A small resist component (20%) accounts for background friction. Transfer probability is zero in stable conditions. [Developer judgement]

---

## Cascade Event Magnitudes

These are hardcoded in `engine.py` (not yet config-driven). Included here for completeness.

| Event                                          | Magnitude | Context                                  |
| ---------------------------------------------- | --------- | ---------------------------------------- |
| form_drop (resist)                             | 0.3       | Direct effect of resist action           |
| trust_decline (chained from form_drop)         | 0.2       | Secondary effect: form drop erodes trust |
| playing_time_change (transfer)                 | 0.4       | Player signals desire to leave           |
| adaptation_progress (adapt)                    | 0.15      | Positive adaptation recorded             |
| tactical_confusion (low_understanding + adapt) | 0.2       | Confusion alongside adaptation           |
| squad_unrest (2+ resist in same week)          | 0.5       | Squad-level discontent signal            |
| manager_tactical_shift (job_security_warning)  | 0.5       | Manager shifts to defensive tactics      |
| manager_dismissal (job_security_critical)      | 0.8       | Manager dismissed                        |

### Rationale

- Magnitudes are on a 0.0–1.0 scale where higher = more significant. Values are provisional and not yet calibrated against backtest outcomes. The importance_threshold (0.05) filters out trivial events. [Developer judgement throughout]

---

## Known Limitations

- All parameter values are provisional (M3). Formal calibration against historical outcomes is planned for M4 backtest evaluation.
- Cascade event magnitudes are hardcoded, not config-driven. Config externalisation is a future improvement.
- `pace` and `physicality` player attributes are fixed at 50.0 (neutral) because StatsBomb Open Data does not include sprint speed or physical contact metrics.
- `consistency` is simulation-stored but not yet connected to lineup selection or match result calculation.
- The match model uses team-level xG baselines; individual player contribution to xG is not simulated per-match.
