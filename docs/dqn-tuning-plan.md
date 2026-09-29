# DQN tuning plan

Status: **plan only, not executed.** Nothing in `src/` has changed yet. This
document records the current DQN setup in `src/umamusime/bots/dqn.py`, the
problems found by reading it against OpenSpiel's `pytorch/dqn.py`, throwaway
probe measurements, and the ordered steps to raise the greedy score while
keeping training under 5 minutes. When the work is done, replace this file
with a `dqn-tuning.md` in the style of `mcts-tuning.md`.

## Current setup

| Setting | Value |
| --- | --- |
| Network | MLP `[64, 64]`, ReLU |
| Optimizer / loss | Adam, `lr=0.01`, MSE |
| Discount | 1.0 |
| Replay buffer / batch | 10 000 / 128 |
| Update cadence | `learn_every=10`, target copy every 1000 steps (`tau=0.995`) |
| Exploration | `exp` schedule, `epsilon_end=0.1`, duration `1500 * 78 // 2` |
| Episodes | 1500 |
| Observation | 55 floats: turn, six stats, energy, five facility levels, per-card placement one-hot + friendship |

Reward is the per-turn `stat_score` delta. A race soft-fail ends the career
with no explicit penalty; the only race signal the agent gets is the lost
future reward.

## Problems found

1. **Epsilon never decays.** OpenSpiel's exponential schedule clamps `t` to
   `duration`, so the floor is `0.1 + 0.9 / e ≈ 0.43`. Measured: epsilon is
   0.431 at the halfway point and stays there for the rest of training. The
   agent plays 43% random moves forever, and the replay buffer fills with
   low-energy failures and early soft-fails.
2. **The observation hides what decides the score.** The race schedule
   (Kikuka Sho 556 stamina, Tenno Sho 815 stamina) is only recoverable from
   `turn / 78`, and per-action outcomes (which cards attend which facility,
   rainbow status, this turn's exact gains, fail probability, energy delta)
   must be re-derived from 42 one-hot/friendship inputs. `UmaState` already
   computes all of these.
3. **Training is ~15 s, not minutes.** Measured 8 ms/episode with exploration
   (careers die around turn 29) and ~15 ms/episode once the policy finishes
   careers, on one CPU thread. The 600–10 000 episode plateau noted in
   `dqn.py` is explained by the epsilon floor and missing features, not by
   the time budget.

## Probe results

Greedy mean over 30 fresh environment seeds (`1000..1029`), training seed 42
unless noted, `torch.set_num_threads(1)`. The shipped checkpoint scored mean
1706 (0/30 finished) on the same seeds and 1760 / 1900 / 1812 on the
`compare.py` seeds 42–44.

| Variant | Train | Mean | SD | Finished |
| --- | --- | --- | --- | --- |
| Shipped config (1500 ep, 55-float obs) | 14.8 s | 2153 | 595 | 0/30 |
| Shipped config, training seed 7 | 14.8 s | 2987 | 1359 | 0/30 |
| + linear epsilon to 0.05 only | 14.9 s | 2624 | 1222 | 0/30 |
| + rich observation only (115 floats) | 20.6 s | 4362 | 1859 | 1/30 |
| rich obs + linear eps, 6000 ep | 91.5 s | 6060 | 700 | 23/30 |
| same, training seed 7 | 88.5 s | 4932 | 1052 | 6/30 |
| same + buffer 50 000 | 98.5 s | 6476 | 594 | 27/30 |
| same + `learn_every=4`, target every 500 | 131.7 s | 6277 | 948 | 28/30 |
| same + `[128, 128]` | 99.4 s | 5437 | 683 | 25/30 |
| same, 12 000 ep | 188.5 s | 6127 | 585 | 28/30 |
| Huber + `lr=1e-3` (1500 ep) | 13.2 s | 1069 | 69 | 0/30 |
| `gamma=0.99` (1500 ep) | 14.5 s | 2856 | 1259 | 0/30 |
| rich + linear eps + `lr=3e-3`, 6000 ep | 87.5 s | 2180 | 871 | 0/30 |
| rich + linear eps + `gamma=0.99`, 6000 ep | 88.6 s | 2639 | 1558 | 0/30 |

Takeaways:

- The feature vector is the biggest single lever (2153 → 4362 alone).
- The epsilon fix plus more episodes turns that into finished careers
  (6060, 23/30). A larger buffer or more frequent updates adds a few hundred
  more.
- Lowering the learning rate, Huber loss, and discounting all hurt at this
  budget; keep `lr=0.01`, MSE, `gamma=1.0`.
- Training-seed variance is large (6060 vs 4932 for the same config), which
  drives the evaluation protocol below.

## Steps

### Step 0: Measurement harness (first)

- Add an `evaluate(agent, env, seeds)` helper in `dqn.py` that plays N greedy
  careers on fixed seeds (`1000..1000+N`) and reports mean, SD, median, and
  finish rate; make the periodic training log use it instead of a single
  episode.
- Compare every candidate on 30 eval seeds × at least 3 training seeds
  (`seed=` on `dqn.DQN` plus `env.seed`) and rank by mean of means. A 3-seed
  `compare.py` run is too noisy to tune on (the same checkpoint scored
  5643 / 2126 / 6489 on seeds 42–44).
- Record wall time with `torch.set_num_threads(1)`; the MLP is too small for
  intra-op threading to help (4 threads and 1 thread both measured
  8 ms/episode).

### Step 1: Feature vector (highest value, `observer.py`)

Keep the 55 existing features and append, scaled to roughly [−1, 1]:

- Next race: turns until it `/ MAX_TURNS`, `(speed − min_speed) / MAX_STAT`,
  `(stamina − min_stamina) / MAX_STAT`, plus
  `(stamina − max remaining min_stamina) / MAX_STAT` so the 815 Tenno Sho
  gate is visible from Year 1. Optionally the same margins for the
  second-next race.
- Per training action (5 × 9): this turn's `_training_gains(action)` for the
  six stats `/ 60`, `_failure_probability(action)`,
  `_energy_delta(action) / 50`, attending-card count `/ 6`. This makes the
  immediate reward nearly a linear function of the input.
- Per card: rainbow bit (`friendship >= 80`). Per facility: count of
  attending rainbow cards whose main stat matches `/ 6`.
- Summer-camp and finale flags are cheap but optional; facility levels
  already read 5 during camp.

This is the 115-float `rich` observation used in the probes. Expose the
per-action helpers through a small public method on `UmaState` rather than
reaching into private methods from the observer. Update `docs/rules.md` if
it documents the observation. MCTS is unaffected (it does not read
observations). Regenerate `dqn_checkpoint.pt` since the input width changes.

### Step 2: Exploration schedule (one-line fix, with Step 1)

`epsilon_decay_schedule_str=dqn.EpsilonDecaySchedule.LINEAR`,
`epsilon_end=0.05`, `epsilon_decay_duration = int(0.6 * episodes * MAX_TURNS)`.
Alone it is within noise; combined with Step 1 it is what produces finished
careers.

### Step 3: Episode count against the 5-minute cap

Measured cost with the rich observation and `[64, 64]`: ~15 ms/episode at
6000 episodes (91 s) and ~16 ms at 12 000 (189 s), one thread.

- Ship 6000–8000 episodes (90–125 s) as the default; 12 000 did not beat
  6000 on the same seed and roughly doubles time.
- Hard limit: keep any candidate under ~250 s single-threaded so
  `compare.py` (which retrains) stays well under 5 minutes on slower
  machines. Put the measured seconds in the `_TRAINING_EPISODES` comment.
- Treat episodes as the outer knob and derive `epsilon_decay_duration` (and
  `update_target_network_every`, if changed) from it.

### Step 4: Update cadence and replay buffer (spend leftover budget here)

- `replay_buffer_capacity=50_000`: 6000 episodes is 200 k–400 k transitions,
  so 10 000 forgets the early game. Measured +416 mean, 27/30 finished, +7 s.
- `learn_every=4` with `update_target_network_every=500`: +217 mean, 28/30
  finished, +40 s. Try `learn_every=5` and `min_buffer_size_to_learn=5000` as
  the cheaper middle.
- Sweep these two jointly with 3 training seeds each; they are the only
  settings where the 5-minute budget is actually binding.

### Step 5: Network size (low priority)

`[128, 128]` was slightly worse than `[64, 64]` on the same seed and 8%
slower; the input is 115 floats and the target is near-linear once Step 1
is in. Sweep `[64, 64]`, `[128, 64]`, `[128, 128]`, `[256]` only after Steps
1–4 are fixed, on 3 seeds. Do not go deeper than two hidden layers at this
step budget.

### Step 6: Leave alone unless Steps 1–5 stall

- `lr=0.01`, MSE, `gamma=1.0`: each alternative measured worse (Huber /
  `lr=1e-3` never left ~1070; `lr=3e-3` and `gamma=0.99` collapsed to
  ~2200–2600 with the rich observation). If instability shows up with the
  larger buffer, try `gradient_clipping=10` before touching the learning
  rate.
- Reward shaping (a soft-fail penalty) changes the game's scoring and
  belongs in `state.py` / `docs/rules.md`, not in DQN tuning.
- After the sweep, fix the training seed with the best 30-seed mean and
  document that the 6000-episode config ranges roughly 4900–6500 across
  training seeds so `compare.py` readers know the spread.

## Deliverables and checks

- Code: `observer.py` (features), `state.py` (public per-action helpers),
  `dqn.py` (schedule, episodes, buffer, cadence, evaluation helper, updated
  comments), `docs/rules.md` (observation section), and `docs/dqn-tuning.md`
  replacing this plan with the final tables.
- Verification: `uv run ruff check src`, `uv run ty check`, the smoke
  command from `AGENTS.md`, one `uv run python -m umamusime.bots.dqn` run to
  confirm training time, and `uv run python -m umamusime.compare` once at the
  end.
