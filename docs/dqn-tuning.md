# DQN tuning

OpenSpiel's PyTorch `DQN` with a small MLP. The algorithm was not changed;
the observation, exploration schedule, and constructor parameters were tuned
to raise the greedy score inside a 5-minute training budget. Defaults live in
`DQNConfig` in `src/umamusime/bots/dqn.py`; features live in
`src/umamusime/observer.py`.

## Shipped settings

| Parameter | Value | Before |
| --- | --- | --- |
| Episodes | 8000 | 1500 |
| Network | MLP `[64, 64]`, ReLU | same |
| Optimizer / loss | Adam, `lr=0.01`, MSE | same |
| Discount | 1.0 | same |
| Exploration | linear 1.0 → 0.05 over the first 60% of steps | `exp`, floor ~0.43 |
| Replay buffer | 50 000 | 10 000 |
| Batch / `learn_every` | 128 / 10 | same |
| Target sync | every 250 steps (`tau=0.995`) | every 1000 |
| Observation | 115 floats | 55 floats |
| Training seed | 42 | unseeded environment |

Training takes about 130–140 s on one CPU thread (137.9 s inside
`umamusime.compare`), well under the 5-minute cap. The MLP is too small for
intra-op threading to help.

## What was wrong

1. **Epsilon never decayed.** OpenSpiel's `exp` schedule clamps the step
   count to the decay duration, so epsilon bottoms out at
   `end + (1 - end) / e ≈ 0.43`. The agent played 43% random moves for the
   whole run and its replay buffer filled with low-energy failures and early
   race soft-fails. The comment that 1500 episodes was where a 600–10 000
   sweep plateaued described this floor, not a capacity limit.
2. **The observation hid what decides the score.** Race gates (Kikuka Sho
   556 stamina, Tenno Sho 815 stamina) could only be inferred from
   `turn / 78`, and each training's outcome had to be rebuilt from one-hot
   placements and friendship.

## Observation

The first 55 floats are unchanged: turn, six stats, energy, five facility
levels, and per card a placement one-hot plus friendship. Appended, all
roughly in [−1, 1]:

- 4 race features: turns to the next race, speed and stamina margin against
  it, and stamina margin against the hardest remaining race (so the 815 gate
  is visible from year 1). All zero once no race remains.
- 5 × 9 training features: this turn's six stat gains (`/ 60`), fail
  probability, energy delta (`/ 50`), and share of cards attending.
- 6 rainbow bits, one per card, and per training the share of cards that are
  rainbowed on their own specialty.

These come from `UmaState.training_gains`, `failure_probability`,
`energy_delta`, `attending`, and `is_rainbow`, which were made public for
the observer. MCTS does not read observations.

## Method

Every candidate was trained from scratch with `torch.set_num_threads(1)` and
scored greedily on 30 fixed career seeds (`1000..1029`), for three training
seeds. The metric is the mean of the three per-seed means. Per-seed means of
the same config spread by roughly ±300, so differences under about 300 are
noise; the score of a single `compare.py` seed (for example 5643 / 2126 /
6489 for one checkpoint) is far too noisy to tune on. Times below were
measured with four sweeps sharing four cores and read 10–30% high.

## Early probes (single training seed)

| Variant | Mean | Finished |
| --- | --- | --- |
| Old config (1500 ep, 55 floats), seed 42 / 7 | 2153 / 2987 | 0/30 |
| + linear epsilon only | 2624 | 0/30 |
| + 115-float observation only | 4362 | 1/30 |
| + both, 6000 ep | 6060 | 23/30 |
| Huber loss + `lr=1e-3` | 1069 | 0/30 |
| `gamma=0.99` | 2856 | 0/30 |
| Both fixes + `lr=3e-3` | 2180 | 0/30 |
| Both fixes + `gamma=0.99` | 2639 | 0/30 |

The observation and the exploration schedule each help a little alone and
together turn most careers into finished ones. Lower learning rate, Huber
loss, and discounting all hurt, so `lr=0.01`, MSE, and `gamma=1.0` stayed.

## Sweeps (three training seeds each, mean of means)

Round 1, training seeds 42 / 7 / 123, target sync every 1000 unless noted:

| Setup | Mean | Finished (of 90) |
| --- | --- | --- |
| 6000 ep | 6017 | 59 |
| 8000 ep | 6359 | 84 |
| 6000 ep, `learn_every=5` | 6315 | 81 |
| 6000 ep, `learn_every=4`, target sync 500 | 6017 | 48 |

Round 2, 8000 episodes, network size and cadence:

| Setup | Mean | Finished (of 90) |
| --- | --- | --- |
| `[64, 64]` (reference, round 1) | 6359 | 84 |
| `[128, 64]` | 6039 | 67 |
| `[128, 128]` | 5618 | 63 |
| `[256]` | 6151 | 6 |
| `learn_every=5` | 6167 | 65 |

Round 3, 8000 episodes:

| Setup | Mean | Finished (of 90) |
| --- | --- | --- |
| `[32, 32]` | 6483 | 32 |
| epsilon decay fraction 0.8 | 6409 | 66 |
| epsilon floor 0.02 | 6458 | 68 |
| target sync every 250 | 6566 | 73 |

Round 4, fresh training seeds 1 / 2 / 3 to avoid selecting on the same noise:

| Setup | Mean | Finished (of 90) |
| --- | --- | --- |
| 8000 ep reference | 6337 | 53 |
| + target sync 250 | 6416 | 67 |
| + target sync 250, floor 0.02 | 6509 | 76 |
| + target sync 250, `[32, 32]` | 6413 | 65 |

Pooled over all six training seeds, 8000 episodes with target sync 250
scores 6491 against 6348 for the same run with sync 1000, and adding the
0.02 floor gives 6483. The floor was kept at 0.05 for simplicity.

Round 5, extra features on top of the shipped config (second-next race
margins, summer-camp flag, finale flag; 120 floats): 6344 over the six
seeds with 73/180 finished, against 6491 and 140/180 without. Not adopted.

## Findings

- **Episodes.** 8000 beat 6000 by about 340. Beyond that the cost grows
  linearly (about 16 ms/episode once careers finish) and a 12 000-episode
  probe on one seed matched 6000.
- **Network size.** Larger networks did not help; the target is close to
  linear in the derived features. `[64, 64]` stays; `[32, 32]` was within
  noise but finished fewer careers.
- **Replay buffer.** 50 000 holds most of a run's transitions; 10 000
  forgot the early game (single-seed probe: +416 mean and 27/30 finished at
  6000 episodes).
- **Update cadence.** More frequent learning (`learn_every` 4 or 5) cost
  20–40% more time for no reliable gain. A faster target sync (250) was the
  one cadence change that held up on fresh seeds.
- **Training-seed variance is large.** The same config's per-seed means
  ranged about 5900–6750 and finish counts from 0/30 to 30/30 at similar
  scores, so a single `compare.py` DQN run should be read as a ±300 spread.

## Result

`uv run python -m umamusime.compare` after tuning (seeds 42–44):

| Method | Runs | Best | Time |
| --- | --- | --- | --- |
| Random | 954, 637, 1027 | 1027 | instant |
| MCTS | 5960, 4442, 2903 | 5960 | 47 s per game |
| DQN | 6004, 6299, 6364 | 6364, finished | 138 s train |

Before tuning the shipped checkpoint scored 1760 / 1900 / 1812 in about
15 s of training, and 1706 on average (0/30 finished) over 30 fresh seeds.
