# DQN tuning

OpenSpiel's PyTorch `DQN` with a small MLP. The algorithm was not
changed. The step budget, exploration schedule, and a snapshot of the
greedy policy were tuned to raise the greedy score inside a 5-minute
training budget. Defaults live in `DQNConfig` in
`src/umamusime/bots/dqn.py`. Features live in
`src/umamusime/observer.py`.

`±` below is a standard error: the sample standard deviation of the
per-training-seed differences, divided by sqrt(n). A candidate clears
the screening rule when its mean gain over the reference is greater
than twice that standard error. Adoption also asked the sign to hold
on a fresh confirmation set. D1 and D2 were adopted with a positive
confirmation mean that did not itself clear 2 SE. From D3 on, a miss
on that rule was a rejection.

## Shipped settings

| Parameter | Value | Before (`541cb44`) |
| --- | --- | --- |
| Training budget | 400 000 agent steps | 8000 episodes |
| Replay buffer | 100 000 | 50 000 |
| Exploration | linear 1.0 → 0.05 over the first 30% of steps (120 000) | linear 1.0 → 0.05 over 60% of an assumed 8000 × 78 steps (374 400), which the run never completed |
| Snapshot selection | 50 held-out careers every 20 000 steps after epsilon reaches its floor; restore the best weights | none |
| Network | MLP `[64, 64]`, ReLU | same |
| Optimizer / loss | Adam, `lr=0.01`, MSE | same |
| Discount | 1.0 | same |
| Batch / `learn_every` | 128 / 10 | same |
| Target sync | every 250 steps (soft update 0.995) | same |
| Min buffer to learn | 1000 | same |
| Observation | 115 floats | same |
| Default training seed | 42 | same |

Discount, MSE, the soft-update coefficient, and ReLU are OpenSpiel
defaults. This code passes Adam, the linear schedule, and the sizes
above. One agent step is one `DQN.step` call, including the terminal
step of each episode.

Training the shipped config takes about 236–252 s on a quiet
machine: 251.7 s for `dqn.train()` with default Torch threading,
and 236–240 s with one thread.
A solo of seed 3741676989 scored mean 6863.0, restored step 360 034,
and finished 131/200. That run was 235.7 s before the cleanup of the
snapshot code and 239.9 s on the cleaned code, with the same score
and the same restored step. The previous 8000-episode config took
about 142 s (mean 141.9 s on the six D6 seeds, range 138.1–144.3 s).

## Observation

115 floats, unchanged by this retune. The first 55 are the career
state: `turn / 78`, five stats over `MAX_STAT` (1200), skill points
over `skill_point_obs_scale`, energy over 100, five facility levels
over 5, then per card a placement one-hot (six outcomes) and
friendship over 100. Appended:

- 4 race features: turns until the next race, speed and stamina
  margin against it, and stamina margin against the hardest remaining
  race. All zero once no race remains. Tenno Sho (Spring) is that
  hardest stamina gate from the start (815).
- 5 × 9 training features: this turn's six stat gains (`/ 60`), fail
  probability, energy delta (`/ 50`), and the share of cards
  attending.
- 6 rainbow bits, one per card, and per training the share of
  attending cards that are rainbowed on their own specialty.

These come from `UmaState.training_gains`, `failure_probability`,
`energy_delta`, `attending`, and `is_rainbow`. MCTS does not read
observations.

A speed margin against the hardest remaining speed gate was not
added. URA Finale Finals is the last race and the unique maximum
speed gate (580), so that margin is `speed / 1200` minus the
constant `580 / 1200` on every observation the agent sees. The
speed feature is already in the first 55.

## What was wrong

The previous decay length was
`int(0.6 × 8000 × MAX_TURNS)` with `MAX_TURNS` 78, which is
374 400 steps. A career does not take 78 agent steps. Race turns
are not decisions, and a soft fail ends the career. The 8000-episode
runs stopped at 272 445–286 371 steps, 34.1–35.8 steps per episode.
OpenSpiel's linear schedule is `max(1 - 0.95 × t / 374400, 0.05)`.
At those step counts epsilon was still 0.273–0.309, and
`reached_floor` was false on all twelve old runs. The last thousand
training episodes ended near turn 51–59.

Counting steps directly, and decaying over 30% of them, does reach
0.05. The policy then keeps training for a long time at the floor.
On the six fresh seeds in D6, five runs of that schedule scored
6219–6992. Seed 3392167989 scored 3668.3, finished 31/200, and had
59 early deaths. The same shape, one seed far below the others,
showed up under nearby settings: D1 decay over 50% of steps, floor
0.05, seed 1506158909, mean 4622.4 (57 early deaths); D2 at
275 000 steps with a 100 000 buffer, seed 1516921060, mean 5273.0;
D2 confirmation at 400 000 steps with a 200 000 buffer, seed
1180240126, mean 5348.8. Of the fourteen runs of the adopted
400 000-step, 100 000-buffer, floor-0.05 schedule with no snapshot
(four screening, four confirmation, six in D6), one collapsed.

Snapshot selection keeps a copy of the weights after the floor,
every 20 000 steps, scored greedily on 50 held-out careers. The
first time the mean is strictly best wins a tie. At the end of the
budget those weights are loaded into both the Q network and the
target network. On the collapsed D6 seed the restored step was
320 010 and the eval mean was 6903.6, with 1 early death.

## Method

Training seeds are 32-bit ints from `secrets.randbits(32)`, drawn
once, logged, and not reused. Each run is scored greedily
(epsilon 0) on one shared block of 200 careers, through `base.play`
and `split_rngs`, so the careers match across configs. An early
death is a race fail on turn 33 or earlier (through Japanese Derby).
Finished means the career did not fail a race.

D1–D5 used four screening seeds (1516921060, 45623533, 2692142897,
1506158909) and, when a candidate was confirmed, four others
(168950876, 1180240126, 2136911399, 1945238534). The eval block was
2081209073–2081209272. D6 and the D7 screen used six new training
seeds (3741676989, 2687964709, 4109534177, 2057313969, 253451421,
3392167989) and eval block 3772577048–3772577247. D7 confirmation
used six more (3803307932, 1208411400, 641555398, 1800148260,
337701470, 2974456480) and eval block 1958708182–1958708381.
Seed 42 and three other seeds were excluded from the draws.

The paired difference is the candidate's mean minus the reference's
mean on the same training seed. Some reference rows were copied
forward rather than retrained: D2's 275 000-step, 50 000-buffer
rows are the adopted D1 runs; D3–D5's reference rows are the D2
400 000-step, 100 000-buffer runs; D7's old and new rows are the
D6 runs. Means match exactly where a row was copied.

Sweeps used at most three processes, with one Torch thread each.
`train_seconds` is the training loop only. Snapshot evaluations
sit inside that loop. Times in the tables are those parallel runs
unless a row says solo. A solo rerun was used when a candidate was
slower than the reference, or to time the shipped snapshot config.
The score of seed 1516921060 at the D2 reference matched on a solo
(mean 6785.2), so the shared-core runs did not move the greedy
means.

The harness and the JSONL logs are not in the repository. They are
under `/opt/cursor/artifacts/tuning/dqn/`.

## Epsilon schedule (D1)

All runs used 275 000 steps, the length an 8000-episode career
actually reached. The reference kept the old decay length of
374 400, so epsilon ended at 0.302 and never hit the floor. The
other six schedules decay linearly over 30%, 50%, or 70% of the
275 000 steps, to a floor of 0.05 or 0.02.

| Schedule | Mean | vs reference | Finished | Early | Train |
| --- | --- | --- | --- | --- | --- |
| Old duration (reference) | 6475.7 |  | 671/800 | 17 | 156.4 s |
| 30%, floor 0.05 | 6715.3 | +239.6 ± 64.5 | 586/800 | 2 | 156.7 s |
| 30%, floor 0.02 | 6564.6 | +89.0 ± 143.8 | 602/800 | 6 | 167.6 s |
| 50%, floor 0.05 | 6164.7 | −310.9 ± 503.6 | 485/800 | 57 | 161.8 s |
| 50%, floor 0.02 | 6635.5 | +159.8 ± 61.2 | 511/800 | 14 | 163.4 s |
| 70%, floor 0.05 | 6697.8 | +222.2 ± 109.5 | 663/800 | 3 | 150.8 s |
| 70%, floor 0.02 | 6599.9 | +124.2 ± 232.5 | 492/800 | 25 | 144.7 s |

Three cleared 2 SE: 30% to 0.05, 50% to 0.02, and 70% to 0.05.
The highest mean was 30% to 0.05, higher on all four seeds. The
50% / 0.05 schedule did not: seed 1506158909 scored 4622.4
(26/200 finished, 57 early deaths). Once a schedule reached the
floor, the last thousand training episodes ended near turn 73–75
instead of turn 55.

Confirmation of 30% to 0.05, four fresh seeds: mean 6577.2 against
the reference 6262.5, difference +314.7 ± 187.6 (2 SE is 375.2),
higher on 3 of 4 seeds. The low seed was 1945238534 at 6269.5.
That does not clear 2 SE. It was adopted. Later stages use this
schedule: epsilon falls over the first 30% of whatever the step
budget is, and the floor is 0.05.

## Step budget and replay buffer (D2)

The reference is the adopted D1 config: 275 000 steps, buffer
50 000, decay over 30% to 0.05. Those rows were not retrained.
Decay stays at 30% of the step budget, so a longer run also
reaches the floor later (120 000 steps at 400 000; 142 500 at
475 000).

| Steps | Buffer | Mean | vs 275k / 50k | Finished | Train |
| --- | --- | --- | --- | --- | --- |
| 275 000 | 50 000 | 6715.3 |  | 586/800 | 156.7 s |
| 275 000 | 100 000 | 6463.9 | −251.4 ± 404.8 | 589/800 | 160.0 s |
| 275 000 | 200 000 | 6690.7 | −24.6 ± 124.6 | 437/800 | 168.5 s |
| 400 000 | 50 000 | 6758.5 | +43.3 ± 72.9 | 568/800 | 231.2 s |
| 400 000 | 100 000 | 6805.2 | +89.9 ± 37.4 | 626/800 | 231.7 s |
| 400 000 | 200 000 | 6869.8 | +154.5 ± 51.2 | 525/800 | 266.9 s |
| 475 000 | 50 000 | 6709.0 | −6.2 ± 158.2 | 536/800 | 260.7 s |
| 475 000 | 100 000 | 6746.3 | +31.0 ± 91.5 | 520/800 | 273.4 s |
| 475 000 | 200 000 | 6774.4 | +59.1 ± 68.3 | 565/800 | 315.9 s |

Two cleared, both on all four seeds: 400 000 / 100 000, and
400 000 / 200 000. The 275 000 / 100 000 mean is pulled down by
seed 1516921060 at 5273.0. 475 000 steps with a 200 000 buffer
averaged 315.9 s with three workers, past the 5-minute budget,
and did not clear.

Confirmation, four fresh seeds, against the D1 confirmation rows
(mean 6577.2):

| Config | Mean | vs reference | Train |
| --- | --- | --- | --- |
| 400 000 / 100 000 | 6702.6 | +125.5 ± 127.1 | 246.5 s |
| 400 000 / 200 000 | 6466.3 | −110.9 ± 398.0 | 266.8 s |

400 000 / 100 000 was higher on 3 of 4 seeds. Twice the SE is
254.2, so it does not clear. The minimum seed mean was 6593.8.
400 000 / 200 000 changed sign: seed 1180240126 scored 5348.8,
and the run was slower. Adopted 400 000 steps and a 100 000
buffer. The larger buffer was not.

## Reward scale, loss, and clipping (D3)

The reference is the adopted D2 run (scale 1, `lr=0.01`, MSE),
rows copied, mean 6805.2. Rewards were multiplied by 0.01 or
0.001 and the learning rate was lowered with them. Huber loss
and gradient clipping at 10 were then tried on the best of those
scaled runs, which was scale 0.01 and `lr=0.001`.

| Setup | Mean | vs reference | Finished | Early |
| --- | --- | --- | --- | --- |
| Scale 1, `lr=0.01`, MSE | 6805.2 |  | 626/800 | 8 |
| Scale 0.01, `lr=0.01` | 5433.2 | −1371.9 ± 436.8 | 235/800 | 37 |
| Scale 0.01, `lr=0.003` | 6346.5 | −458.6 ± 155.0 | 469/800 | 15 |
| Scale 0.01, `lr=0.001` | 6708.5 | −96.6 ± 123.4 | 514/800 | 7 |
| Scale 0.001, `lr=0.01` | 2039.0 | −4766.1 ± 20.9 | 0/800 | 800 |
| Scale 0.001, `lr=0.003` | 6139.8 | −665.3 ± 220.5 | 379/800 | 34 |
| Scale 0.001, `lr=0.001` | 6411.1 | −394.0 ± 62.0 | 406/800 | 9 |
| Scale 0.01, `lr=0.001`, Huber | 2892.3 | −3912.9 ± 735.5 | 44/800 | 432 |
| Scale 0.01, `lr=0.001`, clip 10 | 6612.4 | −192.7 ± 248.6 | 528/800 | 40 |

Nothing cleared. Scale 0.01 at `lr=0.001` was the closest and was
higher on only 2 of 4 seeds. Scale 0.001 at `lr=0.01` died on
Spring Stakes on all 800 careers; three of the four seeds scored
2051.1. Huber on the best scaled setting finished 44/800. Clipping
did not help. Rejected. The training loss and the reward scale
were left as they were.

## Cadence and width (D4)

One change at a time, on the adopted D2 config. The reference rows
are those D2 runs.

| Change | Mean | vs reference | Finished | Train | Solo |
| --- | --- | --- | --- | --- | --- |
| Reference | 6805.2 |  | 626/800 | 231.7 s | 227.5 s |
| `learn_every` 5 | 6622.7 | −182.4 ± 187.6 | 435/800 | 320.2 s | 313.9 s |
| Batch 256 | 6931.1 | +126.0 ± 63.5 | 579/800 | 233.1 s | 233.7 s |
| Hidden `[128, 128]` | 6482.9 | −322.3 ± 204.5 | 518/800 | 243.4 s | 239.9 s |
| Target sync every 1000 | 6608.5 | −196.6 ± 133.5 | 604/800 | 227.0 s |  |

Batch 256 was higher on 3 of 4 seeds. The gain is 126.0 and twice
the SE is 126.9, so it does not clear. The solo of seed
1516921060 matched the screening mean (6935.7) in 233.7 s.
`learn_every` 5 was slower and worse: the solo took 313.9 s, past
300 s, and that seed scored 6096.1 with 29/200 finished. The wider
network was worse. A slower target sync was worse and slightly
faster in parallel, so it was not retimed alone. Nothing adopted.

## Pace features (D5)

Both features were one extra float, so the observation was 116
during the run and 115 after. The reference rows are again the D2
400 000 / 100 000 runs. Pace is the largest upcoming deficit of
that stat, divided by remaining decision turns times 60, and
clipped to 1. Decision turns are turns in `[state.turn, race.turn)`
that are not race turns. A deficit with no decision left uses the
cap. Speed pace was tried first. It did not clear, so stamina pace
was tried alone, not stacked on it.

| Feature | Mean | vs reference | Finished | Early | Train | Solo |
| --- | --- | --- | --- | --- | --- | --- |
| None | 6805.2 |  | 626/800 | 8 | 231.7 s |  |
| Speed pace | 6705.1 | −100.0 ± 174.3 | 438/800 | 19 | 254.4 s | 252.1 s |
| Stamina pace | 6689.8 | −115.4 ± 16.3 | 585/800 | 2 | 255.7 s | 253.6 s |

Speed pace was higher on 3 of 4 seeds. Seed 1506158909 scored
6253.5 (56/200 finished, 16 early deaths), and the SE is large.
Stamina pace was lower on all four seeds, by 115.4 ± 16.3.
Endings over the 800 careers:

| | Finished | Japan Cup | Arima Kinen | Finals | Semifinal |
| --- | --- | --- | --- | --- | --- |
| Reference | 626 | 31 | 25 | 71 | 19 |
| Speed pace | 438 | 135 | 54 | 118 | 21 |
| Stamina pace | 585 | 74 | 38 | 63 | 18 |

Both features moved failures toward Japan Cup. Solos were about
20 s slower than the reference solo and reproduced the screening
means. Rejected. `observer.py` is unchanged.

## Previous DQN, fresh seeds (D6)

The adopted defaults, still without snapshot selection, against
the 8000-episode DQN at `541cb44`. Six new training seeds and a
new 200-career block. The old code ran from that commit in its
own worktree. Careers use the same `split_rngs` split in both.

| | Old | New |
| --- | --- | --- |
| Mean of means | 6474.8 | 6166.0 |
| Minimum seed mean | 6304.8 | 3668.3 |
| Finished | 811/1200 | 801/1200 |
| Early deaths | 7 | 79 |
| Final epsilon | 0.273–0.305 | 0.05 |
| Steps | 273 918–286 371 | 400 001–400 048 |
| Episodes | 8000 | 7569–7771 |
| Train | 141.9 s | 228.3 s |

Paired new minus old: −308.9 ± 517.8. Twice the SE is 1035.6.
The six differences, in seed order 3741676989, 2687964709,
4109534177, 2057313969, 253451421, 3392167989, were +465.3,
−452.3, −254.8, +605.6, +520.0, and −2737.1. The last of those
is seed 3392167989 (3668.3, 31/200, 59 early deaths). The old
runs did not collapse; their lowest seed mean was 6304.8, and
all seven early deaths were Japanese Derby.

Old endings over 1200 careers: finished 811, URA Finale Finals
183, Japan Cup 97, Arima Kinen 63, URA Finale Semifinal 38,
Japanese Derby 7, Tenno Sho (Spring) 1. New endings: finished
801, Finals 83, Japan Cup 76, Tenno Sho 71, Japanese Derby 58,
Kikuka Sho 51, Arima Kinen 27, Semifinal 12, Spring Stakes 12,
Satsuki Sho 9. The 79 early deaths are Derby 58, Spring Stakes
12, and Satsuki Sho 9. Kikuka Sho is turn 43, so it is not early.

The shipped defaults did not beat the previous DQN. The failure
is one collapsed seed, not a lower score on every seed.

## Snapshot selection (D7)

Same six D6 seeds and the same eval block. Old and new rows are
the D6 runs. The new candidates sit on the adopted defaults.
`end0.10` and `end0.20` only change the floor. `lr3e-3` is Adam
at 0.003 with rewards left unscaled. `best_snapshot` scores 50
held-out careers every 20 000 steps after step 120 000 and
restores the best weights. The held-out seeds are 50 distinct
32-bit ints from the third `SeedSequence` child of the training
seed, so they do not alias `split_rngs`. For these twelve
training seeds and for seed 42, none of those draws landed in
the three eval blocks above. `end0.10_snapshot` is that rule
plus a floor of 0.10. It was run because, among the three
single changes, floor 0.10 had both the highest mean and the
highest minimum.

| Config | Mean | Min seed | vs old | Finished | Early | Train |
| --- | --- | --- | --- | --- | --- | --- |
| Old | 6474.8 | 6304.8 |  | 811/1200 | 7 | 141.9 s |
| New, no snapshot | 6166.0 | 3668.3 | −308.9 ± 517.8 | 801/1200 | 79 | 228.3 s |
| Floor 0.10 | 6831.8 | 6693.8 | +357.0 ± 63.2 | 822/1200 | 6 | 234.3 s |
| Floor 0.20 | 6571.4 | 5830.9 | +96.5 ± 124.0 | 780/1200 | 9 | 220.5 s |
| `lr=0.003` | 6417.7 | 5891.3 | −57.1 ± 119.7 | 618/1200 | 4 | 221.5 s |
| Best snapshot | 6889.5 | 6842.3 | +414.7 ± 63.0 | 894/1200 | 2 | 236.8 s |
| Floor 0.10 + snapshot | 6909.6 | 6800.6 | +434.8 ± 62.1 | 934/1200 | 3 | 238.7 s |

Against the no-snapshot runs, none of the five cleared 2 SE.
The collapsed seed makes that SE about 520. Against the old DQN,
floor 0.10, best snapshot, and the combination cleared. Floor
0.20 did not (its minimum was 5830.9, seed 253451421, though
that seed still finished 172/200). `lr=0.003` did not (minimum
5891.3, and seed 2687964709 finished 23/200). Seed 3392167989
came back under floor 0.10 (6951.7, 0 early), best snapshot
(6903.6, restored step 320 010, 1 early), and the combination
(6923.0, step 380 027, 0 early).

Restored steps for best snapshot, in the D6 seed order:
360 034, 300 045, 400 003, 320 011, 360 029, 320 010. One of the
six kept the final step. The combination, same order, restored
400 040, 400 046, 380 045, 380 031, 280 019, 380 027.

The screening seeds were the ones that had exposed the collapse,
so both snapshot configs were confirmed on six fresh seeds and a
fresh eval block.

| | Old | Best snapshot | Floor 0.10 + snapshot |
| --- | --- | --- | --- |
| Mean of means | 6471.1 | 6909.8 | 6784.9 |
| Minimum seed mean | 6354.7 | 6789.7 | 6697.8 |
| Finished | 950/1200 | 1019/1200 | 889/1200 |
| Early deaths | 11 | 2 | 5 |
| Train | 136.8 s | 241.3 s | 242.3 s |

Confirmation differences, in the confirmation seed order: best
snapshot minus old +438.7 ± 57.8 (clears; the six differences
were +181.0, +442.0, +438.5, +574.6, +562.9, +433.4). Floor 0.10
plus snapshot minus old +313.8 ± 51.7 (clears). Floor 0.10 plus
snapshot minus best snapshot −124.9 ± 23.6, negative on all six
seeds.

Pooled over all twelve seeds (the D6 block, then the confirmation
block; the two blocks are different careers):

| Difference | Mean ± SE | Clears |
| --- | --- | --- |
| Best snapshot − old | +426.7 ± 40.9 | yes |
| Floor 0.10 + snapshot − old | +374.3 ± 42.6 | yes |
| Floor 0.10 + snapshot − best snapshot | −52.4 ± 29.8 | no |

Adopted best snapshot. The floor stays 0.05. On the confirmation
careers the higher floor finished fewer runs (889/1200) than the
old DQN (950/1200), despite a higher mean.

Confirmation endings for the adopted config, 1200 careers:
finished 1019, URA Finale Finals 107, Arima Kinen 27, Japan Cup
20, URA Finale Semifinal 18, Tenno Sho (Spring) 7, Japanese Derby
2. The two early deaths are the two Derbies. On the screening
block the same config finished 894/1200, with Japan Cup 82,
Arima Kinen 57, and Finals 130.

## Findings

Training-seed variance is the large term. Without snapshot
selection, one seed in fourteen of the adopted step-budget runs
fell to 3668, and the six-seed comparison against the old DQN
was −309 ± 518. With snapshot selection, the twelve seed means
sit between 6789.7 and 6950.6. A single `compare.py` career is
still one draw from a distribution whose within-run standard
deviation is several hundred points.

What remains are late races. On the fresh confirmation block the
adopted config's failures were URA Finale Finals, Arima Kinen,
Japan Cup, and the Semifinal, plus seven Tenno Sho and two
Derbies. Early collapse did not return (2 early deaths in 1200
on each of the two six-seed blocks).

These did not help, on the measurements above. A floor of 0.02
or 0.20. Decaying over 50% or 70% of the budget, once 30% had
been adopted. A 200 000 buffer, and 475 000 steps. Reward scales
0.01 and 0.001, Huber loss, and gradient clipping at 10.
`learn_every` 5 (also over 300 s alone), a `[128, 128]` network,
and target sync every 1000 steps. Batch 256 gained 126.0 against
a 2 SE bar of 126.9. Speed pace and stamina pace. A floor of
0.10, with or without snapshots: it cleared against the old DQN
on the screening seeds, and with snapshots it cleared again on
fresh seeds, but it lost to best snapshot on every confirmation
seed and finished fewer careers. Learning rate 0.003 with
unscaled rewards did not clear either.

The snapshot objective is a 50-career mean, not the 200-career
report. It usually restored a step between 260 000 and 380 000.
It can restore the final weights, and on one confirmation run of
the 0.10 combination it restored step 160 022, the third
checkpoint after the floor.

## Result

`uv run python -m umamusime.compare` on the shipped code (paired
seeds 42–44, default training seed 42):

| Method | Runs | Best | Time |
| --- | --- | --- | --- |
| Random | 954.2, 636.8, 1026.8 | 1026.8, Spring Stakes soft fail | instant |
| MCTS | 5959.7, 4441.9, 2902.7 | 5959.7, URA Finale Finals soft fail | 49.5 s for one game |
| DQN | 7430.7, 6409.4, 7274.9 | 7430.7, finished, no training fails | 249.2 s training |

Three careers are far too few to compare methods; the paired
200-career results above are the evidence for the settings.
