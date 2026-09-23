# MCTS tuning

OpenSpiel's `MCTSBot` with random rollouts to a terminal career state. The
algorithm was not changed; only constructor and evaluator parameters were
swept. Defaults live in `src/umamusime/bots/mcts.py`.

## Shipped settings

| Parameter | Value |
| --- | --- |
| Rollout length | Full career (until terminal) |
| `uct_c` | 200 |
| `max_simulations` | 100 |
| `n_rollouts` | 15 |
| Chance sampling in rollouts | Cumulative sum (not `numpy.choice(p=...)`) |

A game is about 40 seconds. Careers that reach the finale can take about a
minute. Career chance nodes and the search use separate RNGs so a fixed seed
does not reshuffle environment luck when search settings change.

## Why full-career rollouts

`returns()` is the absolute career score. A short random rollout only sees
the next few turns of gains, so every root action looks similar. Soft-fail
race gates (especially stamina) sit further ahead than a 6-turn window, so
raising `uct_c` or reshuffling the simulations/rollouts budget at that
horizon did almost nothing to finish rate.

On **96 paired seeds** (environment seeds 2000–2095), full-career rollouts
at `uct_c=200` compared with a 6-turn horizon at `uct_c=2`:

| Setup | Finished | Mean turns | Mean score | Median score | Mean time |
| --- | --- | --- | --- | --- | --- |
| 6-turn horizon, `uct_c=2` | 1/96 | 41.8 | 2990 ± 227 | 2894 | 13.7s (max 27s) |
| Full career, `uct_c=2` | 2/96 | 52.2 | 3733 ± 304 | 2892 | 34.3s (max 54s) |
| Full career, `uct_c=200` | 12/96 | 60.8 | 4970 ± 349 | 5761 | 41.8s (max 62s) |
| Full career, `uct_c=400` | 8/96 | 57.9 | 4588 ± 343 | 4150 | 39.8s (max 63s) |

Paired differences against the 6-turn baseline:

- Full career, `uct_c=2`: **+743 ± 355** score, **+10.4 ± 3.2** turns
- Full career, `uct_c=200`: **+1981 ± 358** score, **+19.0 ± 3.1** turns (75/96 seeds higher)
- Full career, `uct_c=400`: **+1598 ± 363** score, **+16.1 ± 3.1** turns

`uct_c=200` versus `uct_c=2` at full horizon: **+1238 ± 452** score,
**+8.6 ± 3.7** turns (69/96). `uct_c=400` versus `uct_c=200`: **−383 ± 470**
(45/96). An earlier 32-seed tuning set had favored 400; that did not hold on
fresh seeds.

Most careers still soft-fail a stamina gate. At the shipped settings the
common endings were Kikuka Sho (33), Japan Cup (30), and finished careers
(12). Parameter tuning alone does not clear Tenno Sho (Spring)'s 815 stamina
gate reliably.

## What else was tried

Budget was capped at about 60 seconds per game. Screening used 32 seeds;
confirmation used 96 fresh seeds.

**`uct_c` at a 6-turn horizon.** Values from 2 to 400 were within noise.
Root visit shares explained why: with absolute scores in the thousands and
sibling Q values tens of points apart, `uct_c=2` puts ~94% of visits on one
action after a single sample. Raising `uct_c` to ~200 spreads visits, but a
short rollout still cannot see the next gate.

**Simulations × rollouts at fixed total rollouts (~1500).** Splits from
`(1500, 1)` through `(25, 60)` at a 6-turn horizon stayed within noise.
`(300, 5)` looked slightly better on 32 seeds but finish rate stayed near
zero. With ~100 simulations the tree is effectively one ply: six placement
chance nodes sit between decisions, so extra simulations mostly resample
next-turn chance rather than deepen the tree.

**Horizon.** At ~50s/game budgets, horizon 12 matched horizon 6. From about
24 turns upward, more careers got past Kikuka Sho. A 48-turn cutoff matched
no cutoff on all 32 seeds, because random rollouts usually die at a race
before then. Full career is the default.

**Other `MCTSBot` flags.** `solve=False` matched `solve=True` on the tuning
seeds. `dont_return_chance_node=True` and 140 simulations did not beat the
chosen setup inside the time cap. 50 simulations at `uct_c=200` was about
half the wall time and not clearly worse on 32 seeds; the confirmed result
uses 100.

## Evaluator note

Rollouts use a cumulative-sum draw instead of `numpy.random.choice(..., p=...)`.
`choice` re-validates the distribution on every call (~5µs) versus ~0.3µs for
the cumulative sum. With eight chance nodes a turn, that dominates search
time. The rollout policy is still uniform over legal actions.
