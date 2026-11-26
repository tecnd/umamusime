# MCTS tuning

OpenSpiel's `MCTSBot` with uniform-random rollouts to a terminal career.
The rollout policy stays uniform. The bot is unchanged; the levers are
`uct_c`, `max_simulations`, and `n_rollouts`. Defaults live in
`src/umamusime/bots/mcts.py`.

## Shipped settings

| Parameter | Value |
| --- | --- |
| Rollout policy | Uniform random, to a terminal state |
| `uct_c` | 200 |
| `max_simulations` | 100 |
| `n_rollouts` | 15 |
| Chance sampling in rollouts | Cumulative sum (not `numpy.choice(p=...)`) |

Career chance nodes and the search use separate RNGs (`split_rngs`), so a
fixed seed does not reshuffle placements or fail rolls when search settings
change. A game is about 41 s. The measured times are below.

## Little headroom

Little headroom is left. With uniform rollouts and `MCTSBot` unchanged, a
larger budget does not produce a gain that separates from noise, and it
does not stop careers dying at Kikuka Sho.

Uniform rollouts from the opening decision end near turn 30. None of 2,000
got past Kikuka Sho, so a rollout return is the score at the next race gate
or two, not the score of a finished career. From later states of a career
that itself died at Kikuka Sho, 1,500 rollouts still almost never passed
that race (fraction 0.000 to 0.001).

The tree does not look further than the root decision. Search starts at
the decision, after this turn's placements have already been resolved.
The next training decision sits behind an optional fail/success roll and
that turn's six placement chance nodes (6 outcomes each, so
6^6 = 46,656 placement combinations). OpenSpiel expands a node only once
it has already been evaluated, and it samples chance children rather than
trying them all. A few hundred simulations almost never walk one of those
paths to the next decision. The search is a one-decision look-ahead
scored by rollouts.

Repeating the search usually does not change the move. At three of four
saved states the shipped budget (100 × 15) picked the same action in 15
or 16 of 16 searches, and 700 × 15 picked that same action in 8 of 8.
Only the earliest of those states was noisy.

On 48 fresh paired seeds, 500 × 15 against the shipped 100 × 15 (both
`uct_c=200`) changed the mean score by +84.4 ± 340.6, and was higher on
25 of 48 seeds. The mean among careers that passed Kikuka Sho rose by
+378.8 ± 290.5, about 1.3 standard errors. Passes went from 29/48 to
26/48. Passing Kikuka Sho is close to a coin flip at either budget.
500 × 15 is about as much search as fits in 300 s. With all four cores
busy, its 78-turn careers averaged 285.4 s, and the longest took 328.8 s,
over that cap. Because that budget was rejected, `uct_c` at 500
simulations and rollout speedups were not pursued. Doing better would
need a rollout policy that can see the race gates. That was out of scope.

`±` below is a standard error. For a mean it is the sample standard
deviation divided by sqrt(n). For a paired difference it is that quantity
on the per-seed differences. For the gap between the two pass-Kikuka
means it is the root-sum-square of the two standard errors. A career
passes Kikuka Sho when it finishes, or when it fails a race after turn 43.
Failing URA Finale Finals is not a finish: the career ends on turn 78
with that race as the ending.

## 100 × 15 against 500 × 15

48 fresh paired seeds, `uct_c=200`, 15 rollouts. Wall times are with all
four cores busy.

| | 100 × 15 | 500 × 15 |
| --- | --- | --- |
| Mean score | 4598.86 | 4683.28 |
| Median score | 4238.50 | 4209.05 |
| Passed Kikuka Sho | 29/48 | 26/48 |
| Mean score among those | 5753.63 ± 201.76 | 6132.42 ± 209.02 |
| Finished | 3 | 4 |
| Mean wall time | 41.1 s | 210.1 s |
| Max wall time | 62.0 s | 328.8 s |

Endings at 100 × 15: Kikuka Sho 18, Japan Cup 13, Tenno Sho (Spring) 8,
finished 3, URA Finale Finals 3, URA Finale Semifinal 2, Spring Stakes 1.

Endings at 500 × 15: Kikuka Sho 22, Japan Cup 11, Tenno Sho (Spring) 5,
finished 4, URA Finale Finals 4, Arima Kinen 2.

Paired score difference (500 minus 100), all 48 seeds: +84.4 ± 340.6,
higher on 25 seeds. Difference of the pass-Kikuka means: +378.8 ± 290.5.
Paired passes: both 17, only 100 × 15 on 12, only 500 × 15 on 9, neither
on 10.

At 500 × 15 the eight careers with end turn 78 averaged 285.4 s (max
328.8 s). The longest of them is over a 300 s cap under that load.

The same comparison on 12 seeds (42, 43, 44, and nine fresh seeds listed
below) agrees in direction and is noisier. Shipped settings: mean 4511,
median 3860, 41.6 s per game (max 58 s). Seeds 42, 43, and 44 score
5959.7, 4441.9, and 2902.7. At 500 × 15 the mean was 4865, paired
difference +354 ± 779. Six of the 12 died at Kikuka Sho at both budgets,
but not the same six: three that died at 100 × 15 passed at 500 × 15, and
three that passed then died.

## What the rollouts see

2,000 uniform rollouts from the first decision of a career: mean return
1117, standard deviation 245, mean end turn 29.9. Endings were Spring
Stakes 1,590, Asahi Hai 165, Satsuki Sho 141, Japanese Derby 64, Kikuka
Sho 40, and none past Kikuka Sho. About 0.92 ms per rollout.

From four decision states of the seed-44 career, which later died at
Kikuka Sho (turns 12, 24, 34, and 40), 1,500 uniform rollouts each put
the fraction past Kikuka Sho at 0.000 to 0.001.

Independent searches from those states: at turns 24, 34, and 40, 100 × 15
chose one action in 15 or 16 of 16 searches, and 700 × 15 chose that same
action in 8 of 8. Turn 12 split 5 / 9 / 2 across three actions at
100 × 15; 7 of 8 searches agreed at 700 × 15.

Turn 34 of that career is the blind spot in miniature. Stamina was 416,
Kikuka Sho needs 556 at turn 43, energy was 57, and every training had a
0% fail chance. The search rested anyway: Q(rest) 2350 against
Q(train stamina) 2312, and 16 of 16 searches chose rest.

## One-decision tree

This is a reading of `UmaState` and OpenSpiel's `mcts.py`, not a count of
nodes from a run.

`_start_current_turn` puts a non-race turn in `Phase.PLACEMENT`. Each of
the six support cards is its own chance node (`NUM_PLACEMENT_OUTCOMES`
is 6: five facilities or away). After the sixth card the phase is
`DECISION`. If the chosen training has a fail chance strictly between 0
and 1, `apply_action` inserts one fail/success chance node before the
turn advances. Race turns call `_resolve_race` and never ask for a
decision. Two consecutive training decisions are therefore separated by
that optional result node and by six placement nodes. The placements
alone have 6^6 = 46,656 combinations.

`MCTSBot` does not pass `dont_return_chance_node`, so OpenSpiel's default
(`False`) applies. `_apply_tree_policy` treats a node with `explore_count`
0 as a leaf: the first visit evaluates it, and only a later visit expands
its children and steps once. At a chance node the child is sampled from
`chance_outcomes`, not chosen by UCT. Reaching the next decision means
revisiting one placement path often enough to walk those chance nodes one
ply at a time. A few hundred simulations cannot do that for more than a
vanishing fraction of 46,656 paths, so a second-ply decision is almost
never a node in the tree. Root values are rollout returns from just after
the chosen action, or from a chance node further along a path that was
sampled again. They are not values from a searched later decision.

## Evaluator

`FastRolloutEvaluator` draws chance outcomes with a cumulative sum instead
of `numpy.random.choice(..., p=...)`. Legal actions are still chosen
uniformly. An earlier timing, not re-measured, put `choice` at about 5 µs
per call and the cumulative sum at about 0.3 µs, and described that gap as
most of the search time when a turn has several chance nodes. A profile of
rollouts on the current code found `is_chance_node()` at about 18% of
rollout time.

## Earlier sweeps (not re-measured)

Kept only as the reason full-career rollouts and `uct_c=200` are the
shipped settings. These runs were not repeated. They also predate the
split between the career RNG and the search RNG, and they stopped at
about 60 s per game. The 96-seed rows used environment seeds 2000–2095.
That note reported 12 of 96 careers finished at the shipped settings.
The 48-seed run above finished 3 of 48. Do not treat 12/96 as the current
finish rate.

| Setup | Finished | Mean turns | Mean score | Median | Mean time |
| --- | --- | --- | --- | --- | --- |
| 6-turn horizon, `uct_c=2` | 1/96 | 41.8 | 2990 ± 227 | 2894 | 13.7 s (max 27 s) |
| Full career, `uct_c=2` | 2/96 | 52.2 | 3733 ± 304 | 2892 | 34.3 s (max 54 s) |
| Full career, `uct_c=200` | 12/96 | 60.8 | 4970 ± 349 | 5761 | 41.8 s (max 62 s) |
| Full career, `uct_c=400` | 8/96 | 57.9 | 4588 ± 343 | 4150 | 39.8 s (max 63 s) |

Paired against the 6-turn, `uct_c=2` row: full career at `uct_c=2` was
+743 ± 355 score and +10.4 ± 3.2 turns; at `uct_c=200`, +1981 ± 358 and
+19.0 ± 3.1 turns (75/96 higher); at `uct_c=400`, +1598 ± 363 and
+16.1 ± 3.1 turns. At full horizon, `uct_c=200` beat `uct_c=2` by
+1238 ± 452 score and +8.6 ± 3.7 turns (69/96). `uct_c=400` against 200
was −383 ± 470 (45/96 higher for 400). A 32-seed set had favored 400;
these 96 seeds did not.

Also not re-measured, under the same 60 s cap. At a 6-turn horizon,
`uct_c` from 2 to 400 stayed within noise. `uct_c=2` put about 94% of
root visits on one action after a single sample, because scores are in
the thousands and sibling values differ by tens. Splits of about 1,500
rollouts, from (1500, 1) through (25, 60), stayed within noise at that
horizon. (300, 5) looked slightly better on 32 seeds, with the finish
rate still near zero. Horizon 12 matched horizon 6 near 50 s per game.
A 48-turn cutoff matched no cutoff on all 32 seeds in that run.
`solve=False` matched `solve=True`. `dont_return_chance_node=True` with
140 simulations did not beat the chosen setup inside the cap. 50
simulations was about half the wall time and not clearly worse on 32
seeds; the run that was kept used 100.

## Reproducing a game

The sweep harness is not in the repo. One game is:

```python
bot = mcts.MCTSBot(
    uct_c=200,
    max_simulations=100,
    n_rollouts=15,
    random_state=split_rngs(seed).bot_rng,
)
result = base.play(bot, seed=seed)
```

`mcts` and `base` are `umamusime.bots`, and `split_rngs` is
`umamusime.rng.split_rngs`. Seeds were drawn with `secrets.randbits(32)`.

The 48 paired seeds, in the order they were drawn:

553861838, 321477558, 708916399, 3414207265, 787120070, 3115485769,
3035911500, 2829040175, 656684982, 3470580851, 3589812242, 813646982,
2222993079, 3500874148, 1741810890, 3620779820, 2918101424, 486677752,
4182591388, 1658835958, 2191351422, 1484945804, 3293237028, 2117666266,
1339978612, 2806187252, 2763373120, 3627844658, 518549434, 1837317992,
4230699660, 3135650588, 475505155, 3179566602, 1705513981, 808043655,
1433735515, 851742047, 1294083515, 285528583, 400226253, 4273954939,
3833760548, 1448032476, 3430898394, 2869847314, 2742486213, 1568230823.

The 12-seed run used 42, 43, 44, and these nine:

1895120625, 3837576509, 3035940293, 2207217333, 3510745758, 3831102505,
408915293, 4130134562, 3917317507.
