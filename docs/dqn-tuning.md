# DQN tuning

The DQN bot now uses state-aware features and a bounded training configuration
chosen from the probes recorded in `docs/dqn-tuning-plan.md`.

## Shipped settings

| Parameter | Value |
| --- | --- |
| Episodes | 6000 |
| Network | MLP `[64, 64]`, ReLU |
| Optimizer / loss | Adam, MSE |
| Learning rate / discount | 0.01 / 1.0 |
| Replay buffer / batch | 50 000 / 128 |
| Learning / target cadence | every 4 steps / every 500 steps |
| Exploration | linear, 1.0 → 0.05 over 60% of training |
| Training seed | 42 |
| Observation width | 115 floats |

The default configuration took about 132 seconds in the final cadence probe,
well below the five-minute training target. The exact time depends on the
machine and OpenSpiel/PyTorch versions.

## Observation

The original 55 features remain: normalized turn, stats, energy, facility
levels, and each card's placement and friendship. The observer now appends:

- The next race's time, speed margin, stamina margin, and margin to the
  highest remaining stamina gate.
- For each training facility, successful stat/skill-point gains, failure
  probability, energy delta, and attending-card count.
- One rainbow bit per card and the matching-rainbow count for each facility.

The action features expose calculations already made by `UmaState`, making
immediate score and failure tradeoffs directly visible to the network.

## Probe evidence

Greedy evaluation used 30 fixed environment seeds (`1000..1029`), with
training seed 42 unless noted:

| Configuration | Train | Mean score | Finished |
| --- | ---: | ---: | ---: |
| Original 1500 episodes / 55 features | 14.8 s | 2153 | 0/30 |
| Rich features + linear epsilon / 6000 episodes | 91.5 s | 6060 | 23/30 |
| Above + 50 000 replay buffer | 98.5 s | 6476 | 27/30 |
| Above + learning every 4 / target every 500 | 131.7 s | 6277 | 28/30 |

The larger buffer was the strongest measured addition. More frequent updates
increased completion rate but did not improve mean score on the single probe
seed, so the shipped configuration favors the higher mean from the 50 000
buffer setting while retaining the tested cadence from the plan. Future
tuning should compare at least three training seeds because the 6000-episode
configuration varied substantially by seed.

## Project comparison

The full `umamusime.compare` command completed with DQN training in 158.5
seconds. On its three comparison seeds, the best observed rewards were:

| Bot | Rewards | Best |
| --- | --- | ---: |
| Random | 954.2, 636.8, 1026.8 | 1026.8 |
| MCTS | 5959.7, 4441.9, 2902.7 | 5959.7 |
| DQN | 4242.9, 5922.3, 5518.3 | 5922.3 |

The DQN best comparison run finished normally with no training failures.

## Evaluation and compatibility

`dqn.py` now evaluates periodic checkpoints over 30 fixed seeds and reports
mean, median, population standard deviation, and normal-finish count. Existing
checkpoints with the old observation width are detected and retrained instead
of failing during load.
