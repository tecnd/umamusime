# Umamusime

Single-player, 60 turns. You start at 100 energy with all stats at 0. The score is the sum of per-turn rewards from **successful** actions.

Each stat (speed, stamina, power, guts, wit) is clipped to `[0, 1200]`. Energy is clipped to `[0, 100]`.

## Actions

| Action | Energy | Stats on success | Score | Can fail? |
| --- | --- | --- | --- | --- |
| Rest | +50 | — | 0 | No |
| Train speed | −20 | +10 speed, +5 power | 3.0 | Yes |
| Train stamina | −20 | +9 stamina, +4 guts | 1.0 | Yes |
| Train power | −20 | +5 stamina, +8 guts | 1.0 | Yes |
| Train guts | −20 | +4 speed, +4 power, +8 guts | 1.0 | Yes |
| Train wit | +5 | +2 speed, +9 wit | 1.5 | No |

## Failure

Speed, stamina, power, and guts training can fail. The fail chance uses energy **after** the −20 cost, not current energy.

- Remaining energy ≥ 50: 0% fail
- Remaining energy 0: 99% fail
- Between 0 and 50: linear from 99% to 0%

On failure:

- The turn is used
- Energy is not spent
- Score is 0
- The training’s named stat drops by 10 (failed guts → −10 guts), then clipped to `[0, 1200]`
