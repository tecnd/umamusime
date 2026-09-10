# Umamusime

Single-player, 72 turns (3 years). Each turn is half a month: turn 1 is Year 1, Early January and turn 72 is Year 3, Late December. You start at 100 energy with all stats and skill points at 0. The score is the sum of per-turn rewards from **successful** actions, including 0.9 per skill point gained.

In years 2 and 3, Early July through Late August (inclusive) are summer camp turns.

Each stat (speed, stamina, power, guts, wit) is clipped to `[0, 1200]`. Energy is clipped to `[0, 100]`. Skill points are not capped. There is no dedicated skill-point training.

## Actions

| Action | Energy | Stats on success | Skill pts | Score | Can fail? |
| --- | --- | --- | --- | --- | --- |
| Rest | +50 | — | 0 | 0 | No |
| Train speed | −20 | +10 speed, +5 power | 2 | 4.8 | Yes |
| Train stamina | −20 | +9 stamina, +4 guts | 2 | 2.8 | Yes |
| Train power | −20 | +5 stamina, +8 guts | 2 | 2.8 | Yes |
| Train guts | −20 | +4 speed, +4 power, +8 guts | 2 | 2.8 | Yes |
| Train wit | +5 | +2 speed, +9 wit | 4 | 5.1 | No |

Score on success is the action reward (Rest 0, Speed 3.0, Stamina/Power/Guts 1.0, Wit 1.5) plus `0.9 ×` skill points.

## Failure

Speed, stamina, power, and guts training can fail. The fail chance uses energy **after** the −20 cost, not current energy.

- Remaining energy ≥ 50: 0% fail
- Remaining energy 0: 99% fail
- Between 0 and 50: linear from 99% to 0%

On failure:

- The turn is used
- Energy is not spent
- Score is 0
- No skill points are awarded
- The training’s named stat drops by 10 (failed guts → −10 guts), then clipped to `[0, 1200]`
