# Umamusime

Single-player, 72 turns (3 years). Each turn is half a month: turn 1 is Year 1, Early January and turn 72 is Year 3, Late December. You start at 100 energy with all stats and skill points at 0. The score is the sum of per-turn rewards from **successful** actions, including 0.9 per skill point gained.

In years 2 and 3, Early July through Late August (inclusive) are summer camp turns.

Each stat (speed, stamina, power, guts, wit) is clipped to `[0, 1200]`. Energy is clipped to `[0, 100]`. Skill points are not capped. There is no dedicated skill-point training.

## Actions

Rest is not a facility. Training facilities start at level 1, gain a level after **4 successful uses**, and cap at level 5. Failed trainings do not count as uses. Fail chance uses energy **after** that training’s current energy cost.

| Action | Energy | Stats on success | Skill pts | Score | Can fail? |
| --- | --- | --- | --- | --- | --- |
| Rest | +50 | — | 0 | 0 | No |
| Train speed | −21…−27 | speed / power (see facilities) | 2 | 4.8 | Yes |
| Train stamina | −19…−25 | stamina / guts | 2 | 2.8 | Yes |
| Train power | −20…−26 | stamina / power | 2 | 2.8 | Yes |
| Train guts | −22…−28 | speed / power / guts | 2 | 2.8 | Yes |
| Train wit | +5 | speed / wit | 4 | 5.1 | No |

Score on success is the action reward (Rest 0, Speed 3.0, Stamina/Power/Guts 1.0, Wit 1.5) plus `0.9 ×` skill points.

### Training facilities

| Training | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
| --- | --- | --- | --- | --- | --- |
| Speed (speed / power / energy) | +10 / +5 / −21 | +11 / +5 / −22 | +12 / +5 / −23 | +13 / +6 / −25 | +14 / +7 / −27 |
| Stamina (stamina / guts / energy) | +9 / +4 / −19 | +10 / +4 / −20 | +11 / +4 / −21 | +12 / +5 / −23 | +13 / +6 / −25 |
| Power (stamina / power / energy) | +5 / +8 / −20 | +5 / +9 / −21 | +5 / +10 / −22 | +6 / +11 / −24 | +7 / +12 / −26 |
| Guts (speed / power / guts / energy) | +4 / +4 / +8 / −22 | +4 / +4 / +9 / −23 | +4 / +4 / +10 / −24 | +5 / +4 / +11 / −26 | +5 / +5 / +12 / −28 |
| Wit (speed / wit / energy) | +2 / +9 / +5 | +2 / +10 / +5 | +2 / +11 / +5 | +3 / +12 / +5 | +4 / +13 / +5 |

## Failure

Speed, stamina, power, and guts training can fail. The fail chance uses energy **after** that training’s current energy cost, not current energy.

- Remaining energy ≥ 50: 0% fail
- Remaining energy 0: 99% fail
- Between 0 and 50: linear from 99% to 0%

On failure:

- The turn is used
- Energy is not spent
- Score is 0
- No skill points are awarded
- The facility is not used (no level-up progress)
- The training’s named stat drops by 10 (failed guts → −10 guts), then clipped to `[0, 1200]`
