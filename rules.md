# Umamusime

Single-player career, 72 turns over 3 years. Each turn is half a month: turn 1 is Year 1, Early January and turn 72 is Year 3, Late December.

You start at 100 energy. Speed, stamina, power, guts, wit, and skill points start at 0, plus whatever the support deck grants as **initial {stat}**. The default deck therefore begins at 20 speed, 35 stamina, 55 wit. Those starting stats do not count toward score.

In years 2 and 3, Early July through Late August (inclusive) are summer camp turns: every training facility is treated as level 5 for that turn, then returns to its real level. Successful camp trainings still count as facility uses. Support cards have no extra camp rules.

Each of speed, stamina, power, guts, and wit is clipped to `[0, 1200]`. Energy is clipped to `[0, 100]`. Skill points are not capped. There is no dedicated skill-point training.

## Score

Score is the sum of per-turn rewards. Each turn's reward is the weighted sum of the stat **deltas that actually landed** after clipping:

| Stat | Speed | Stamina | Power | Guts | Wit | Skill points |
| --- | --- | --- | --- | --- | --- | --- |
| Weight | 3.0 | 1.0 | 1.0 | 1.0 | 1.5 | 0.9 |

Energy changes are not scored. A failed training scores its clipped −10 as negative reward (failed speed at 0 speed scores 0, not −30).

## Turn structure

1. Each of the six support cards independently rolls where it shows up (chance).
2. You pick an action.
3. Speed, stamina, power, and guts then roll fail/success (chance). Rest and wit never fail.

## Actions

Rest is not a facility. Training facilities start at level 1, gain a level after **4 successful uses**, and cap at level 5. Failed trainings do not count as uses. Fail chance uses energy **after** that training's current energy cost, including wit friendship recovery on wit training.

| Action | Energy | Base stats on success | Base skill pts | Can fail? |
| --- | --- | --- | --- | --- |
| Rest | +50 | — | 0 | No |
| Train speed | −21…−27 | speed / power | 4 | Yes |
| Train stamina | −19…−25 | stamina / guts | 4 | Yes |
| Train power | −20…−26 | stamina / power | 4 | Yes |
| Train guts | −22…−28 | speed / power / guts | 4 | Yes |
| Train wit | +5 | speed / wit | 5 | No |

### Training facilities

Each level-up adds +1 to the main stat. The first two level-ups cost 1 more energy; the last two also add +1 to a secondary stat and cost 2 more energy. Wit energy stays +5.

| Training | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
| --- | --- | --- | --- | --- | --- |
| Speed (speed / power / SP / energy) | +11 / +6 / +4 / −21 | +12 / +6 / +4 / −22 | +13 / +6 / +4 / −23 | +14 / +7 / +4 / −25 | +15 / +8 / +4 / −27 |
| Stamina (stamina / guts / SP / energy) | +10 / +6 / +4 / −19 | +11 / +6 / +4 / −20 | +12 / +6 / +4 / −21 | +13 / +7 / +4 / −23 | +14 / +8 / +4 / −25 |
| Power (stamina / power / SP / energy) | +6 / +9 / +4 / −20 | +6 / +10 / +4 / −21 | +6 / +11 / +4 / −22 | +7 / +12 / +4 / −24 | +8 / +13 / +4 / −26 |
| Guts (speed / power / guts / SP / energy) | +5 / +5 / +8 / +4 / −22 | +5 / +5 / +9 / +4 / −23 | +5 / +5 / +10 / +4 / −24 | +6 / +5 / +11 / +4 / −26 | +6 / +6 / +12 / +4 / −28 |
| Wit (speed / wit / SP / energy) | +2 / +10 / +5 / +5 | +2 / +11 / +5 / +5 | +2 / +12 / +5 / +5 | +3 / +13 / +5 / +5 | +4 / +14 / +5 / +5 |

## Support cards

The deck is always exactly six cards, passed as `UmaGame(cards=...)` and defaulting to `cards.DEFAULT_DECK`. Only these card stats are modelled: main stat type, friendship bonus, initial friendship, training effectiveness, mood effect, initial {stat}, {stat} bonus, wit friendship recovery, and specialty priority.

### Default deck

| Card | Main | Friendship bonus | Mood effect | Training eff. | Initial friendship | Specialty priority | Wit recovery | Initial stats | Stat bonus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kitasan Black | speed | 25 | 30 | 15 | 35 | 100 | — | — | power 1 |
| Tokai Teio | speed | 32 | 60 | — | 25 | 35 | — | speed 20 | power 1 |
| Sweep Tosho | speed | 30 | 40 | 10 | 25 | 50 | — | — | skill points 1 |
| Fine Motion | wit | 37.5 | 30 | 15 | 15 | 35 | 5 | wit 35 | wit 1 |
| Super Creek | stamina | 37.5 | — | 15 | 30 | 55 | — | stamina 35 | stamina 1 |
| Agnes Tachyon | wit | 20 | 40 | 5 | 25 | 50 | 4 | wit 20 | wit 1, skill points 1 |

### Placement

Every turn, each card independently rolls one of six outcomes: not showing up, or attending one of the five training facilities. Weights are 50 for not showing up and 100 per training, plus the card's specialty priority added to the training matching its main stat. A facility can end up with several cards, or none. Rest is not a placement target.

Example: a speed card with 60 specialty priority has weights 160 speed, 100 each other training, 50 away (~26% speed, ~16% each other training, ~8% away). Kitasan Black (priority 100) is 200 / 650 ≈ 31% speed and 50 / 650 ≈ 8% away.

### Friendship

Each card has a friendship gauge from 0 to 100, starting at its initial friendship (or 0). A **successful** training gains 5 friendship with every card attending that facility, then clamps to 100. Rest, failures, and cards elsewhere gain nothing. A card at 80 or more friendship is **rainbowed**.

This turn's rainbow checks use friendship **before** the +5, so a card at 75 does not rainbow until the next time you train with it.

### Stat gains

For each stat a training already grants (including skill points), the gain is

```
floor((BaseTraining + StatBonus)
      × ∏(1 + FriendshipBonus/100)
      × (1 + BaseMood × (1 + ΣMoodEffect/100))
      × (1 + ΣTrainingEffectiveness/100)
      × (1 + 0.05 × NumCharacters)
      × (1 + UmaGrowth/100))
```

- `BaseMood` is always 0.2 and `UmaGrowth` is always 0.
- Sums and products run over the cards attending the chosen facility only.
- `NumCharacters` is the number of cards on the chosen facility.
- `FriendshipBonus` only counts for a card that is rainbowed **and** whose main stat matches the training; these multiply together.
- `Mood effect` and `training effectiveness` are always active and do not need friendship.
- `StatBonus` is the sum of the attending cards' bonus for that stat, and only applies where the training's base gain is already non-zero — a power bonus does nothing on wit training.
- Skill points are treated as another stat: they take stat bonuses and the same multiplier.

Rest gains nothing. Cards that did not show up contribute nothing. An empty facility still gets the 1.2 mood term (`1 + 0.2 × 1`), so level-1 speed with nobody there is 13 speed / 7 power / 4 skill points.

Worked example: level 5 speed with rainbow Kitasan Black, non-rainbow Tokai Teio, and Fine Motion. Base is 15 / 8 / 4; the two speed cards add +1 power each. Only Kitasan contributes friendship bonus:

`1.25 × (1 + 0.2 × (1 + 120/100)) × 1.30 × 1.15 = 2.691`

→ 40 speed, 26 power, 10 skill points, −27 energy. Fine Motion's rainbow status does not matter here: her friendship bonus only applies on wit.

### Wit friendship recovery

On wit training, each attending card that is already rainbowed adds its wit friendship recovery to the +5 energy. These stack additively (Fine Motion + Agnes Tachyon both rainbowed → +14 energy).

## Failure

Speed, stamina, power, and guts can fail. The fail chance uses energy **after** that training's current energy cost, not current energy.

- Remaining energy ≥ 50: 0% fail
- Remaining energy 0: 99% fail
- Between 0 and 50: linear from 99% to 0%

On failure:

- The turn is used
- Energy is not spent
- No stats or skill points are gained
- No friendship is gained
- The facility is not used (no level-up progress)
- The training's named stat drops by 10 (failed guts → −10 guts), then clipped to `[0, 1200]`
- Score is the weighted clipped loss
