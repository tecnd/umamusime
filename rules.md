# Umamusime

Single-player, 72 turns (3 years). Each turn is half a month: turn 1 is Year 1, Early January and turn 72 is Year 3, Late December. You start at 100 energy, with all stats and skill points at 0 plus whatever your support cards grant up front.

In years 2 and 3, Early July through Late August (inclusive) are summer camp turns: every training facility is treated as level 5 for that turn, then returns to its real level. Successful camp trainings still count as facility uses. Support cards behave normally during camp.

Each stat (speed, stamina, power, guts, wit) is clipped to `[0, 1200]`. Energy is clipped to `[0, 100]`. Skill points are not capped. There is no dedicated skill-point training.

## Score

Score is the sum of per-turn rewards, and each turn's reward is the weighted sum of the stat changes that actually landed:

| Stat | Speed | Stamina | Power | Guts | Wit | Skill points |
| --- | --- | --- | --- | --- | --- | --- |
| Weight | 3.0 | 1.0 | 1.0 | 1.0 | 1.5 | 0.9 |

A failed training therefore scores its −10 stat loss as negative reward.

## Turn structure

1. Each of the six support cards independently rolls where it shows up (chance).
2. You pick an action.
3. Stat trainings roll fail/success (chance).

## Actions

Rest is not a facility. Training facilities start at level 1, gain a level after **4 successful uses**, and cap at level 5. Failed trainings do not count as uses. Fail chance uses energy **after** that training's current energy cost.

| Action | Energy | Base stats on success | Base skill pts | Can fail? |
| --- | --- | --- | --- | --- |
| Rest | +50 | — | 0 | No |
| Train speed | −21…−27 | speed / power | 4 | Yes |
| Train stamina | −19…−25 | stamina / guts | 4 | Yes |
| Train power | −20…−26 | stamina / power | 4 | Yes |
| Train guts | −22…−28 | speed / power / guts | 4 | Yes |
| Train wit | +5 | speed / wit | 5 | No |

### Training facilities

| Training | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
| --- | --- | --- | --- | --- | --- |
| Speed (speed / power / SP / energy) | +11 / +6 / +4 / −21 | +12 / +6 / +4 / −22 | +13 / +6 / +4 / −23 | +14 / +7 / +4 / −25 | +15 / +8 / +4 / −27 |
| Stamina (stamina / guts / SP / energy) | +10 / +6 / +4 / −19 | +11 / +6 / +4 / −20 | +12 / +6 / +4 / −21 | +13 / +7 / +4 / −23 | +14 / +8 / +4 / −25 |
| Power (stamina / power / SP / energy) | +6 / +9 / +4 / −20 | +6 / +10 / +4 / −21 | +6 / +11 / +4 / −22 | +7 / +12 / +4 / −24 | +8 / +13 / +4 / −26 |
| Guts (speed / power / guts / SP / energy) | +5 / +5 / +8 / +4 / −22 | +5 / +5 / +9 / +4 / −23 | +5 / +5 / +10 / +4 / −24 | +6 / +5 / +11 / +4 / −26 | +6 / +6 / +12 / +4 / −28 |
| Wit (speed / wit / SP / energy) | +2 / +10 / +5 / +5 | +2 / +11 / +5 / +5 | +2 / +12 / +5 / +5 | +3 / +13 / +5 / +5 | +4 / +14 / +5 / +5 |

## Support cards

The deck is exactly six cards, passed as `UmaGame(cards=...)` and defaulting to `cards.DEFAULT_DECK`. Only these card stats are modelled: main stat type, friendship bonus, initial friendship, training effectiveness, mood effect, initial {stat}, {stat} bonus, wit friendship recovery, and specialty priority.

**Initial {stat}** is granted once, at the start of the career.

### Placement

Every turn, each card independently rolls one of six outcomes: not showing up, or attending one of the five training facilities. Weights are 50 for not showing up and 100 per training, plus the card's specialty priority added to the training matching its main stat. A facility can end up with several cards, or none.

### Friendship

Each card has a friendship gauge from 0 to 100, starting at its initial friendship. A **successful** training gains 5 friendship with every card attending that facility; rest, failures, and cards elsewhere gain nothing. A card at 80 or more friendship is **rainbowed**.

### Stat gains

For each stat a training grants, the gain is

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
- `StatBonus` is the sum of the attending cards' bonus for that stat, and only applies to stats the training already grants — a power bonus does nothing on wit training.
- Skill points are treated as another stat: they take stat bonuses and the same multiplier.

Rest gains nothing, and cards that did not show up contribute nothing.

### Wit friendship recovery

On wit training, each attending card that is rainbowed adds its wit friendship recovery to the +5 energy. These stack additively.

## Failure

Speed, stamina, power, and guts training can fail. The fail chance uses energy **after** that training's current energy cost, not current energy.

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
