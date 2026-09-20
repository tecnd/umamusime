# Umamusime

Umamusime is a simplified implementation of Umamusume in OpenSpiel/pyspiel to compare AI algorithms for fun. By default it uses initial stats and growth rates for Mihono Bourbon. Training facility values are copied from the buffed URA Finale scenario.

## What's implemented

* Support cards
* Training calculations
* Scoring
* Career races (configurable schedule; soft-fail on speed/stamina gates)
* Partial events (Inspiration stat buffs and fixed energy gifts before the labeled turn)

## What's not implemented

* Skills and skill hints
* Inheritance
* Full event / scenario scripting beyond the fixed Inspiration and energy gifts

## Known issues

* Race turns skip training with no race rewards; only the soft-fail gate is modelled
