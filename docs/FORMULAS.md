# Formula catalogue for trader analysis

Source: the user's catalogue of 26 cross-discipline prediction formulas (Claude Docs, 2026-10-29), re-read for
TRADER analysis (one trader's trade list, or a firm's book), not price prediction. Plus 4 extra candidates that
fit trader analysis better than anything in the original list.

Status: PLANNED = already in ROADMAP.md · CANDIDATE = plausible, needs a written case before a study ·
EXCLUDED = no clear use on trader data (keep excluded unless a firm's question needs it).

## How to use this file (rules for Claude)
1. Do NOT try formulas one after another and keep the winners: that is fishing, and the synthetic bench would
   reward whatever happens to fit it. Every formula tested counts as an attempt.
2. To move a CANDIDATE to a study: write a short case here (which firm question, why it beats what is PLANNED,
   what result would kill it), ask the user, then pre-register.
3. Prefer the simplest piece that answers the firm's question; SKILL-03 showed the plain t-stat ranks as well as
   the sophisticated scores.

## Original catalogue (26)
| # | Formula | Origin | Use on trader data | Status |
| --- | --- | --- | --- | --- |
| 1 | Kalman filter | Aerospace | Time-varying edge per trader (is the mean trade result drifting?) | CANDIDATE (alternative to DRIFT-01's CUSUM) |
| 2 | Particle filter | Radar / robotics | Same as Kalman with regimes; too heavy for short trade lists | EXCLUDED |
| 3 | CUSUM | Quality control | Edge lost; size / frequency jumps after losses | PLANNED (DRIFT-01, TILT-01) |
| 4 | EWMA control chart | Quality control | Live monitoring band of a funded trader vs their own history | PLANNED (DRIFT-01) |
| 5 | Hawkes process | Seismology | Trade bursts after losses (tilt); clustered activity across the book | PLANNED (TILT-01, PAYOUT-01) |
| 6 | Omori / ETAS | Seismology | How fast a trader "calms down" after a big loss | CANDIDATE (extension of TILT-01) |
| 7 | SIR contagion | Epidemiology | Spread of a behaviour (copy groups, same signal) across accounts | EXCLUDED (abuse detection is covered by competitors) |
| 8 | CRPS / Brier score | Meteorology | Judge our own probabilities (is "70 % skilled" right 70 % of the time?) | PLANNED (calibration, phase 1) |
| 9 | Isotonic calibration | Meteorology | Fix over-confident skill probabilities | PLANNED (SKILL-04/05 candidate fix) |
| 10 | Holt-Winters | Demand forecasting | Seasonality of a firm's activity / payouts (month-end, news days) | CANDIDATE (PAYOUT-01 only) |
| 11 | Analogue method | Meteorology | "Traders who looked like this at day 30 ended up ..." (nearest neighbours) | CANDIDATE (needs real firm data) |
| 12 | Kaplan-Meier | Medicine | Survival of funded accounts with censoring | PLANNED (SURVIVAL-01) |
| 13 | Cox proportional hazards | Biostatistics | Which behaviours speed up a breach | PLANNED (SURVIVAL-01) |
| 14 | Weibull hazard | Reliability | Does breach risk rise or fall with account age? | CANDIDATE (part of SURVIVAL-01) |
| 15 | Extreme value theory (GPD) | Hydrology / insurance | Tail of a trader's losses; worst payout month for the firm | PLANNED (PAYOUT-01) |
| 16 | Buhlmann credibility | Actuarial | How much to trust a trader's record vs the population | DONE (SKILL-01/02, superseded by NPMLE) |
| 17 | Elo rating | Chess | Ranking traders with uncertainty | PLANNED, optional (RANK-01) |
| 18 | Bradley-Terry | Psychometrics | Same, estimated with standard errors | PLANNED, optional (RANK-01) |
| 19 | James-Stein shrinkage | Statistics / baseball | Shrink every trader's mean to the population | DONE (generalised by empirical Bayes / NPMLE) |
| 20 | Hurst exponent | Hydrology | Streakiness of a trader's results (real persistence vs luck) | CANDIDATE |
| 21 | DFA | Physics / biology | Robust Hurst for short, trending records | EXCLUDED (records too short) |
| 22 | Permutation entropy | Nonlinear dynamics | Mechanical patterns (bots, fixed grids, martingale sequences) | CANDIDATE (TILT-01 extension) |
| 23 | Ornstein-Uhlenbeck | Physics | No direct use on trade lists | EXCLUDED |
| 24 | Lotka-Volterra | Ecology | No direct use on trade lists | EXCLUDED |
| 25 | Logistic growth | Demography | Growth of a firm's funded book; not per trader | EXCLUDED |
| 26 | Lomb-Scargle | Astronomy | Periodic behaviour in irregular trade times (always trades at news releases) | CANDIDATE |

## Extra candidates (not in the original catalogue, strong fit for trader analysis)
| Formula | Origin | Use on trader data | Status |
| --- | --- | --- | --- |
| Sequential probability ratio test (Wald SPRT) | Quality control, WWII | Decide "skilled / not skilled / keep watching" as trades arrive, with fixed error rates; directly answers "how many trades do we need" | CANDIDATE, deprioritised: SKILL-07 showed the small-N problem is unseen rare losses, which an SPRT cannot see either |
| Bayesian online change-point (Adams-MacKay) | Machine learning | Probability that a trader's behaviour changed at each trade | CANDIDATE (alternative for DRIFT-01) |
| Beta-binomial model | Biostatistics | Win rate with uncertainty per style; only useful together with payoff size | CANDIDATE (low priority) |
| Deflated Sharpe ratio | Finance (Bailey, Lopez de Prado) | Correct for how many traders a firm evaluates | PLANNED (copy from the Research Engine) |
| Moderated variance (Smyth 2004, limma) | Genomics | Shrink each trader's variance toward the population's, more for short records | TRIED (dev seeds 9400-9409, SKILL-08 exploration): too weak on heterogeneous traders (d0 about 3); dropped |
| Style-pooled loss tail | Insurance (credibility by risk class) | Borrow the frequency / size of large losses from traders with a similar observed win rate, so a short record of a high win-rate trader is not trusted before its rare losses appear | CANDIDATE (the SKILL-07 failure mode; best tested on real firm data) |
