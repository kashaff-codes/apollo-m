# APOLLO-M — product brief for the review panel

One page per question a panelist is likely to ask, each answered with what the
system does today and the evidence for it. Where something is not done, it says so.

---

## 1. What is it for, and for whom?

**The problem.** Online communities rarely collapse without warning. Toxicity
rises, regulars leave, arguments spill over from neighbouring communities, and
moderators find out when the worst day arrives: a brigade, a doxxing thread, a
mass ban, the community going private. Moderation is reactive because the only
signal people watch is the incident itself.

**What APOLLO-M does.** It measures every community it tracks on the same
0–100 Community Health Index (toxicity, polarisation, author churn,
echo-chamber behaviour), sorts communities into four alert bands with a
recommended action for each, and forecasts where each one is heading. A trust
and safety lead sees, in one place, which communities are deteriorating and
why, days before the incident instead of after it.

**Who buys it.**

| Buyer | Their job | What they get |
|---|---|---|
| Trust & Safety teams at community platforms (forums, Discord-style servers, social apps) | Decide where moderators spend the next hour | A ranked, explained watchlist across thousands of communities |
| Moderation-tooling vendors | Sell dashboards to moderators | APOLLO-M's index and alerts through the REST API, inside their product |
| Brand and game-studio community teams | Keep their own communities healthy | Early warning on their subreddits and servers, and on the topics trending into them |
| Researchers, NGOs, regulators studying online harm | Measure harm across communities over time | Comparable, reproducible measurements and the full method |

**Impact.** Moderator time is the scarce resource. Ranking communities by
measured, explained risk moves that time from the loudest complaint to the
community most likely to deteriorate next, and the explanation (which of the
four measurements moved) tells the moderator what to do.

---

## 2. How reliable is it? (the honest answer)

| What we claim | Evidence | How strong |
|---|---|---|
| **Which communities are more toxic than others** is a stable, real property | Rank correlation of community toxicity between three independent collections (API sample Jul–Aug, complete August dump, Oct–Nov rebuild): ρ = 0.916, 0.864, 0.892 | Strong |
| **The health index orders communities sensibly** | On the declared benchmark (planted ground truth), the index separates deteriorating from stable communities (ROC AUC in `outputs/web_evidence.json`) | Moderate: measured on simulation |
| **The toxicity score means "hostile or profane language"**, not harassment | Against 291 hand-labelled comments, toxic-bert's precision for the toxic class is 0.21; 52 of 78 false positives contain profanity | Weak as a harassment detector, and stated as such |
| **Point forecasts** of next-week toxicity | On held-out days, the TFT does not beat a 14-day recent mean on any of the three real corpora; 80% intervals cover 73–84% | Weak; the live forecast therefore uses the recent mean |
| **Forward test** (trained on August, predicting September days that had not happened) | `scripts/forward_test.py`, results in `outputs/forward_test/` | Pending at time of writing |

**What to tell a panel in one sentence:** *the ranking and the trend are
reliable and reproducible across independent data collections; the absolute
toxicity number measures hostile language rather than harassment; and the
forecasting model does not beat a simple baseline yet, which we publish rather
than hide.*

---

## 3. How does data get in? Is "live" actually live?

Three paths, each with a different job:

| Path | What it is | How fresh | How it runs |
|---|---|---|---|
| **Live ingest** | Newest comments for every tracked community, from Arctic Shift (a public mirror of Reddit, minutes behind it), scored with toxic-bert, health recomputed | Every 6 hours | Automatically: `.github/workflows/live-ingest.yml`. No one runs anything |
| **Topic watch** | "How many posts mention *remote hiring* in the last hour, and where?" A direct query to Arctic Shift at the moment you ask | Minutes | On demand, in the console |
| **Monthly archive** | The complete Reddit comment archive for a month (Academic Torrents `RC_YYYY-MM`, 367.5 M comments for August 2026) | Weeks behind (published monthly) | Once per month, for history and calibration |

**Adding a community** is one line in `live/communities.txt`; it is measured from
the next run on. 53 communities are tracked today.

**Why the archive at all, if there is a live feed?** The live feed is a sample
(the newest comments each run). The archive is every comment. It gives exact
daily volumes and complete author histories for backfill and model training,
and it is how we checked that the cheap live sample measures the same thing:
the API sample and the complete dump rank communities with ρ = 0.916. It is
downloaded, not scraped, and it is not real time.

**What is not live:** the four research corpora (Jul–Aug, August, Oct–Nov,
benchmark) are fixed snapshots by design, so results can be reproduced. The
console labels each one with its window.

---

## 4. How much data is there?

| | Communities | Comments | Window |
|---|---|---|---|
| At mid-evaluation (Aug 2026) | 24 real (+60 simulated) | 116,853 real | Jul 2 – Aug 11 |
| Complete August archive | 39 | 1,087,806 | Aug 1 – 31 |
| Oct–Nov rebuild | 38 | 246,239 | Oct 23 – Nov 15 |
| **Research corpora, total** | **49 distinct** | **1,450,898 real** | Jul – Nov 2026 |
| Forward test (fresh, unseen) | 39 | 91,101 | Sep 1–5, Nov 16–17 |
| Live ingest | 53 tracked | grows every 6 hours | rolling |

---

## 5. If someone wants to buy it, how fast can they run it?

| Option | What they do | Time |
|---|---|---|
| **Hosted** (what runs today) | Send us their community list; we add the lines; they get the console, the web view and API keys | Same day |
| **API inside their tools** | Call `GET /communities`, `/alerts`, `/forecast/{community}` from their moderation dashboard; Prometheus metrics for their monitoring | Days |
| **Self-hosted** | `render.yaml` blueprint (API, monitoring) or Docker; run `live-ingest` on their own scheduler with their own list | A week |

Not built yet and on the roadmap: webhooks when a community changes band,
per-customer accounts on the API, platforms other than Reddit (Discord,
forums) through the same ingest interface.

---

## 6. What it still lacks (and what we would build next)

1. **Ground truth for "destabilised".** Today the index is validated by
   stability across collections and on a simulation. The next step is linking
   it to outcomes a platform can observe: mass removals, bans, quarantines,
   communities going private, moderator turnover. That is also the research
   contribution: forecasting community instability against real outcomes.
2. **A better toxicity measure.** Precision 0.21 against human labels;
   calibrate per community or replace with a model trained on moderation
   decisions.
3. **A forecaster that beats the baseline.** Months of history per community
   (the live ingest is now accumulating it) and outcome labels, before
   claiming forecasting skill.
4. **Webhooks and per-customer API keys** for the integration path.
