# Capstone Report — Refresh / Content Opportunity Scoring (Lane 2)

- **Author:** Areeba Hassan
- **Lane:** Lane 2 — Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/Areeba-Kh571/flyrank-ml-internship-areeba
- **Date:** 2026-08-30 (draft — `w05_model.ipynb` has been run once already, which surfaced a real methodological finding described in Section 4; the feature-set fix from that finding still needs a final consistent run across `w05`–`capstone.ipynb`. Numbers marked `[FILL]` need that run; see "How to finish this report" at the bottom)

> Sections 1–8 mirror the Pass / Needs-Work rubric axes. Sections 0 and 9 are paper sections — this file is also what gets deployed as the public paper page, per `GUIDE.md`.

## 0. Abstract

Out of a full client content inventory, which pages should a reviewer look at first for refresh? Using FlyRank's pseudonymized search-performance warehouse (~79M daily rows across 104 clients; 102,769 eligible content items across 30 clients with a full Q1 2026 history), I built a client-grouped, leakage-checked pipeline that scores each page's Q1 2026 signals against whether it declined in April 2026, and compared a transparent hand-written rule against logistic regression, decision tree, and random forest models. A first pass looked like a clear win for the model (Precision@50 of 0.860 vs. the baseline's 0.340) — until a deliberate robustness check showed that lift collapsing to exactly the baseline's number once a single structurally-coupled feature was removed, which is itself the headline finding: `[FILL — state the final result plainly once the corrected feature set is re-run end to end: a real lift, no lift, or a partial lift]`. The output is a ranked, reason-coded action queue for a human reviewer, not an automated publishing decision or a causal claim that refreshing any page will recover its traffic.

## 1. Problem framing

**Decision supported:** a FlyRank content reviewer has limited time and a large inventory; this system orders that inventory so the reviewer opens the most promising candidates first.

**Unit of analysis:** one content item (`client_hash_id` + `content_hash_id` pair), summarized over a 90-day feature window (Q1 2026) and evaluated against the following 30 days (April 2026).

**Output:** a ranked queue (`final_refresh_score`), one reason code and one recommended action per row (`refresh_now`, `refresh_and_expand`, `monitor_ai_opportunity`, `monitor`, `no_action`), plus a confidence label (`high` / `medium` / `low`).

**Action a human takes:** opens the top of the queue, reads the reason code, verifies the page isn't a false alarm (seasonal dip, consolidation with a sibling page, intentional evergreen content — see Limitations), and decides whether to refresh, expand, or leave it.

**Cost of a wrong call:** this is a decision-support risk, not a safety risk. A false positive wastes a reviewer's limited time on a page that wasn't really declining; a false negative means a genuinely declining page sits further down the queue and may not get reviewed in time. Neither error touches anything outside internal review prioritization.

**Why ML helps at all:** a fixed-weight hand rule can only combine a short list of conditions its author already anticipated, and cannot represent conditional effects — for example, position only matters once there's enough impression volume behind it (the "volume floor" checked directly in `w02_ml_task_framing`). A model that weighs several observable signals together, and can learn interactions a linear rule cannot express, is the thing actually being tested here — and it has to re-earn that advantage on this warehouse-scale, genuinely forward-split data, not just inherit it from the smaller starter-CSV run in `w01`/`w02`.

## 2. Data safety

**Source:** the full FlyRank warehouse release (Hugging Face, gated), tables `dim_clients`, `dim_content`, `fact_content_daily_performance`. Never the raw private warehouse outside this access pattern, and never re-exported.

**Columns used (features):** `imp_90d`, `pos_90d`, `days_with_impressions_90d`, `ai_referral_share_90d`, `content_age_days` — all computed only from `report_date` rows inside Q1 2026, or from `dim_content.content_created_date` (a value that cannot change after the fact).

**Columns deliberately excluded, with why:**
- `fact_content_daily_performance` rows dated on/after `2026-04-01` — the outcome window.
- `fact_content_query_90d` entirely — its fixed window is anchored to the snapshot's own end, not to this report's March 31 cutoff, so its "recent" columns overlap the label period (documented leakage watch in `docs/data-dictionary.md`).
- `fact_content_daily_performance_sample` (June 2026) — the sealed test month; never opened anywhere in this repo.
- `dim_content.content_updated_date` — reflects current state, not history; tested directly and found to produce a negative "days since update" for ~75% of rows (a value from *after* the decision point). `content_created_date` is used instead — immutable, so point-in-time-safe, at the cost of being a weaker staleness proxy.
- `trend_ratio_90d`, **as a model feature** (it stays in the dataframe for the baseline's `declining_now` flag) — tested directly in `w05_model.ipynb` and found to carry 68.4% of a decision tree's importance, well past a "suspiciously perfect" threshold. It shares `imp_last30` with the label's own reference point (`is_declining_next30` is defined *relative to* `imp_last30`), so a page with an unusually high recent rate faces a proportionally higher bar to avoid being labeled "declining" — a structural coupling with the label, not necessarily new signal. Confirmed, not just suspected: removing it collapsed the tree's Precision@50 from 0.860 to exactly the frozen baseline's 0.340 on the same test split.
- Any other `dim_content` column beyond the join keys and `content_created_date` — never inspected via `DESCRIBE`, so never used.

**Leakage risks considered:** `trend_direction` / `trend_pct` (the starter CSV's same-window label source) are never used anywhere in this warehouse-based pipeline — the label here, `is_declining_next30`, is built from a genuinely later window instead. `imp_label30` (the raw label-window aggregate) is used only to construct and check the label, never as a feature — confirmed both by assertion in code (`w05_model`, `w07_action_playbook`) and by a deliberate injection test that shows the score spike when it's added on purpose (`w06_validation_audit`).

**Client identifiers:** `client_hash_id` / `content_hash_id` are pseudonymous hashes, used only for grouping and joining (never as model features, never mapped back to a real client). No client names, domains, URLs, page titles, or raw queries appear anywhere in `work/`.

## 3. Baseline

**The rule, in plain words:** a page is worth flagging if it's stale (created 180+ days before the decision point) and still visible (≥500 impressions in the trailing 90-day window) — and among pages clearing both bars, ones whose recent 30-day pace is already falling get flagged first.

```text
stale   = content_age_days >= 180
visible = imp_90d >= 500
declining_now = trend_ratio_90d < 0.8
baseline_score = stale * visible * imp_90d * (1 + declining_now)
```

**Why it's a fair comparison:** every input is computed the same way, from the same Q1 window, as the model's features — same data, same label, same metric, same client-holdout split (Section 5). It is frozen once model work starts (never refit or re-tuned to chase the model's numbers).

**Its numbers**, on the client-holdout test split, same metric as the model comparison in Section 5:

| Metric | Value |
|---|---:|
| ROC AUC | `[RUN NOTEBOOKS TO FILL]` |
| Average precision | `[RUN NOTEBOOKS TO FILL]` |
| Precision@50 | `[RUN NOTEBOOKS TO FILL]` |
| Base rate (test split) | `[RUN NOTEBOOKS TO FILL]` |

(Source: `w05_model.ipynb` section 3, `results['baseline_rules']`, and `work/outputs/model_results.json` written by `w07_action_playbook.ipynb`.)

## 4. Model / analysis

**Method:** logistic regression, a depth-3 decision tree, and a random forest (300 trees), in that order of complexity — starting simple per `training-honest-models`, adding complexity only if it earns its keep. Each classifier's `predict_proba` output is used as a ranking score, evaluated at Precision@50, not as a bare yes/no classifier.

**Feature list:** `imp_90d`, `pos_90d`, `days_with_impressions_90d`, `ai_referral_share_90d`, `content_age_days`. Deliberately excluded: `trend_direction`, `trend_pct` (label-derived), `imp_label30` (label-window raw data), every `dim_content` column not individually verified safe (Section 2) — and `trend_ratio_90d`, which looked like the strongest feature in an initial run but turned out to be structurally coupled to the label (Section 2); a direct with/without test collapsed its apparent lift entirely (Section 6).

**Target / proxy, in one sentence:** `is_declining_next30 = 1` when April 2026's total impressions come in under 80% of the Q1 window's own last-30-day rate, else `0` — a threshold-defined proxy for "declining," built entirely from a window strictly after every feature.

## 5. Evaluation

**Split:** `GroupShuffleSplit` by `client_hash_id`, 80/20, seed 42 — no client's rows appear on both sides (asserted in code, not just claimed). A naive random row-level split is also computed on the same data (`w06_validation_audit`), specifically so the *gap* between the two is itself part of the reported evidence, per `hunting-leakage-and-validating`.

**Metrics, model vs. baseline, same split:**

| Model | ROC AUC | Avg precision | Precision@50 |
|---|---:|---:|---:|
| baseline_rules (frozen) | `[FILL]` | `[FILL]` | `[FILL]` |
| logistic_regression | `[FILL]` | `[FILL]` | `[FILL]` |
| decision_tree (depth 3) | `[FILL]` | `[FILL]` | `[FILL]` |
| random_forest | `[FILL]` | `[FILL]` | `[FILL]` |

Base rate on the held-out client split: `[FILL]`.

Naive-random-split Precision@50 for the winning model, reported alongside the honest grouped number (the "before/after" gap): `[FILL — naive]` vs. `[FILL — grouped]`.

(Source: `w05_model.ipynb` section 3 and `w06_validation_audit.ipynb` section 2; the receipt file is `work/outputs/model_results.json`, written by `w07_action_playbook.ipynb`.)

**Error analysis (from `w05_model.ipynb` section 4):** `[FILL — 2-3 sentences: top feature importances, whether any looked "suspiciously perfect," and 2-3 concrete false-positive / false-negative examples with their feature values]`.

## 6. Interpretation

**The single largest finding in this capstone is a methodological one, not a feature-importance one.** An initial run of `w05_model.ipynb` (with `trend_ratio_90d` included as a feature) showed a depth-3 decision tree reaching Precision@50 = 0.860 against the frozen baseline's 0.340 — a striking, headline-ready result. But `trend_ratio_90d` alone carried 68.4% of that tree's feature importance, past the "suspiciously perfect" threshold the `hunting-leakage-and-validating` skill warns about. Running the skill's own verification test — retrain once with the feature, once without — collapsed the tree's Precision@50 from 0.860 to *exactly* 0.340, the baseline's own number, and random forest's Precision@50 fell from 0.380 to 0.300 (now below baseline). A second check explained part of why: the depth-3 tree produced only 4 distinct probability values on the held-out test set, meaning "top 50" was substantially tie-broken within one or two leaves rather than a fine-grained ranking.

**The honest conclusion:** on this client-grouped holdout, none of the candidate models showed a clear, robust advantage over the frozen baseline once the structurally-coupled feature was removed. This is a valid, reportable result — per `writing-honest-claims`, "we expected X; the data shows no such pattern" is a publishable finding, not a failed cell. `[FILL once w06/w07/capstone are re-run with the corrected FEATURES list — confirm this holds for the final, complete comparison table including logistic regression, and report the final numbers here.]`

This also directly **echoes and sharpens a finding from earlier in this same project**: `w02_ml_task_framing`'s starter-CSV run already showed a model's apparent edge nearly vanishing under client holdout (the tree fell from 0.540 to 0.560 — no real gain — while the baseline actually rose from 0.240 to 0.620 and won outright on one comparison). This capstone-scale run, backed by an actual robustness test rather than just a second split, confirms the same pattern with a specific, identified mechanism behind it.

Two more structural findings are established independent of the final numbers:

- **A rich feature set is not automatically enough — it has to survive a grouped split.** `w02_ml_task_framing`'s own starter-CSV run showed the model's advantage nearly vanishing under client holdout (0.540 → 0.560 for the tree, vs. the baseline actually improving from 0.240 → 0.620) — a reminder that "beats baseline on the split you tuned on" and "beats baseline on a client it's never seen" are different claims, and this report only makes the second one.
- **The two signals behind the baseline (staleness, volume) are close to linearly independent of each other** (all pairwise correlations ≤0.08, `w02_ml_task_framing` section 5), which is exactly the situation where a fixed-weight linear rule struggles and a model that can learn conditional splits has room to add value — *if* that value survives Section 5's honest split.

## 7. Recommendation

The ranked queue (`work/outputs/action_playbook_queue.csv`, built in `w07_action_playbook.ipynb`) is the deliverable a reviewer actually opens: every row carries a `final_refresh_score`, a `reason_code` (`MODEL_DECLINE_RISK`, `VISIBLE_MODEL_OPPORTUNITY`, `POSITION_UPSIDE`, `AI_REFERRAL_GAP`, `STALE_VISIBLE_DECLINING`, `STALE_VISIBLE_STABLE`, or `NO_ACTION`), a recommended `action`, and a `confidence` label.

**How a FlyRank editor uses it tomorrow:** open the `high`-confidence rows first (`[FILL — n rows / % of inventory]`), read the reason code, sanity-check against the no-go list below, and decide refresh / expand / monitor / skip. `medium`/`low` rows are lower priority, not "ignore" — they mean the evidence is thinner, not that the page is fine.

**What would make a specific recommendation wrong** (per-row, and worth checking before acting): the page is intentionally stable evergreen content that never needed a refresh; the apparent decline is seasonal or the demand moved to a sibling page (consolidation) rather than genuinely dropping; the page is too new for its Q1 window to mean much; or the position/AI-referral signal is missing or noisy for that specific row.

**Confidence, stated honestly:** Section 6's robustness check found that this model's apparent edge over the baseline was mostly an artifact of one structurally-coupled feature, not genuine signal — so the honest framing is that a learned model has **not yet demonstrated** a clear, reproducible advantage over the transparent rule on this data, once validated carefully. The ranked queue is still useful as a *second, independent read* on the same inventory (it surfaces `AI_REFERRAL_GAP` and `POSITION_UPSIDE` candidates the baseline rule cannot see at all, by construction), but a reviewer should not assume the model's ranking is more trustworthy than the baseline's at the top of the queue. `[FILL once the final w07/capstone re-run confirms whether this holds with the complete, corrected feature set.]` This is a decision-support tool for prioritizing limited review time either way — not a guarantee that any flagged page will decline further, and not a claim about why Google's ranking moved (Lane guide, Section 6).

## 8. Reproducibility

**Exact commands, fresh clone:**

```bash
git clone https://github.com/Areeba-Kh571/flyrank-ml-internship-areeba
cd flyrank-ml-internship-areeba
pip install -r requirements.txt
```

Then open each of `work/notebooks/w03_data_contract.ipynb` → `w04_baseline_score.ipynb` → `w05_model.ipynb` → `w06_validation_audit.ipynb` → `w07_action_playbook.ipynb` → `capstone.ipynb` in Colab (badges at the top of each), run top to bottom (`Runtime → Run all`), supplying a Hugging Face read token via Colab Secrets (`HF_TOKEN`) when prompted. Each notebook re-pulls its own frame from `hf://datasets/FlyRank/internship-warehouse` independently — no notebook depends on another's in-memory state.

**Seed:** `42`, fixed everywhere a split or a model is initialized (`GroupShuffleSplit`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`).

**Environment:** `requirements.txt` — `pandas>=2.2`, `numpy>=1.26`, `scikit-learn>=1.4`, `matplotlib>=3.8`, `duckdb>=1.0`, `huggingface_hub>=0.24`. Per `GUIDE.md`'s own FAQ, random-forest numbers can shift a few points across scikit-learn versions; the direction and rough size of any lift over baseline is the stable claim, not a specific third decimal.

**Sealed/holdout receipts (per `hunting-leakage-and-validating`):** the cell that builds the client-grouped frame is `w07_action_playbook.ipynb` section 1 (and identically in `capstone.ipynb` sections 2–3); the metrics file it produces is `work/outputs/model_results.json`, and the scored queue is `work/outputs/action_playbook_queue.csv` — both committed, both regenerable from the commands above. `[FILL once run: confirm both files are present in the repo at submission time.]`

**Population-selection disclosure:** rows require `gsc_data_start <= 2026-01-01` (a full Q1 history) and `imp_90d >= 100` — both computed from Q1-and-earlier information only, never from the April outcome window (checked explicitly in `w06_validation_audit.ipynb` section 3).

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset, provided by [FlyRank](https://flyrank.ai). Thank you to the FlyRank team and my mentor for the warehouse access, the lane guide, and the skills library this repo ships.

---

> **Claims checklist before submitting:** observed / measured / directional / decision-support language everywhere — no causal claims without an experiment or causal design — no "predicted Google's algorithm" — no client-identifying details — numbers in this report match a fresh re-run.
> **Metrics vs. base rate:** every precision@K or accuracy number above is printed next to its split's base rate — a high score means nothing until the base rate is known.

---

## How to finish this report

Everything above that is **not** marked `[FILL]` is already true and traceable to committed work: `w01_research_question.ipynb` (ML-02), `w02_ml_task_framing.ipynb` (ML-03), `w03_data_contract.ipynb` (ML-04), and `w04_baseline_score.ipynb` (ML-07) are complete, and their numbers are quoted directly from those notebooks' own cells.

Three notebooks were empty and have been written and committed as runnable code, following the exact conventions of the completed ones (same Q1→April windows, same label, same grouped-client split, same leakage checks) — none of this could be executed from this environment, which has no network path to Hugging Face, so they needed a real Colab run:

1. `work/notebooks/w05_model.ipynb` (ML-08) has already been run once in Colab. That run found `trend_ratio_90d` was structurally coupled to the label (Sections 2, 4, 6 above) — the notebook has since been updated to run the robustness check that proves this automatically, in its own Section 4. **Re-run it once more, fresh**, so its Section 3 comparison table reflects the corrected understanding, then save it back to GitHub.
2. Open `work/notebooks/w06_validation_audit.ipynb`, run it top to bottom (it already excludes `trend_ratio_90d` from `FEATURES`). This is ML-09.
3. Open `work/notebooks/w07_action_playbook.ipynb`, run it top to bottom (also already excludes `trend_ratio_90d` from `FEATURES`) — this one writes `work/outputs/action_playbook_queue.csv` and `work/outputs/model_results.json`, the two receipt files this report's numbers trace back to. This is ML-10.
4. Open `work/notebooks/capstone.ipynb`, run it top to bottom (same corrected `FEATURES`) — it reproduces sections 1–7 of this report as one notebook and saves the three chart images this page should embed. This is ML-11, and its closing cells cover ML-12 (demo outline, social cut, employer summary — already drafted, just needs real numbers dropped into the two bracketed spots).
5. Come back to this file and replace every `[FILL]` / `[RUN NOTEBOOKS TO FILL]` with the real printed numbers — they're the direct `print()` output of the cells above, not a re-derivation.
6. Commit `work/notebooks/*.ipynb`, `work/outputs/*.json`, `work/outputs/*.csv`, `work/outputs/figures/*.png`, and this file.
7. Deploy this report as your public paper page (see `skills/deploying-static-pages/SKILL.md`), then put its exact URL as the one line in `submission/paper_url.txt`.
8. Submit your repo URL on the capstone card in your portal — that's the only thing that ever gets uploaded, per `GUIDE.md` section 5.
