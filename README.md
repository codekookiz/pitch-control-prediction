# ⚾ Pitch Control Success Prediction

An experimental project built for **LG Aimers 9기 — "투구 제구 성공 확률 예측 AI"**, a DACON hackathon predicting per-pitch control success on hidden 2025 KBO data, scored by **Brier Skill Score** under a strict 10-minute offline inference budget.

> **Role**
>
> Individual Project
>
> - Designed and iterated a multi-model gradient boosting stacking pipeline
> - Built causal, leakage-safe feature engineering under strict competition rules
> - Ran systematic hyperparameter, blend-composition, and calibration experiments
> - Automated local validation, submission packaging, and a full experiment log

---

# 📖 About

This project predicts the probability that a single KBO pitch lands under the pitcher's control (`control_success`), using only information available **before** the pitch — game state, count, leverage, and causal historical rates for the pitcher/batter/team.

The core challenge isn't a single model — it's building a pipeline that stays valid under the competition's rules (no use of the evaluation batch's own distribution, no future information) while squeezing out real gains through careful ensembling, calibration, and hyperparameter search, all validated against a hard 10-minute inference limit and a 5-submission-per-day budget.

Rather than chasing one big architecture change, the project evolved through ~40 tracked iterations — recency-weighted sample weighting, causal as-of-date rate features, K-fold OOF stacking, blend-weight optimization across independently trained model versions, and a disciplined local-validation methodology built specifically to avoid wasting scarce real submissions on bad ideas.

---

# 🏗️ Architecture

```text
train.csv (2019–2024, causal features only)
        │
        ▼
Feature Engineering
 (cold-start flags, count/leverage interactions,
  causal team/pitcher/batter as-of rates)
        │
        ▼
Recency-weighted sample_weight (decay=0.9)
        │
 ┌──────┼──────────┬──────────┐
 │      │           │          │
 ▼      ▼           ▼          ▼
LightGBM  XGBoost  CatBoost(d8)  CatBoost(d6)
 │      │           │          │
 └──────┴─────┬─────┴──────────┘
              ▼
   K-fold OOF Logistic Meta-learner
   (stacking, 5-fold, no leakage)
              │
              ▼
   Bias-fix (train-only season trend
   extrapolated to 2025, fixed constant)
              │
              ▼
   Multi-version Blend
   (weighted average of independently
    trained model generations)
              │
              ▼
   Local Validation
   (5-row smoke / 245K-row timing /
    4-fold chronological proxy)
              │
              ▼
   Submission Packaging (zip + requirements.txt)
```

---

# ⚙️ Tech Stack

### Gradient Boosting

- LightGBM
- XGBoost
- CatBoost

### Modeling / Stacking

- scikit-learn (`LogisticRegression`, `OrdinalEncoder`, `KFold`, `IsotonicRegression`)
- Custom K-fold OOF stacking meta-learner

### Data

- pandas
- NumPy
- joblib (compressed model artifacts)

### Language

- Python

---

# ✨ Key Features

## 🧠 4-Model Stacking Ensemble

LightGBM, XGBoost, and two CatBoost variants (depth 8 / depth 6) are trained independently and combined through a K-fold out-of-fold logistic meta-learner — never a fixed hand-tuned weight, to avoid overfitting to a single validation split.

---

## ⏱️ Causal, Leakage-Safe Feature Engineering

All historical rate features (pitcher/batter/team as-of success rates) are computed as strictly **causal, expanding-window** aggregates — each row only sees data strictly before it in time — with 2025 inference relying on train-only lookups, never the evaluation batch's own distribution.

---

## 📈 Analytically-Solved Calibration

The post-prediction bias correction (`bias_shift`) and the two-model blend weight were both shown to be **exactly quadratic** in Brier score for a fixed prediction vector. Instead of grid search, both were solved analytically by fitting a parabola to a handful of real leaderboard points — reaching the exact optimum in far fewer real submissions.

---

## 🧪 Disciplined Local Validation Methodology

With only 5 real submissions per day, a 4-fold chronological local proxy (train on early seasons, validate on each later season in turn) is used to pre-filter every new idea. Through repeated calibration against real results, a **"2-out-of-4 fold win = reliable real-world rejection"** rule was established and validated across a dozen+ independent experiments — turning a scarce resource (submissions) into a cheap one (local compute).

---

## 🧾 Full Experiment Log

Every version — accepted or rejected — is recorded in `SUBMISSIONS.md` with its hypothesis, local validation numbers, real leaderboard result, and conclusion, making the search process fully reproducible and preventing re-testing of already-closed directions.

---

# 📂 Project Structure

```text
.
├── open/
│   ├── data/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── sample_submission.csv
│   │   └── trackman_history.csv
│   ├── data_description.md
│   └── submit/
│       ├── train_final.py       # local training pipeline -> model/artifacts.pkl
│       ├── script.py            # evaluation-server inference script
│       ├── requirements.txt
│       └── SUBMISSIONS.md       # full experiment log (every version, accepted or rejected)
├── eda.ipynb
├── eda_leakage_safe.ipynb
└── README.md
```

---

# 📚 What I Learned

This project reinforced that ensemble quality depends less on any single model's tuning and more on **where diversity actually comes from** — a seed-only twin model added nothing to a blend, while a genuinely different training regime (recency weight, column subsampling) did, even when individually weaker.

It also sharpened my sense of when local validation can and can't be trusted: direction was usually reliable, magnitude rarely was, and a specific proxy shape (near-50/50 fold wins on a brand-new feature) turned out to be a consistently trustworthy rejection signal — while the same proxy, applied to blend-value rather than individual quality, was not. Distinguishing which claims a cheap local test can actually support, versus which ones need a real submission, was the most transferable lesson of the project.

Finally, working under hard competition constraints (causal-only features, a 10-minute inference budget, a 5-submission daily cap) forced a much more deliberate experimentation loop than open-ended Kaggle-style iteration — every idea had to clear a local bar before it was allowed to spend a real, scarce resource.

---

# 🚀 Future Improvements

- Player-level entity matching against `trackman_history.csv` (team-level matching was attempted but proved too unreliable to trust as a feature)
- A structurally different 2nd-level ensemble that goes beyond blending independently-trained model generations
- Broader hyperparameter search on the axes that showed the most promise (tree-complexity regularization)

---

# 🏆 Competition

- **Event:** LG Aimers 9기 — 투구 제구 성공 확률 예측 AI (DACON)
- **Metric:** Brier Skill Score, hidden 2025 KBO pitch-level data
- **Data:** 2019–2024 KBO pitch-level records (~1.47M rows), team/pitcher/batter identifiers anonymized

---

# 📊 Experiments

| Axis | Outcome |
|-----------|---------|
| Recency-weighted sample_weight (decay) | Adopted — peaked at decay=0.9, non-monotonic elsewhere |
| Multi-version blend (independently trained models) | Adopted — 4-way blend is the current best submission |
| Bias-shift / blend-weight calibration | Solved analytically (quadratic-in-weight closed form) |
| Column subsampling as a diversity axis | Adopted — small but real blend gain |
| Hand-crafted rate-derived features (entropy, trend) | Rejected — hurt real performance despite plausible hypotheses |
| Row-subsampling, `extra_trees`, `num_leaves`, `learning_rate`, `dart` | Rejected / mostly closed after systematic local screening |
| Isotonic / monotonic-constraint calibration | Rejected — meta-learner already near-optimally calibrated |
| Neural network (standalone and as a blend member) | Rejected — GBM consistently superior on this tabular task |
| Context-aware (gating) meta-learner | Rejected — base predictions too correlated for nonlinear gains |
