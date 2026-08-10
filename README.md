# LoadWatch

**Workload-based soft-tissue injury *risk scoring* for NBA players** — not a binary injury predictor.

LoadWatch is a full-stack sports analytics app that estimates a **relative risk score (0–100)** from minutes, schedule density, rest, age, and injury recency. Coaches and front-office users can open a Streamlit dashboard, pick any rostered player, and see season risk trends, the top factors driving the latest score, and a recent workload calendar.

---

## Problem statement

NBA soft-tissue injuries (strains, sprains, tendinopathies) are associated with accumulated and concentrated workload — dense minutes, back-to-backs, short rest — and with prior injury history. Binary “will he get hurt?” models overclaim certainty and fail under severe class imbalance. LoadWatch instead ranks player-games by **estimated relative soft-tissue risk**, so staff can prioritize recovery, minutes management, and monitoring without treating the score as a medical diagnosis.

---

## Data sources

| Source | What we use | Link |
|--------|-------------|------|
| **NBA Stats API** via [`nba_api`](https://github.com/swar/nba_api) | Per-player game logs (minutes, matchup, home/away), team rosters (age), optional tracking SpeedDistance | [stats.nba.com](https://www.nba.com/stats) |
| **Official NBA injury reports** via [`nba-injury-report`](https://github.com/mkeenan195/nba-injury-report) | Pre-game Out/Doubtful designations + reason text; collapsed to onset spells | [ak-static.cms.nba.com injury PDFs](https://ak-static.cms.nba.com/referee/injury/) |
| **Prosportstransactions** (fallback) | Historical IL notes if NBA PDFs unavailable | [prosportstransactions.com](https://www.prosportstransactions.com/basketball/) |
| Optional local CSV | Drop-in injury log with `player`, `date`, `injury_type` columns | e.g. Kaggle “NBA Injury Stats” style dumps |

Player names from Prosports are fuzzy-matched to `nba_api` player IDs (`thefuzz` + `difflib`). Soft-tissue events are filtered by keyword (strain, sprain, hamstring, calf, …) with contact/illness exclusions.

**Default modeling season:** `2023-24` (full regular season). The pipeline accepts `--season 2024-25` as well.

---

## Methodology

### Features (per player-game, known *entering* the game)

- `minutes_last_7d` / `minutes_last_14d` — rolling prior minutes  
- `games_last_7d` — schedule density  
- `is_back_to_back`, `rest_days` — recovery gap  
- `season_minutes_cumulative` — season workload stock  
- `age` — from team rosters  
- `days_since_last_injury` — recurrence signal  
- `distance_last_7d` — included **only if** tracking data is available; otherwise omitted (documented limitation)

### Labels

`label = 1` if a soft-tissue injury is logged within the **next 7 days** after the game; else `0`. Expect a heavily imbalanced set (~95%+ negatives).

### Models

1. **Baseline:** logistic regression with `class_weight='balanced'` (coefficients saved for interpretability).  
2. **Main model:** XGBoost with `scale_pos_weight` for imbalance.  
3. **Validation:** **chronological** train/test split (earlier games → train, later → test). Random k-fold is avoided to prevent future leakage.  
4. **Metrics:** ROC-AUC and **PR-AUC** (preferred under imbalance). Accuracy is not treated as a primary metric.  
5. **Risk score:** predicted probability → **percentile rank 0–100** within the scored season. Bands: Low &lt; 60, Moderate 60–85, High ≥ 85.  
6. **Explainability:** SHAP `TreeExplainer` on XGBoost; top factors shown in the dashboard.

---

## Key findings / insights

From the trained **2023-24** pipeline (chronological holdout starting 2024-03-05):

| Model | ROC-AUC | PR-AUC | Notes |
|-------|---------|--------|-------|
| Logistic regression (balanced) | **0.618** | **0.066** | Stronger on this time split |
| XGBoost (`scale_pos_weight`) | 0.539 | 0.054 | Used for SHAP / dashboard scores |

Baseline positive rate ≈ **3.8%** of player-games (7-day soft-tissue horizon) — PR-AUC is the more honest ranking metric than accuracy.

**Feature readouts**
- Logistic coefficients: higher **minutes in the last 7 days** raise estimated risk the most; more **days since last soft-tissue injury** and higher **games_last_7d** (after controlling for minutes) move the other way — denser short weeks with *lower* per-game minutes look different from heavy-minute bursts.
- Global mean \|SHAP\| on XGBoost highlights **season minutes**, **injury recency**, **age**, and **7-day minutes / distance** as top drivers.

**Narrative examples** (risk percentile in the week before a soft-tissue IL note):
- **Anthony Davis (LAL)** — elevated risk ranks ahead of bilateral Achilles tendinopathy designations (Feb–Mar 2024).
- **Tyrese Haliburton (IND)** — high risk rank preceding a left hamstring strain / injury-management stretch (Jan 2024).
- **Giannis Antetokounmpo (MIL)** — elevated ranks near Achilles tendinitis listing (Mar 2024).
- **Ivica Zubac (LAC)** — high risk ahead of a right calf strain (Feb 2024).

These are **descriptive alignments**, not proof the model “predicted” the injury. Full tables: `models/metrics.json`, `models/logistic_coefficients.csv`, `models/shap_global_importance.csv`.

---

## Dashboard screenshots

After launching the app, capture:

1. Player risk trend with injury markers  
2. Current risk gauge + SHAP factor bars  
3. Team risk overview ranking  

Save images under `docs/` (optional) and link them here for submissions.

---

## Limitations

- **Small positive class:** soft-tissue labels from public IL notes are incomplete and noisy; many “out” designations lack precise onset timing.  
- **No biomechanics / GPS / force-plate data** — only box-score and schedule proxies for load.  
- **Correlation ≠ causation** — scores reflect historical associations, not causal effects of minutes.  
- **Tracking distance** may be unavailable or only season-level; the distance feature is dropped when the endpoint fails.  
- **Name matching** between Prosports and NBA IDs is imperfect for suffix/II/III and trade mid-season edge cases.  
- **Not medical advice** — do not use as clearance or diagnosis.

---

## Project layout

```
Load_Watch/
├── README.md
├── LICENSE
├── requirements.txt          # dashboard (Cloud)
├── requirements-pipeline.txt # training pipeline
├── data/raw/            # API + scrape caches (parquet)
├── data/processed/      # features, injuries, scored tables
├── src/
│   ├── data_collection.py
│   ├── injury_data.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── explain.py
│   └── utils.py
├── scripts/run_pipeline.py
├── notebooks/eda.ipynb
├── dashboard/app.py
└── models/              # joblib / xgb json / metrics
```

---

## How to run locally

**Requires Python 3.11+** (developed on 3.13).

```bash
cd D:\Load_Watch
python -m venv .venv

# Windows
.\.venv\Scripts\activate

pip install -r requirements-pipeline.txt

# Build data, train models, compute SHAP (first run downloads/caches NBA + injury data)
python scripts/run_pipeline.py --season 2023-24

# Dashboard
streamlit run dashboard/app.py
```

Re-runs reuse `data/raw/*.parquet` caches. Force a feature rebuild with `--rebuild-features`. Skip SHAP with `--skip-shap` for a faster baseline smoke test.

---

## Deploy on Streamlit Community Cloud (public demo)

The dashboard reads **pre-scored** artifacts at runtime — it does **not** call the NBA API or retrain models. For a public demo you must commit the processed parquet files (not the full pipeline outputs).

### What the app needs in the repo

| Path | In git today? | Purpose |
|------|---------------|---------|
| `data/processed/scored_2023-24.parquet` | **No** (gitignored until you add it) | Player-game risk scores + SHAP factors |
| `data/processed/injuries_clean_2023-24.parquet` | **No** | Injury markers on charts |
| `models/metrics.json` | Yes | Model transparency table |
| `dashboard/app.py`, `src/` | Yes | App code |

Parquet is preferred (~6 MB total). CSV fallbacks work but `scored_2023-24.csv` is ~35 MB.

### One-time: commit demo data and push

From the project root (after pulling these config changes):

```bash
git add .gitignore .streamlit/config.toml .python-version requirements-cloud.txt src/explain.py README.md
git add data/processed/scored_2023-24.parquet data/processed/injuries_clean_2023-24.parquet
git commit -m "Add Streamlit Cloud demo artifacts and deploy config"
git push origin master
```

If the parquet files are missing locally, run `python scripts/run_pipeline.py --season 2023-24` first.

### Streamlit Cloud UI steps

1. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **Create app** → **From existing repo**.
3. **Repository:** `Trie-hard/Load_Watch`
4. **Branch:** `master`
5. **Main file path:** `dashboard/app.py`
6. **App URL (slug):** e.g. `loadwatch` → public URL `https://loadwatch.streamlit.app`
7. **Advanced settings → Python version:** `3.11` (matches `.python-version`)
8. **Dependencies:** `requirements.txt` (dashboard-only; fast install on Cloud). Use `requirements-pipeline.txt` for local training.
9. Click **Deploy**. First build is usually under a minute with the slim deps.

### “Try it out” link for Devpost

After deploy, use:

`https://<your-app-slug>.streamlit.app`

Example: `https://loadwatch.streamlit.app`

### Troubleshooting

- **“No scored data found”** — parquet files are not on the branch Streamlit built; commit and push them.
- **"Error installing requirements"** — Cloud uses slim `requirements.txt`; install `requirements-pipeline.txt` only for local training (`xgboost`, `shap`, `nba-injury-report`, etc.).
- **Import errors for `shap` / `xgboost`** — dashboard uses lazy imports in `explain.py`; those packages are not needed at runtime on Cloud.
- **Repo size** — do not commit `data/raw/`, `models/*.joblib`, or `scored_2023-24.csv`; parquet is enough.

### Alternatives if Streamlit Cloud is blocked

| Platform | Notes |
|----------|--------|
| [Hugging Face Spaces](https://huggingface.co/spaces) | Streamlit SDK; add `README.md` with `sdk: streamlit` and same `requirements.txt` |
| [Railway](https://railway.app) | `streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0` |
| [Render](https://render.com) | Free web service; same start command as Railway |

---

## Actionable impact

1. **Flag High-band players before a back-to-back** and discuss minutes caps or extra recovery with performance staff.  
2. **Compare risk ranks across the roster** (Team overview) to allocate load-management slots when multiple players spike in the same week.  
3. **Use SHAP drivers in return-to-play meetings** — e.g. if “minutes last 7 days” dominates, taper denser usage before blaming age alone.

---

## License

MIT — see [LICENSE](LICENSE).
