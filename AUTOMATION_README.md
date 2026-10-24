# Car4U — Automation & Testing Additions

This adds testing, CI, scheduled retraining, and containerization to the existing project.
Drop these files into your project root (they assume the layout below — adjust paths in the
workflow/Dockerfile if your real layout differs):

```
project_root/
├── app/
│   └── app.py
├── src/
│   ├── price_prediction.py
│   ├── recommendation_system.py
│   ├── train_models.py
│   └── train_predictor.py
├── data/
│   └── used_cars_v2.csv
├── models/
│   ├── car_model.pkl
│   └── price_predictor_model.pkl
├── tests/                        <-- NEW
├── .github/workflows/            <-- NEW
├── requirements.txt              <-- NEW
├── pytest.ini                    <-- NEW
├── Dockerfile                    <-- NEW
├── docker-compose.yml            <-- NEW
└── .dockerignore                 <-- NEW
```

## 1. Tests (`tests/`)

Run locally:
```bash
pip install -r requirements.txt
pytest tests/ -v
```

Covers: model training/loading, prediction sanity (positive prices, newer-car-costs-more
regression check), recommendation filtering (budget/fuel/seats/km), empty-result handling,
save/load round-trips, and the `_obj_to_num` string parser edge cases.

## 2. CI (`.github/workflows/ci.yml`)

Runs automatically on every push/PR to `main`/`develop`: installs deps, lints with `ruff`,
runs the full test suite with coverage, and checks `app.py` parses cleanly.

## 3. Scheduled retraining (`.github/workflows/retrain.yml`)

Triggers when:
- `data/used_cars_v2.csv` changes on `main`, **or**
- every Monday at 03:00 UTC (freshness safety net), **or**
- manually, from the Actions tab (`workflow_dispatch`)

It retrains both models, **validates them** (sane price range, non-empty recommendations)
before touching your repo, and only commits the updated `.pkl` files if validation passes.
If validation fails, the workflow fails loudly instead of shipping a broken model.

> ⚠️ This workflow needs `contents: write` permission for the default `GITHUB_TOKEN`
> (Settings → Actions → General → Workflow permissions → "Read and write") so it can push
> the retrained model files back to the repo.

## 4. Docker (`Dockerfile`, `docker-compose.yml`)

```bash
docker compose up --build        # builds + trains models (if missing) + runs the app on :8501
docker compose run --rm retrain  # retrain without rebuilding the app image
```

## 5. The autonomous pipeline (Marketplace data → model)

The project now has a full data-to-model loop, with a safety gate so a retrain never
silently deploys a worse model:

1. **Real users list real cars** through the Marketplace ("Sell Your Car" page). Each
   listing is stored in `data/car4u.db`, tied to a real account.
2. **You export new listings into the training set** by running:
   ```bash
   python src/export_listings.py
   ```
   This pulls every not-yet-exported listing out of the database and appends it to
   `data/used_cars_v2.csv` in the same schema the models already expect. It's
   idempotent — running it twice won't duplicate rows. Commit and push the updated CSV.
3. **GitHub notices the CSV changed** and automatically triggers `retrain.yml` (no
   manual step needed once you've pushed).
4. **The price predictor now measures its own honest accuracy.** `PricePredictor`
   holds out 20% of the data it never trains on, and reports MAE (mean error, in
   Lakhs) and R² on that held-out set — saved to `models/price_predictor_metrics.json`.
5. **The workflow compares new vs. old metrics before deploying.**
   `src/compare_metrics.py` checks whether the freshly trained model's MAE is at
   least as good as (within 2% tolerance of) the currently deployed one. If it's
   worse, the workflow automatically reverts to the previous model instead of
   shipping a regression — you'll see this clearly in the Actions log either way.

**Why the database itself isn't in this pipeline directly:** `car4u.db` lives on
whichever machine runs the Streamlit app (your laptop, or wherever you deploy it),
and contains user password hashes — it shouldn't be committed to a public GitHub
repo. Step 2 above is the intentional bridge: you (or a scheduled task on wherever
the app is deployed) export the *car data only* into the CSV and push that, and
everything downstream (steps 3–5) happens automatically in the cloud.


`requirements.txt` uses **minimum-version floors** (`>=x,<y`) instead of exact pins
(`==x`) for `streamlit`, `pandas`, `numpy`, and `scikit-learn`. This is deliberate:

- New Python releases (3.13, 3.14, ...) only get pre-built wheels from these packages
  starting at specific versions (e.g. pandas needed 2.2.3 for Python 3.13 wheels and 2.3.3
  for Python 3.14 wheels; scikit-learn needed 1.7.2 for Python 3.14 wheels). If you pin an
  exact older version, pip falls back to compiling from source — which fails without a C
  compiler installed (Visual Studio Build Tools on Windows).
- Using floors lets pip always pick the latest compatible release for whatever Python
  you're running, avoiding this problem entirely — on your machine, in CI (Python 3.11),
  and in Docker (Python 3.11) alike.
- The upper bound (`<3.0` for pandas, `<2.0` for scikit-learn) avoids known breaking changes
  in the next major version.

**After installing, retrain your models once** so the `.pkl` files match the pandas version
that actually got installed:
```bash
python src/train_models.py
python src/train_predictor.py
```
This avoids any mismatch between the pandas version that saved the old `.pkl` files and the
one now loading them. Re-run `pytest tests/ -v` afterward to confirm everything still works.

