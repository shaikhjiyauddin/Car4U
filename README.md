<p align="center">
  <img src="https://img.shields.io/badge/Car4U-Intelligent%20Car%20Platform-blue?style=for-the-badge&logo=streamlit" alt="Car4U">
</p>

<p align="center">
  <strong>AIML Group no. 6</strong><br>
  Intelligent used-car recommendation, price prediction and peer-to-peer marketplace — powered by ML, built with Streamlit.
</p>

<p align="center">
  <a href="#features">Features</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#run-locally">Quick start</a> ·
  <a href="#autonomous-pipeline">Autonomous pipeline</a> ·
  <a href="#verification">Verification</a>
</p>

---

## The idea

Finding the right used car should not mean bouncing between listing sites, guessing at fair prices, and hoping the seller picks up the phone.

**Car4U brings the entire workflow into one place:** browse 7,254 real Indian-market cars, get AI-driven recommendations tailored to your preferences, predict fair prices with a trained regression model, compare cars side-by-side, list your own car for sale, and chat directly with sellers — no phone numbers exchanged, no guesswork.

**The AI models learn from real data; the marketplace generates new data that feeds back into the models.** This is a closed-loop autonomous pipeline: sellers list cars → listings are exported into the training set → models retrain automatically → predictions improve over time.

## Features

- **Smart Recommendations:** Content-based filtering on budget, fuel type, transmission, location, owner type, seats, safety rating, and kilometers driven — with cosine-similarity "find similar cars" for every result.
- **Price Prediction:** Linear regression model trained on real transactions, with honest held-out MAE and R² metrics after every retrain.
- **Live Marketplace:** Users sign up, list cars with photos, and chat directly with buyers — all in-app, no phone numbers published.
- **Side-by-Side Comparison:** Select up to 3 cars and compare every spec in a single table.
- **Searchable Database:** Browse, search, and filter the full 7,254-car dataset by location, fuel type, transmission, and owner type.
- **Rich Analytics:** Interactive Plotly charts — price distributions, brand comparisons, fuel-type breakdowns, power-vs-price scatters, correlation heatmaps, and live marketplace growth timelines.
- **Autonomous Retraining Pipeline:** GitHub Actions automatically retrains models when the dataset changes, validates them, and only deploys if the new model is at least as good as the old one.
- **Containerized Deployment:** Docker and docker-compose for one-command builds and runs.

## How it works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER INTERACTION                                │
│                                                                         │
│   Browse / Search ──► Get Recommendations ──► Predict Price             │
│          │                    │                     │                    │
│          ▼                    ▼                     ▼                    │
│   Compare Cars         Chat with Seller      List Your Car              │
│                              │                     │                    │
└──────────────────────────────┼─────────────────────┼────────────────────┘
                               │                     │
                               ▼                     ▼
                        ┌─────────────┐     ┌──────────────┐
                        │  SQLite DB  │◄────│ User Listing │
                        │  (car4u.db) │     │  with Photo  │
                        └──────┬──────┘     └──────────────┘
                               │
                    export_listings.py
                               │
                               ▼
                    ┌──────────────────┐
                    │ used_cars_v2.csv │  (training data)
                    └────────┬─────────┘
                             │  git push
                             ▼
                    ┌──────────────────────────────────────┐
                    │    GitHub Actions — retrain.yml       │
                    │                                      │
                    │  Train models ──► Validate ──► Gate  │
                    │                                 │    │
                    │                    ┌─────────────┤    │
                    │                    │ Improved?   │    │
                    │                  Yes            No    │
                    │                    │             │    │
                    │               Commit .pkl   Revert   │
                    └──────────────────────────────────────┘
```

**Built with:** Python · Streamlit · scikit-learn · pandas · NumPy · Plotly · Pillow · SQLite · Docker · GitHub Actions.

The recommendation engine uses content-based filtering with cosine similarity on standardized features. The price predictor is a linear regression model with one-hot-encoded categoricals. The marketplace backend is a dependency-free SQLite layer with salted PBKDF2 password hashing. See the [autonomous pipeline](#autonomous-pipeline) section below for details.

## Run locally

Use **Python 3.11+** and pip.

```bash
git clone https://github.com/shaikhjiyauddin/Car4U.git
cd Car4U
pip install -r requirements.txt

# Train the models (first time only)
python src/train_models.py
python src/train_predictor.py

# Launch the app
streamlit run app/app.py
```

Open the localhost URL printed in the terminal. The app runs on port **8501** by default.

### Docker

```bash
docker compose up --build        # builds, trains models, runs on :8501
docker compose run --rm retrain  # retrain without rebuilding
```

## Project structure

```
Car4U/
├── app/
│   └── app.py                          # Streamlit application (9 pages)
├── src/
│   ├── recommendation_system.py        # Content-based car recommender
│   ├── price_prediction.py             # Linear regression price predictor
│   ├── database.py                     # SQLite marketplace backend
│   ├── train_models.py                 # Train recommendation model
│   ├── train_predictor.py              # Train price predictor model
│   ├── export_listings.py              # Export live listings → CSV
│   └── compare_metrics.py              # Safety gate for model deployment
├── data/
│   └── used_cars_v2.csv                # 7,254 used car records
├── models/                             # Generated by training scripts (gitignored)
│   ├── car_model.pkl                   # Trained recommendation model
│   ├── price_predictor_model.pkl       # Trained price predictor model
│   └── price_predictor_metrics.json    # MAE & R² from last training run
├── tests/
│   ├── test_price_prediction.py        # Price predictor unit tests
│   ├── test_recommendation_system.py   # Recommender unit tests
│   ├── test_database.py                # Marketplace DB tests
│   └── test_autonomous_pipeline.py     # End-to-end pipeline tests
├── .github/workflows/
│   ├── ci.yml                          # Lint + test on push/PR
│   └── retrain.yml                     # Automated model retraining
├── Dockerfile                          # Production container
├── docker-compose.yml                  # One-command deployment
├── requirements.txt                    # Pinned dependencies
├── pytest.ini                          # Test configuration
└── README.md                           # This file
```

## Autonomous pipeline

The project has a complete data-to-model feedback loop with a safety gate:

1. **Real users list real cars** through the Marketplace ("Sell Your Car" page). Each listing is stored in `data/car4u.db`, tied to a real account.
2. **Export new listings** into the training set:
   ```bash
   python src/export_listings.py
   ```
   This pulls every not-yet-exported listing from the database and appends it to `data/used_cars_v2.csv`. It's idempotent — running it twice won't duplicate rows.
3. **GitHub Actions detects the CSV change** and automatically triggers `retrain.yml`.
4. **Models are validated** before deployment — the price predictor must produce sane predictions, and the recommendation system must return results.
5. **The safety gate** (`compare_metrics.py`) compares new vs. old MAE. If the freshly trained model is worse (beyond a 2% tolerance), the workflow automatically reverts to the previous model instead of shipping a regression.

## Verification

Run locally:

```bash
pip install -r requirements.txt
pytest tests/ -v
```

The test suite covers:
- Model training and loading round-trips
- Prediction sanity (positive prices, newer-car-costs-more regression)
- Recommendation filtering (budget, fuel, seats, kilometers)
- Empty-result handling
- The `_obj_to_num` string parser edge cases
- Marketplace database operations (users, listings, messages)
- End-to-end autonomous pipeline (export → retrain → validate)

CI runs automatically on every push/PR to `main`/`develop` via GitHub Actions.

## Scope and next steps

This is a student project built for the AIML course. It does not include production authentication (OAuth), cloud-hosted databases, payment processing, or real-time notifications.

Planned improvements:

- Cloud database migration (PostgreSQL / Firebase) for multi-device access.
- Image-based condition assessment using computer vision.
- Advanced ML models (gradient boosting, neural networks) for price prediction.
- Real-time notification system for new messages and listings.

## Tech stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit, Plotly, Custom CSS |
| ML | scikit-learn (Linear Regression, Cosine Similarity) |
| Data | pandas, NumPy |
| Database | SQLite (stdlib `sqlite3`) |
| Auth | Salted PBKDF2 (stdlib `hashlib`) |
| Images | Pillow |
| CI/CD | GitHub Actions |
| Container | Docker, docker-compose |
| Testing | pytest, pytest-cov, ruff |
