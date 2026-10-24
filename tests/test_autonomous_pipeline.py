"""
Tests for src/export_listings.py and src/compare_metrics.py
(the pieces that make the retraining pipeline autonomous)

Run with: pytest tests/ -v
"""
import os
import sys
import json
import subprocess
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import database as db
import export_listings

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
COMPARE_SCRIPT = os.path.join(REPO_ROOT, 'src', 'compare_metrics.py')


@pytest.fixture
def sample_csv(tmp_path):
    """A tiny, realistic CSV matching the real schema."""
    path = str(tmp_path / "used_cars_sample.csv")
    df = pd.DataFrame([{
        'S.No.': 0, 'Name': 'Honda City', 'Location': 'Mumbai', 'Year': 2016,
        'Kilometers_Driven': 40000, 'Fuel_Type': 'Petrol', 'Transmission': 'Manual',
        'Owner_Type': 'First', 'Mileage': '18.0 kmpl', 'Engine': '1200 CC', 'Power': '85 bhp',
        'Seats': 5, 'New_Price': None, 'Price': 6.0, 'Safety_Rating': 3.0,
        'Car_Brand': 'Honda', 'Model': 'City', 'Condition_Grade': 'B'
    }])
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def test_db(tmp_path):
    path = str(tmp_path / "test.db")
    db.init_db(path)
    return path


class TestExportListings:
    def test_no_listings_exports_nothing(self, sample_csv, test_db):
        count = export_listings.export_new_listings(data_path=sample_csv, db_path=test_db)
        assert count == 0

    def test_export_adds_rows(self, sample_csv, test_db):
        db.create_user('Wajahat', 'wajahat@example.com', 'password123', test_db)
        seller = db.verify_login('wajahat@example.com', 'password123', test_db)
        db.add_listing(seller['id'], 'Maruti', 'Swift', 2019, 5.2, 'Petrol', 'Manual',
                        25000, 'Ahmedabad', 'Great condition', test_db)

        before = len(pd.read_csv(sample_csv))
        count = export_listings.export_new_listings(data_path=sample_csv, db_path=test_db)
        after = pd.read_csv(sample_csv)

        assert count == 1
        assert len(after) == before + 1
        assert after.iloc[-1]['Car_Brand'] == 'Maruti'
        assert after.iloc[-1]['Price'] == 5.2

    def test_export_is_idempotent(self, sample_csv, test_db):
        db.create_user('Wajahat', 'wajahat@example.com', 'password123', test_db)
        seller = db.verify_login('wajahat@example.com', 'password123', test_db)
        db.add_listing(seller['id'], 'Maruti', 'Swift', 2019, 5.2, 'Petrol', 'Manual',
                        25000, 'Ahmedabad', '', test_db)

        first_count = export_listings.export_new_listings(data_path=sample_csv, db_path=test_db)
        second_count = export_listings.export_new_listings(data_path=sample_csv, db_path=test_db)
        assert first_count == 1
        assert second_count == 0  # already exported, shouldn't duplicate

    def test_exported_data_keeps_pipeline_trainable(self, sample_csv, test_db):
        """The whole point: merged data must not break PricePredictor training."""
        sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
        from price_prediction import PricePredictor

        db.create_user('Wajahat', 'wajahat@example.com', 'password123', test_db)
        seller = db.verify_login('wajahat@example.com', 'password123', test_db)
        # Add several listings so there's enough data to train/test split
        for i in range(10):
            db.add_listing(seller['id'], 'Maruti', 'Swift', 2015 + i % 5, 4.0 + i,
                            'Petrol', 'Manual', 20000 + i * 1000, 'Ahmedabad', '', test_db)
        export_listings.export_new_listings(data_path=sample_csv, db_path=test_db)

        predictor = PricePredictor(sample_csv)
        predictor.prepare_data()  # should not raise
        assert predictor.model is not None

    def test_exported_listing_is_traceable_back_to_its_seller(self, sample_csv, test_db):
        """A recommendation built from an exported listing must carry a listing_ref
        that resolves back to the real seller — this is what powers the
        'Chat with Owner' button on the Recommendations page."""
        sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
        from recommendation_system import CarRecommendationSystem

        db.create_user('Wajahat', 'wajahat@example.com', 'password123', test_db)
        seller = db.verify_login('wajahat@example.com', 'password123', test_db)
        db.add_listing(seller['id'], 'Maruti', 'Swift', 2020, 3.5, 'Petrol', 'Manual',
                        10000, 'Ahmedabad', 'Brand new condition', test_db)
        export_listings.export_new_listings(data_path=sample_csv, db_path=test_db)

        system = CarRecommendationSystem(sample_csv)
        assert 'listing_ref' in system.df.columns

        recs = system.recommend_by_preferences({}, n_recommendations=999)
        assert 'listing_ref' in recs.columns
        live_rows = recs[recs['listing_ref'].notna()]
        assert len(live_rows) == 1

        matched_listing = db.get_listing(int(live_rows.iloc[0]['listing_ref']), test_db)
        assert matched_listing['owner_name'] == 'Wajahat'

    def test_historical_rows_have_no_listing_ref(self, sample_csv, test_db):
        """Rows that were never a real Marketplace listing must NOT get a chat button."""
        sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
        from recommendation_system import CarRecommendationSystem

        system = CarRecommendationSystem(sample_csv)
        recs = system.recommend_by_preferences({}, n_recommendations=999)
        assert recs['listing_ref'].notna().sum() == 0


class TestSimilarityEngine:
    def test_get_similar_cars_returns_ranked_results(self, sample_csv, test_db):
        sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
        from recommendation_system import CarRecommendationSystem

        # Add enough rows for a meaningful similarity comparison
        import pandas as pd
        rows = []
        for i in range(15):
            rows.append({
                'S.No.': i + 1, 'Name': f'Car {i}', 'Location': 'Mumbai', 'Year': 2015 + (i % 5),
                'Kilometers_Driven': 30000 + i * 1000, 'Fuel_Type': 'Petrol', 'Transmission': 'Manual',
                'Owner_Type': 'First', 'Mileage': '18.0 kmpl', 'Engine': '1200 CC', 'Power': f'{80+i} bhp',
                'Seats': 5, 'New_Price': None, 'Price': 5.0 + i * 0.3, 'Safety_Rating': 3.0,
                'Car_Brand': 'Honda', 'Model': f'City {i}', 'Condition_Grade': 'B'
            })
        pd.DataFrame(rows).to_csv(sample_csv, index=False)

        system = CarRecommendationSystem(sample_csv)
        target_idx = system.df.index[0]
        similar = system.get_similar_cars(target_idx, n_similar=5)

        assert len(similar) == 5
        assert target_idx not in similar.index
        assert similar['similarity'].is_monotonic_decreasing

    def test_get_similar_cars_missing_index_returns_empty(self, sample_csv, test_db):
        sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
        from recommendation_system import CarRecommendationSystem

        system = CarRecommendationSystem(sample_csv)
        result = system.get_similar_cars(99999, n_similar=5)
        assert len(result) == 0

    def test_similar_cars_works_for_live_marketplace_listing(self, sample_csv, test_db):
        """A car found via get_similar_cars must work even when the target row
        is a live Marketplace listing (car_id is NaN, only listing_ref is set)."""
        sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
        from recommendation_system import CarRecommendationSystem

        db.create_user('Wajahat', 'wajahat@example.com', 'password123', test_db)
        seller = db.verify_login('wajahat@example.com', 'password123', test_db)
        db.add_listing(seller['id'], 'Honda', 'City', 2018, 6.0, 'Petrol', 'Manual',
                        25000, 'Mumbai', '', test_db)
        export_listings.export_new_listings(data_path=sample_csv, db_path=test_db)

        system = CarRecommendationSystem(sample_csv)
        live_row = system.df[system.df['listing_ref'].notna()].iloc[0]
        similar = system.get_similar_cars(live_row.name, n_similar=3)
        assert isinstance(similar, type(system.df))  # is a DataFrame, didn't crash


class TestCompareMetrics:
    def _write(self, path, metrics):
        with open(path, 'w') as f:
            json.dump(metrics, f)

    def test_better_model_is_accepted(self, tmp_path):
        old = tmp_path / "old.json"
        new = tmp_path / "new.json"
        self._write(old, {'mae': 1.5, 'r2': 0.90})
        self._write(new, {'mae': 1.2, 'r2': 0.93})
        result = subprocess.run([sys.executable, COMPARE_SCRIPT, str(old), str(new)])
        assert result.returncode == 0

    def test_worse_model_is_rejected(self, tmp_path):
        old = tmp_path / "old.json"
        new = tmp_path / "new.json"
        self._write(old, {'mae': 1.5, 'r2': 0.90})
        self._write(new, {'mae': 3.0, 'r2': 0.50})
        result = subprocess.run([sys.executable, COMPARE_SCRIPT, str(old), str(new)])
        assert result.returncode == 1

    def test_first_run_with_no_old_metrics_is_accepted(self, tmp_path):
        new = tmp_path / "new.json"
        self._write(new, {'mae': 1.2, 'r2': 0.93})
        missing_old = tmp_path / "does_not_exist.json"
        result = subprocess.run([sys.executable, COMPARE_SCRIPT, str(missing_old), str(new)])
        assert result.returncode == 0

    def test_slightly_worse_within_tolerance_is_accepted(self, tmp_path):
        old = tmp_path / "old.json"
        new = tmp_path / "new.json"
        self._write(old, {'mae': 1.500, 'r2': 0.90})
        self._write(new, {'mae': 1.510, 'r2': 0.899})  # within default 2% tolerance
        result = subprocess.run([sys.executable, COMPARE_SCRIPT, str(old), str(new)])
        assert result.returncode == 0

    def test_missing_new_metrics_blocks_deployment(self, tmp_path):
        old = tmp_path / "old.json"
        self._write(old, {'mae': 1.5, 'r2': 0.90})
        missing_new = tmp_path / "does_not_exist.json"
        result = subprocess.run([sys.executable, COMPARE_SCRIPT, str(old), str(missing_new)])
        assert result.returncode == 1
