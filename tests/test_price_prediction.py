"""
Tests for src/price_prediction.py

Run with:  pytest tests/ -v
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from price_prediction import PricePredictor

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'used_cars_v2.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'price_predictor_model.pkl')


@pytest.fixture(scope="module")
def trained_predictor():
    """Train a fresh predictor once per test module (avoids stale-pickle issues)."""
    predictor = PricePredictor(DATA_PATH)
    predictor.prepare_data()
    return predictor


class TestPricePredictorTraining:
    def test_prepare_data_sets_model(self, trained_predictor):
        assert trained_predictor.model is not None

    def test_model_columns_populated(self, trained_predictor):
        assert trained_predictor.model_columns is not None
        assert len(trained_predictor.model_columns) > 0

    def test_brand_to_models_populated(self, trained_predictor):
        assert len(trained_predictor.brand_to_models) > 0
        assert 'Maruti' in trained_predictor.brand_to_models

    def test_medians_populated(self, trained_predictor):
        assert 'Year' in trained_predictor.medians
        assert 'Kilometers_Driven' in trained_predictor.medians


class TestPricePredictorPredict:
    def test_predict_returns_positive_float(self, trained_predictor):
        price = trained_predictor.predict({
            'Brand': 'Maruti', 'Model': 'Swift', 'Year': 2016,
            'Kilometers_Driven': 40000, 'Fuel_Type': 'Petrol',
            'Transmission': 'Manual', 'Owner_Type': 'First',
            'Location': 'Mumbai', 'Mileage': 20.0, 'Engine': 1200,
            'Power': 85, 'Seats': 5, 'Safety_Rating': 3.0,
            'Condition_Grade': 'B'
        })
        assert isinstance(price, float)
        assert price > 0

    def test_predict_never_negative(self, trained_predictor):
        """Even with a pathological input, predicted price should be clamped >= 0."""
        price = trained_predictor.predict({
            'Brand': 'Unknown_Brand', 'Model': 'Unknown_Model', 'Year': 1990,
            'Kilometers_Driven': 500000, 'Fuel_Type': 'Petrol',
            'Transmission': 'Manual', 'Owner_Type': 'Fourth & Above',
            'Location': 'Mumbai', 'Mileage': 5.0, 'Engine': 500,
            'Power': 30, 'Seats': 2, 'Safety_Rating': 1.0,
            'Condition_Grade': 'E'
        })
        assert price >= 0

    def test_predict_handles_missing_optional_fields(self, trained_predictor):
        """predict() should fall back to medians/defaults for missing keys."""
        price = trained_predictor.predict({'Brand': 'Honda', 'Model': 'City'})
        assert price >= 0

    def test_newer_car_priced_higher_than_older_same_spec(self, trained_predictor):
        """Sanity/regression check: a newer car should generally be worth more."""
        base = {
            'Brand': 'Maruti', 'Model': 'Swift', 'Kilometers_Driven': 40000,
            'Fuel_Type': 'Petrol', 'Transmission': 'Manual', 'Owner_Type': 'First',
            'Location': 'Mumbai', 'Mileage': 20.0, 'Engine': 1200, 'Power': 85,
            'Seats': 5, 'Safety_Rating': 3.0, 'Condition_Grade': 'B'
        }
        old_price = trained_predictor.predict({**base, 'Year': 2008})
        new_price = trained_predictor.predict({**base, 'Year': 2020})
        assert new_price > old_price


class TestPricePredictorSaveLoad:
    def test_save_and_load_roundtrip(self, trained_predictor, tmp_path):
        save_path = tmp_path / "test_model.pkl"
        trained_predictor.save_model(str(save_path))
        assert save_path.exists()

        loaded = PricePredictor.load_model(str(save_path))
        assert loaded.model_columns == trained_predictor.model_columns

        price = loaded.predict({
            'Brand': 'Maruti', 'Model': 'Swift', 'Year': 2016,
            'Kilometers_Driven': 40000, 'Fuel_Type': 'Petrol',
            'Transmission': 'Manual', 'Owner_Type': 'First',
            'Location': 'Mumbai', 'Mileage': 20.0, 'Engine': 1200,
            'Power': 85, 'Seats': 5, 'Safety_Rating': 3.0,
            'Condition_Grade': 'B'
        })
        assert price > 0


class TestObjToNumParsing:
    def test_parses_kmpl(self):
        p = PricePredictor()
        assert p._obj_to_num("18.2 kmpl") == pytest.approx(18.2)

    def test_parses_cc(self):
        p = PricePredictor()
        assert p._obj_to_num("1198 CC") == pytest.approx(1198)

    def test_parses_bhp(self):
        p = PricePredictor()
        assert p._obj_to_num("88.7 bhp") == pytest.approx(88.7)

    def test_handles_null_strings(self):
        p = PricePredictor()
        import math
        assert math.isnan(p._obj_to_num("null bhp"))

    def test_handles_plain_number(self):
        p = PricePredictor()
        assert p._obj_to_num("5") == 5.0
