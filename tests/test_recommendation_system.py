"""
Tests for src/recommendation_system.py

Run with:  pytest tests/ -v
"""
import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from recommendation_system import CarRecommendationSystem

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'used_cars_v2.csv')


@pytest.fixture(scope="module")
def system():
    return CarRecommendationSystem(DATA_PATH)


class TestDataPreparation:
    def test_loads_data(self, system):
        assert system.df is not None
        assert len(system.df) > 0

    def test_no_missing_price_rows(self, system):
        assert system.df['price'].isna().sum() == 0

    def test_engineered_columns_exist(self, system):
        for col in ['age', 'price_per_power', 'condition_num']:
            assert col in system.df.columns

    def test_feature_matrix_shape(self, system):
        assert system.feature_matrix is not None
        assert system.feature_matrix.shape[0] == len(system.df)


class TestLookups:
    def test_get_all_brands_nonempty_sorted(self, system):
        brands = system.get_all_brands()
        assert len(brands) > 0
        assert brands == sorted(brands)

    def test_get_all_fuel_types(self, system):
        fuels = system.get_all_fuel_types()
        assert 'Diesel' in fuels or 'Petrol' in fuels

    def test_get_all_locations(self, system):
        assert len(system.get_all_locations()) > 0

    def test_get_car_details_returns_dict(self, system):
        car_id = system.df.iloc[0]['car_id']
        details = system.get_car_details(car_id)
        assert isinstance(details, dict)
        assert details['car_id'] == car_id


class TestRecommendations:
    def test_recommend_respects_budget(self, system):
        recs = system.recommend_by_preferences({'budget': 10}, n_recommendations=10)
        assert (recs['price'] <= 10).all()

    def test_recommend_respects_fuel_type(self, system):
        recs = system.recommend_by_preferences({'fuel_type': 'Diesel'}, n_recommendations=10)
        if len(recs) > 0:
            assert (recs['fuel_type'] == 'Diesel').all()

    def test_recommend_respects_min_seats(self, system):
        recs = system.recommend_by_preferences({'min_seats': 7}, n_recommendations=10)
        if len(recs) > 0:
            assert (recs['seats'] >= 7).all()

    def test_recommend_respects_max_kilometers(self, system):
        recs = system.recommend_by_preferences({'max_kilometers': 20000}, n_recommendations=10)
        if len(recs) > 0:
            assert (recs['kilometers_driven'] <= 20000).all()

    def test_recommend_returns_empty_df_when_no_match(self, system):
        recs = system.recommend_by_preferences({'budget': 0.01}, n_recommendations=5)
        assert isinstance(recs, pd.DataFrame)
        assert len(recs) == 0

    def test_recommend_respects_n_recommendations(self, system):
        recs = system.recommend_by_preferences({}, n_recommendations=3)
        assert len(recs) <= 3

    def test_recommend_sorted_by_score_descending(self, system):
        recs = system.recommend_by_preferences({}, n_recommendations=10)
        scores = recs['score'].tolist()
        assert scores == sorted(scores, reverse=True)


class TestSaveLoad:
    def test_save_and_load_roundtrip(self, system, tmp_path):
        save_path = tmp_path / "test_rec_model.pkl"
        system.save_model(str(save_path))
        assert save_path.exists()

        loaded = CarRecommendationSystem.load_model(str(save_path))
        assert len(loaded.df) == len(system.df)
        recs = loaded.recommend_by_preferences({'budget': 10}, n_recommendations=3)
        assert isinstance(recs, pd.DataFrame)

    def test_load_model_self_heals_old_pickles_missing_listing_ref(self, system, tmp_path):
        """Regression test: a .pkl saved before the Marketplace/listing_ref feature
        existed must still load and work, instead of crashing recommend_by_preferences
        with a KeyError on 'listing_ref'."""
        # Simulate an old-style pickle by stripping the column before saving
        system.df = system.df.drop(columns=['listing_ref'])
        assert 'listing_ref' not in system.df.columns

        old_style_path = tmp_path / "old_style.pkl"
        system.save_model(str(old_style_path))

        loaded = CarRecommendationSystem.load_model(str(old_style_path))
        assert 'listing_ref' in loaded.df.columns

        recs = loaded.recommend_by_preferences({'budget': 10}, n_recommendations=5)
        assert 'listing_ref' in recs.columns
