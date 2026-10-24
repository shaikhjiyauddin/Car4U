"""
export_listings.py — pulls new, not-yet-exported Marketplace listings from
car4u.db and appends them to data/used_cars_v2.csv in the same schema the
training pipeline (PricePredictor / CarRecommendationSystem) already expects.

This is what makes the project "autonomous": every car a real user lists
through the app eventually grows the training dataset by itself, with no
one manually editing the CSV.

Run automatically by the retrain workflow before training. Safe to run
manually too:  python src/export_listings.py
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
import database as db

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'used_cars_v2.csv')


def listing_to_row(listing):
    """Map one Marketplace listing (dict from database.py) into the
    historical CSV's column schema. Fields the Marketplace form doesn't
    collect are left as NaN so the existing median-fill logic in
    PricePredictor.prepare_data() handles them the same way it already
    handles missing values in the original dataset.

    Listing_Ref carries the real listing's database id through into the
    CSV, so the Recommendation engine can later tell which rows are real,
    chattable Marketplace listings versus historical reference data."""
    return {
        'S.No.': np.nan,
        'Name': f"{listing['brand']} {listing['model']}",
        'Location': listing.get('location') or np.nan,
        'Year': listing['year'],
        'Kilometers_Driven': listing.get('kilometers_driven') or np.nan,
        'Fuel_Type': listing.get('fuel_type') or np.nan,
        'Transmission': listing.get('transmission') or np.nan,
        'Owner_Type': 'First',  # not collected by the Marketplace form yet; reasonable default
        'Mileage': np.nan,
        'Engine': np.nan,
        'Power': np.nan,
        'Seats': np.nan,
        'New_Price': np.nan,
        'Price': listing['price'],
        'Safety_Rating': np.nan,
        'Car_Brand': listing['brand'],
        'Model': listing['model'],
        'Condition_Grade': np.nan,
        'Listing_Ref': listing['id'],
    }


def export_new_listings(data_path=None, db_path=None):
    """Returns the number of listings merged into the CSV."""
    data_path = data_path or DATA_PATH
    new_listings = db.get_unexported_listings(db_path)

    if not new_listings:
        print("No new Marketplace listings to export.")
        return 0

    df_existing = pd.read_csv(data_path)
    if 'Listing_Ref' not in df_existing.columns:
        # One-time migration for CSVs created before this column existed —
        # historical rows have no live listing behind them.
        df_existing['Listing_Ref'] = np.nan

    df_new = pd.DataFrame([listing_to_row(l) for l in new_listings])
    # Keep only columns the CSV actually has, in the CSV's own column order
    df_new = df_new.reindex(columns=df_existing.columns)

    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(data_path, index=False)

    db.mark_listings_exported([l['id'] for l in new_listings], db_path)
    print(f"Exported {len(new_listings)} new Marketplace listing(s) into {data_path}")
    return len(new_listings)


if __name__ == "__main__":
    export_new_listings()
