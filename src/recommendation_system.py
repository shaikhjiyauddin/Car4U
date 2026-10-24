import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

class CarRecommendationSystem:
    """
    Car4U: Intelligent Car Recommendation System
    Uses content-based filtering and collaborative filtering approaches
    Powered by the used cars dataset (7,254 real Indian market cars)
    """
    
    def __init__(self, data_path=None):
        self.feature_matrix = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        if data_path:
            self.df = pd.read_csv(data_path)
            self._prepare_columns()
            self.prepare_features()
        else:
            self.df = None
    
    @staticmethod
    def _obj_to_num(val):
        """Parse string values like '1200 CC', '88.7 bhp', '18.2 kmpl' to float"""
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float)):
            return float(val)
        val = str(val).strip()
        if val.lower().startswith('null'):
            return np.nan
        for suffix in ['kmpl', 'km/kg', 'CC', 'bhp']:
            if val.endswith(suffix):
                try:
                    return float(val.replace(suffix, '').strip())
                except ValueError:
                    return np.nan
        try:
            return float(val)
        except ValueError:
            return np.nan
    
    def _prepare_columns(self):
        """Standardize column names and parse string values to numeric"""
        # Rename columns to lowercase standardized names
        rename_map = {
            'S.No.': 'car_id',
            'Name': 'name',
            'Location': 'location',
            'Year': 'year',
            'Kilometers_Driven': 'kilometers_driven',
            'Fuel_Type': 'fuel_type',
            'Transmission': 'transmission',
            'Owner_Type': 'owner_type',
            'Mileage': 'mileage_raw',
            'Engine': 'engine_raw',
            'Power': 'power_raw',
            'Seats': 'seats',
            'New_Price': 'new_price',
            'Price': 'price',
            'Safety_Rating': 'safety_rating',
            'Car_Brand': 'brand',
            'Model': 'model',
            'Condition_Grade': 'condition_grade',
            'Listing_Ref': 'listing_ref'
        }
        self.df.rename(columns=rename_map, inplace=True)

        # Older CSVs (or a CSV that's never had a Marketplace listing exported into it)
        # won't have this column at all — add it so downstream code can always rely on it.
        if 'listing_ref' not in self.df.columns:
            self.df['listing_ref'] = np.nan
        
        # Drop rows without price
        self.df.dropna(subset=['price'], inplace=True)
        
        # Parse string numeric columns
        self.df['mileage'] = self.df['mileage_raw'].apply(self._obj_to_num)
        self.df['engine'] = self.df['engine_raw'].apply(self._obj_to_num)
        self.df['power'] = self.df['power_raw'].apply(self._obj_to_num)
        
        # Ensure seats is numeric
        self.df['seats'] = pd.to_numeric(self.df['seats'], errors='coerce')
        
        # Ensure safety_rating is numeric
        self.df['safety_rating'] = pd.to_numeric(self.df['safety_rating'], errors='coerce')
        
        # Ensure car_id exists
        if 'car_id' not in self.df.columns:
            self.df['car_id'] = range(len(self.df))
        
        # Reset index for clean indexing
        self.df.reset_index(drop=True, inplace=True)
            
    def save_model(self, model_path):
        """Save the system state to a file"""
        state = {
            'df': self.df,
            'feature_matrix': self.feature_matrix,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
        with open(model_path, 'wb') as f:
            pickle.dump(state, f)
            
    @staticmethod
    def load_model(model_path):
        """Load the system state from a file"""
        with open(model_path, 'rb') as f:
            state = pickle.load(f)
        
        system = CarRecommendationSystem()
        system.df = state['df']
        system.feature_matrix = state['feature_matrix']
        system.scaler = state['scaler']
        system.label_encoders = state['label_encoders']

        # A .pkl saved before the Marketplace/listing_ref feature existed won't
        # have this column — add it so old models don't crash newer app code.
        if 'listing_ref' not in system.df.columns:
            system.df['listing_ref'] = np.nan

        return system
        
    def prepare_features(self):
        """Prepare and encode features for recommendation"""
        # Feature engineering on the main dataframe
        self.df['age'] = datetime.datetime.now().year - self.df['year']
        self.df['price_per_power'] = self.df['price'] / self.df['power'].replace(0, np.nan)
        
        # Condition encoding
        condition_mapping = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}
        if 'condition_grade' in self.df.columns:
            self.df['condition_num'] = self.df['condition_grade'].map(condition_mapping).fillna(3)
        else:
            self.df['condition_num'] = 3

        # Create a copy for feature encoding and matrix generation
        df_features = self.df.copy()
        
        # Encode categorical variables
        categorical_cols = ['brand', 'fuel_type', 'transmission', 'owner_type', 'location']
        for col in categorical_cols:
            if col in df_features.columns:
                le = LabelEncoder()
                df_features[col + '_encoded'] = le.fit_transform(df_features[col].fillna('Unknown'))
                self.label_encoders[col] = le
        
        # Select features for recommendation
        feature_cols = [
            'price', 'year', 'kilometers_driven', 'mileage', 'engine', 'power',
            'seats', 'safety_rating', 'condition_num', 'age', 'price_per_power',
            'brand_encoded', 'fuel_type_encoded', 'transmission_encoded',
            'owner_type_encoded', 'location_encoded'
        ]
        
        # Only use columns that actually exist
        feature_cols = [c for c in feature_cols if c in df_features.columns]
        
        # Handle NaN values
        imputer = SimpleImputer(strategy='median')
        imputed_features = imputer.fit_transform(df_features[feature_cols])
        
        # Scale features
        self.feature_matrix = self.scaler.fit_transform(imputed_features)

    def get_similar_cars(self, row_index, n_similar=5):
        """Genuine content-based filtering: given one car (by its DataFrame
        index — use df.index, not car_id, since live Marketplace listings
        don't have a car_id), find the cars most similar to it using cosine
        similarity over the same scaled feature space (price, year, mileage,
        engine, power, brand/fuel/transmission encodings, etc.) built in
        prepare_features(). This is the actual ML model doing the ranking,
        not a hand-tuned rule.

        Returns an empty DataFrame if the index isn't found or features
        haven't been prepared yet."""
        if self.feature_matrix is None or row_index not in self.df.index:
            return pd.DataFrame()

        position = self.df.index.get_loc(row_index)
        target_vector = self.feature_matrix[position].reshape(1, -1)
        similarities = cosine_similarity(target_vector, self.feature_matrix)[0]

        result_df = self.df.copy()
        result_df['similarity'] = similarities
        result_df = result_df.drop(index=row_index)
        top = result_df.sort_values('similarity', ascending=False).head(n_similar)

        cols = ['car_id', 'brand', 'model', 'name', 'year', 'price', 'fuel_type',
                'transmission', 'kilometers_driven', 'location', 'listing_ref', 'similarity']
        return top[[c for c in cols if c in top.columns]]

    def recommend_by_preferences(self, preferences, n_recommendations=5):
        """
        Recommend cars based on user preferences
        preferences: dict with keys like budget, fuel_type, transmission, location, etc.
        """
        filtered_df = self.df.copy()
        
        # Apply filters based on preferences
        if 'budget' in preferences and preferences['budget']:
            filtered_df = filtered_df[filtered_df['price'] <= preferences['budget']]
        
        if 'fuel_type' in preferences and preferences['fuel_type']:
            filtered_df = filtered_df[filtered_df['fuel_type'] == preferences['fuel_type']]
        
        if 'transmission' in preferences and preferences['transmission']:
            filtered_df = filtered_df[filtered_df['transmission'] == preferences['transmission']]
        
        if 'location' in preferences and preferences['location']:
            filtered_df = filtered_df[filtered_df['location'] == preferences['location']]
        
        if 'owner_type' in preferences and preferences['owner_type']:
            filtered_df = filtered_df[filtered_df['owner_type'] == preferences['owner_type']]
        
        if 'min_seats' in preferences:
            filtered_df = filtered_df[filtered_df['seats'] >= preferences['min_seats']]
        
        if 'min_safety' in preferences:
            filtered_df = filtered_df[filtered_df['safety_rating'] >= preferences['min_safety']]
        
        if 'max_kilometers' in preferences:
            filtered_df = filtered_df[filtered_df['kilometers_driven'] <= preferences['max_kilometers']]
        
        if 'condition_grade' in preferences and preferences['condition_grade'] and preferences['condition_grade'] != 'All':
            condition_mapping = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}
            pref_num = condition_mapping.get(preferences['condition_grade'], 3)
            filtered_df = filtered_df[filtered_df['condition_num'] >= pref_num]
        
        # If no cars match filters, return empty
        if len(filtered_df) == 0:
            return pd.DataFrame()
        
        # Score remaining cars based on preferences
        filtered_df = filtered_df.copy()
        filtered_df['score'] = 0
        
        # Score based on price (lower is better within budget)
        if 'budget' in preferences and preferences['budget']:
            max_price = filtered_df['price'].max()
            if max_price > 0:
                filtered_df['score'] += (1 - filtered_df['price'] / max_price) * 25
        
        # Score based on safety rating
        max_safety = filtered_df['safety_rating'].max()
        if pd.notna(max_safety) and max_safety > 0:
            filtered_df['score'] += (filtered_df['safety_rating'].fillna(0) / max_safety) * 20
            
        # Score based on condition
        filtered_df['score'] += (filtered_df['condition_num'] / 5) * 15
        
        # Score based on mileage efficiency (higher kmpl is better)
        max_mileage = filtered_df['mileage'].max()
        if pd.notna(max_mileage) and max_mileage > 0:
            filtered_df['score'] += (filtered_df['mileage'].fillna(0) / max_mileage) * 15
        
        # Score based on lower kilometers driven
        max_km = filtered_df['kilometers_driven'].max()
        if pd.notna(max_km) and max_km > 0:
            filtered_df['score'] += (1 - filtered_df['kilometers_driven'].fillna(0) / max_km) * 15
        
        # Score based on power (higher is better)
        max_power = filtered_df['power'].max()
        if pd.notna(max_power) and max_power > 0:
            filtered_df['score'] += (filtered_df['power'].fillna(0) / max_power) * 10
        
        # Sort by score and return top recommendations
        recommendations = filtered_df.sort_values('score', ascending=False).head(n_recommendations)
        
        return recommendations[['car_id', 'brand', 'model', 'name', 'year', 'price', 
                               'fuel_type', 'transmission', 'kilometers_driven',
                               'power_raw', 'engine_raw', 'mileage_raw',
                               'seats', 'safety_rating', 'condition_grade',
                               'location', 'owner_type', 'listing_ref', 'score']]
    

    def get_car_details(self, car_id):
        """Get detailed information about a specific car"""
        return self.df[self.df['car_id'] == car_id].to_dict('records')[0]
    
    def get_all_brands(self):
        """Get list of all brands"""
        return sorted(self.df['brand'].dropna().unique().tolist())
    
    def get_all_fuel_types(self):
        """Get list of all fuel types"""
        return sorted(self.df['fuel_type'].dropna().unique().tolist())
    
    def get_all_locations(self):
        """Get list of all locations"""
        return sorted(self.df['location'].dropna().unique().tolist())
    
    def get_all_owner_types(self):
        """Get list of all owner types"""
        return sorted(self.df['owner_type'].dropna().unique().tolist())
    
    def get_all_transmissions(self):
        """Get list of all transmissions"""
        return sorted(self.df['transmission'].dropna().unique().tolist())


if __name__ == "__main__":
    # Test the recommendation system
    system = CarRecommendationSystem('../data/used_cars_v2.csv')
    
    print("=== Preference-Based Recommendation ===")
    preferences = {
        'budget': 10,
        'fuel_type': 'Diesel',
        'min_seats': 5,
        'min_safety': 3
    }
    recs = system.recommend_by_preferences(preferences, n_recommendations=5)
    print(recs[['brand', 'model', 'year', 'price', 'fuel_type']].to_string())
