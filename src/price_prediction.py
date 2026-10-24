import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

class PricePredictor:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.model = None
        self.model_columns = None
        self.brand_to_models = {}
        self.brand_to_median_price = {}
        self.model_to_median_price = {}
        self.outlier_bounds = {}
        self.medians = {}
        self.metrics = {}  # {'mae': float, 'r2': float, 'n_train': int, 'n_test': int}
        
    def _obj_to_num(self, n):
        if pd.isna(n):
            return np.nan
        if isinstance(n, (int, float)):
            return float(n)
        if isinstance(n, str):
            n_clean = n.strip().lower()
            if 'null' in n_clean or 'nan' in n_clean or not n_clean:
                return np.nan
            if n.endswith('kmpl'):
                try:
                    return float(n.split('kmpl')[0])
                except ValueError:
                    return np.nan
            elif n.endswith('km/kg'):
                try:
                    return float(n.split('km/kg')[0])
                except ValueError:
                    return np.nan
            elif n.endswith('CC'):
                try:
                    return float(n.split('CC')[0])
                except ValueError:
                    return np.nan
            elif n.endswith('bhp'):
                try:
                    return float(n.split('bhp')[0])
                except ValueError:
                    return np.nan
            try:
                return float(n)
            except ValueError:
                return np.nan
        try:
            return float(n)
        except (ValueError, TypeError):
            return np.nan

    def prepare_data(self):
        df = pd.read_csv(self.data_path)
        
        # Drop unnecessary columns
        if 'S.No.' in df.columns:
            df.drop(['S.No.'], axis=1, inplace=True)
        if 'New_Price' in df.columns:
            df.drop(['New_Price'], axis=1, inplace=True)
        if 'Listing_Ref' in df.columns:
            # Identifies rows that came from a real Marketplace listing — useful for
            # linking recommendations back to a chattable seller, not a training feature.
            df.drop(['Listing_Ref'], axis=1, inplace=True)
            
        # Convert objects to num
        num_values = ['Mileage', 'Engine', 'Power']
        for colname in num_values:
            if colname in df.columns:
                df[colname] = df[colname].apply(self._obj_to_num)
                df[colname] = df[colname].replace(0.0, np.nan)
                
        # Fill numerical NAs with median
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        if 'Price' in numeric_columns:
            numeric_columns.remove('Price')
            
        for col in numeric_columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            self.medians[col] = median_val
            
        # Drop empty price rows for training
        df.dropna(subset=['Price'], inplace=True)
        
        # Region Mapping
        regions = {'Delhi': 'North', 'Jaipur': 'North', 'Chennai': 'South', 
                   'Coimbatore': 'South', 'Hyderabad': 'South', 'Bangalore': 'South', 
                   'Kochi': 'South', 'Kolkata': 'East', 'Mumbai': 'West', 
                   'Pune': 'West', 'Ahmedabad': 'West'}
        df['Region'] = df['Location'].replace(regions)
        
        # Condition Encoding
        condition_mapping = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}
        if 'Condition_Grade' in df.columns:
            df['Condition_Num'] = df['Condition_Grade'].map(condition_mapping).fillna(3)
        else:
            df['Condition_Num'] = 3
        
        # UI Mappings
        df['Car_Brand'] = df['Car_Brand'].replace('Land', 'Land_Rover')
        df['Car_Brand'] = df['Car_Brand'].replace('ISUZU', 'Isuzu')
        
        self.brand_to_models = df.groupby('Car_Brand')['Model'].unique().apply(list).to_dict()
        
        # Target Encoding
        self.brand_to_median_price = df.groupby('Car_Brand')['Price'].median().to_dict()
        self.model_to_median_price = df.groupby('Model')['Price'].median().to_dict()
        
        df['Model_Median_Price'] = df['Model'].map(self.model_to_median_price)
        
        # Outliers treatment
        cols_for_outliers = ['Kilometers_Driven', 'Engine', 'Power']
        for col in cols_for_outliers:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            self.outlier_bounds[col] = (lower, upper)
            df[col] = np.clip(df[col], lower, upper)
            
        # Drop columns not used in X (keep Fuel_Type for encoding)
        X = df.drop(['Name', 'Location', 'Price', 'Car_Brand', 'Model', 'Condition_Grade'], axis=1, errors='ignore')
        y = df['Price']
        
        # Get dummies (include Fuel_Type for encoding)
        X = pd.get_dummies(X, columns=['Fuel_Type', 'Transmission', 'Owner_Type', 'Region'], drop_first=True)
        self.model_columns = list(X.columns)
        
        # Honest accuracy check: hold out 20% of the data the model never trains on,
        # so self.metrics reflects real-world performance instead of training-set fit.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        eval_model = LinearRegression()
        eval_model.fit(X_train, y_train)
        y_pred = eval_model.predict(X_test)
        self.metrics = {
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'r2': float(r2_score(y_test, y_pred)),
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
        }

        # Train the final production model on ALL the data (train+test) for best
        # real-world accuracy — self.metrics above already captured honest held-out
        # performance before this refit.
        self.model = LinearRegression()
        self.model.fit(X, y)
        
    def save_model(self, path):
        state = {
            'model': self.model,
            'model_columns': self.model_columns,
            'brand_to_models': self.brand_to_models,
            'brand_to_median_price': self.brand_to_median_price,
            'model_to_median_price': self.model_to_median_price,
            'outlier_bounds': self.outlier_bounds,
            'medians': self.medians,
            'metrics': self.metrics
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
            
    @classmethod
    def load_model(cls, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        predictor = cls()
        predictor.model = state['model']
        predictor.model_columns = state['model_columns']
        predictor.brand_to_models = state['brand_to_models']
        predictor.brand_to_median_price = state['brand_to_median_price']
        predictor.model_to_median_price = state['model_to_median_price']
        predictor.outlier_bounds = state['outlier_bounds']
        predictor.medians = state['medians']
        predictor.metrics = state.get('metrics', {})
        return predictor
        
    def predict(self, input_dict):
        # Region
        regions = {'Delhi': 'North', 'Jaipur': 'North', 'Chennai': 'South', 
                   'Coimbatore': 'South', 'Hyderabad': 'South', 'Bangalore': 'South', 
                   'Kochi': 'South', 'Kolkata': 'East', 'Mumbai': 'West', 
                   'Pune': 'West', 'Ahmedabad': 'West'}
        region = regions.get(input_dict.get('Location', 'Mumbai'), 'West')
        
        # Model Target Encoding
        brand = input_dict.get('Brand', 'Maruti')
        model_name = input_dict.get('Model', '')
        
        # fallback hierarchy: Exact Model -> Brand -> Global Median
        median_price = self.model_to_median_price.get(model_name, 
                           self.brand_to_median_price.get(brand, 5.0))
                           
        condition_mapping = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}
        condition_num = condition_mapping.get(input_dict.get('Condition_Grade', 'C'), 3)
        
        # Build df
        df_input = pd.DataFrame([{
            'Year': input_dict.get('Year', self.medians.get('Year', 2014)),
            'Kilometers_Driven': input_dict.get('Kilometers_Driven', self.medians.get('Kilometers_Driven', 50000)),
            'Mileage': input_dict.get('Mileage', self.medians.get('Mileage', 18)),
            'Engine': input_dict.get('Engine', self.medians.get('Engine', 1200)),
            'Power': input_dict.get('Power', self.medians.get('Power', 80)),
            'Seats': input_dict.get('Seats', self.medians.get('Seats', 5)),
            'Safety_Rating': input_dict.get('Safety_Rating', self.medians.get('Safety_Rating', 3.0)),
            'Model_Median_Price': median_price,
            'Condition_Num': condition_num,
            'Fuel_Type': input_dict.get('Fuel_Type', 'Petrol'),
            'Transmission': input_dict.get('Transmission', 'Manual'),
            'Owner_Type': input_dict.get('Owner_Type', 'First'),
            'Region': region
        }])
        
        # Clip outliers
        for col in ['Kilometers_Driven', 'Engine', 'Power']:
            lower, upper = self.outlier_bounds.get(col, (0, float('inf')))
            df_input[col] = np.clip(df_input[col], lower, upper)
            
        # Get dummies (include Fuel_Type)
        df_input = pd.get_dummies(df_input, columns=['Fuel_Type', 'Transmission', 'Owner_Type', 'Region'])
        
        # Align columns
        for col in self.model_columns:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input[self.model_columns]
        
        pred = self.model.predict(df_input)[0]
        return max(0, pred)
