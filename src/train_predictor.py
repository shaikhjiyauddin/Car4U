import os
import sys
import json

# Add current directory to path to import price_prediction
sys.path.append(os.path.dirname(__file__))

from price_prediction import PricePredictor

def train():
    print("--- Car4U Price Predictor Training ---")
    
    # Paths
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    data_path = os.path.join(base_dir, 'data', 'used_cars_v2.csv')
    model_dir = os.path.join(base_dir, 'models')
    model_path = os.path.join(model_dir, 'price_predictor_model.pkl')
    metrics_path = os.path.join(model_dir, 'price_predictor_metrics.json')
    
    # Ensure models directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")
    
    print(f"Loading data from: {data_path}")
    
    try:
        # Initialize and prepare features
        predictor = PricePredictor(data_path)
        predictor.prepare_data()
        
        print("Saving pre-trained model...")
        predictor.save_model(model_path)

        with open(metrics_path, 'w') as f:
            json.dump(predictor.metrics, f, indent=2)
        
        print(f"Success! Model saved to: {model_path}")
        print(f"Held-out accuracy — MAE: {predictor.metrics['mae']:.3f} Lakhs, "
              f"R²: {predictor.metrics['r2']:.4f} "
              f"(trained on {predictor.metrics['n_train']}, tested on {predictor.metrics['n_test']})")
        
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    train()

