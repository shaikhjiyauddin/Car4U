import os
import sys

# Add current directory to path to import recommendation_system
sys.path.append(os.path.dirname(__file__))

from recommendation_system import CarRecommendationSystem

def train():
    print("--- Car4U Model Training ---")
    
    # Paths
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    data_path = os.path.join(base_dir, 'data', 'used_cars_v2.csv')
    model_dir = os.path.join(base_dir, 'models')
    model_path = os.path.join(model_dir, 'car_model.pkl')
    
    # Ensure models directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")
    
    print(f"Loading data from: {data_path}")
    
    try:
        # Initialize and prepare features
        system = CarRecommendationSystem(data_path)
        
        print("Saving pre-trained model...")
        system.save_model(model_path)
        
        print(f"Success! Model saved to: {model_path}")
        print("You can now run the app using the pre-trained model.")
        
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    train()
