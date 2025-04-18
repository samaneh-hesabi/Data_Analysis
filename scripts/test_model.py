import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
from pathlib import Path

def load_model_and_scaler():
    """Load the trained model and scaler"""
    # Use proper path handling
    model_path = Path('models/random_forest_model.joblib')
    scaler_path = Path('models/scaler.joblib')
    
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Model or scaler files not found. Please train the model first.")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def prepare_new_data(n_samples=100):
    """Create a realistic new dataset for testing
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    
    Returns:
    --------
    pd.DataFrame
        New test dataset with realistic feature distributions
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate realistic data with reasonable ranges
    new_data = pd.DataFrame({
        # Median income (in tens of thousands)
        'MedInc': np.random.normal(loc=5.0, scale=2.0, size=n_samples).clip(0.5, 15.0),
        
        # House age (years)
        'HouseAge': np.random.uniform(low=1, high=60, size=n_samples).astype(int),
        
        # Average rooms (typically 3-10 rooms)
        'AveRooms': np.random.normal(loc=5.5, scale=1.5, size=n_samples).clip(3, 10),
        
        # Average bedrooms (typically 1-4 bedrooms)
        'AveBedrms': np.random.normal(loc=2.5, scale=0.5, size=n_samples).clip(1, 4),
        
        # Population (neighborhood population)
        'Population': np.random.normal(loc=1500, scale=500, size=n_samples).clip(100, 5000),
        
        # Average occupancy (people per household)
        'AveOccup': np.random.normal(loc=3.0, scale=0.5, size=n_samples).clip(1.5, 5.0),
        
        # Latitude (California ranges roughly from 32.5 to 42.0)
        'Latitude': np.random.uniform(low=32.5, high=42.0, size=n_samples),
        
        # Longitude (California ranges roughly from -124.5 to -114.0)
        'Longitude': np.random.uniform(low=-124.5, high=-114.0, size=n_samples)
    })
    
    # Round numerical values to reasonable decimals
    new_data = new_data.round({
        'MedInc': 2,
        'AveRooms': 1,
        'AveBedrms': 1,
        'Population': 0,
        'AveOccup': 2,
        'Latitude': 3,
        'Longitude': 3
    })
    
    return new_data

def predict_house_values():
    """Make predictions on new data and analyze results"""
    try:
        # Load model and scaler
        print("Loading model and scaler...")
        model, scaler = load_model_and_scaler()
        
        # Prepare new data
        print("\nGenerating new test data...")
        new_data = prepare_new_data(n_samples=100)
        
        # Scale the features
        print("Scaling features...")
        scaled_data = scaler.transform(new_data)
        
        # Make predictions
        print("Making predictions...")
        predictions = model.predict(scaled_data)
        
        # Add predictions to the dataset
        results = new_data.copy()
        results['Predicted_House_Value'] = predictions
        
        # Print summary statistics
        print("\nPrediction Summary Statistics:")
        print("=" * 50)
        print(f"Number of samples: {len(predictions)}")
        print(f"Average predicted house value: ${predictions.mean():.2f} (in $100,000s)")
        print(f"Min predicted house value: ${predictions.min():.2f} (in $100,000s)")
        print(f"Max predicted house value: ${predictions.max():.2f} (in $100,000s)")
        
        # Print detailed results for first 5 samples
        print("\nDetailed Results (First 5 Samples):")
        print("=" * 50)
        for idx in range(5):
            print(f"\nSample {idx + 1}:")
            print(f"Predicted House Value: ${predictions[idx]:.2f} (in $100,000s)")
            print("Features:")
            for feature in new_data.columns:
                print(f"  {feature}: {new_data.iloc[idx][feature]}")
        
        # Save results to CSV
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / 'test_predictions.csv'
        results.to_csv(output_path, index=False)
        print(f"\nSaved detailed results to: {output_path}")
        
        return results
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

if __name__ == "__main__":
    predict_house_values() 