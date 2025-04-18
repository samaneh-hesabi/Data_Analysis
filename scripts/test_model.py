import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

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

def test_model(model, scaler, test_data):
    """
    Test the trained model on new data
    
    Parameters:
    -----------
    model : sklearn model
        Trained model to test
    scaler : sklearn scaler
        Fitted scaler for feature transformation
    test_data : pd.DataFrame
        Test data to evaluate the model on
    
    Returns:
    --------
    dict
        Dictionary containing test metrics
    """
    # Scale the features
    scaled_data = scaler.transform(test_data)
    
    # Make predictions
    predictions = model.predict(scaled_data)
    
    # Calculate metrics
    mse = mean_squared_error(test_data['MedHouseVal'], predictions)
    r2 = r2_score(test_data['MedHouseVal'], predictions)
    
    print(f"Test Results:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    return {
        'mse': mse,
        'r2': r2,
        'predictions': predictions
    }

if __name__ == "__main__":
    # Example usage
    from ml_models import load_data, preprocess_data, train_and_evaluate_models
    
    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Train models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Test with new data
    new_data = prepare_new_data(n_samples=100)
    test_results = test_model(results['Random Forest'], scaler, new_data) 