import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path

# Set up project directories
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
FIGURES_DIR = PROJECT_ROOT / 'figures'

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Load and prepare data
def load_data():
    from sklearn.datasets import fetch_california_housing
    california = fetch_california_housing()
    df = pd.DataFrame(california.data, columns=california.feature_names)
    df['MedHouseVal'] = california.target
    return df

# Preprocess data
def preprocess_data(df):
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    scaler_path = MODELS_DIR / 'scaler.joblib'
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Train and evaluate models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf'),
        'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'MSE': mse,
            'R2': r2
        }
        
        # Print results
        print(f"\n{name} Results:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        
        # Save the best performing model (Random Forest)
        if name == 'Random Forest':
            model_path = MODELS_DIR / 'random_forest_model.joblib'
            joblib.dump(model, model_path)
            print(f"Saved Random Forest model to {model_path}")
    
    return results

# Plot model comparison
def plot_model_comparison(results):
    # Create a DataFrame from results
    df_results = pd.DataFrame(results).T
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot MSE
    sns.barplot(x=df_results.index, y='MSE', data=df_results, ax=ax1)
    ax1.set_title('Mean Squared Error Comparison')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    ax1.set_ylabel('MSE (Lower is Better)')
    
    # Plot R2 Score
    sns.barplot(x=df_results.index, y='R2', data=df_results, ax=ax2)
    ax2.set_title('R2 Score Comparison')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    ax2.set_ylabel('R2 Score (Higher is Better)')
    
    # Adjust layout and save
    plt.tight_layout()
    comparison_path = FIGURES_DIR / 'model_comparison.png'
    plt.savefig(comparison_path)
    plt.close()
    print(f"Saved model comparison plot to {comparison_path}")

# Main function
def main():
    print("Starting ML model training and evaluation...")
    
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Train and evaluate models
    print("\nTraining and evaluating models...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Plot model comparison
    print("\nGenerating model comparison plots...")
    plot_model_comparison(results)
    
    # Plot feature importance for Random Forest
    print("\nGenerating feature importance plot...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    feature_importance = pd.DataFrame({
        'Feature': df.drop('MedHouseVal', axis=1).columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance (Random Forest)')
    plt.tight_layout()
    
    importance_path = FIGURES_DIR / 'feature_importance.png'
    plt.savefig(importance_path)
    plt.close()
    print(f"Saved feature importance plot to {importance_path}")
    
    print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    main() 