<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">California Housing Price Prediction Project</div>

This project implements a complete machine learning pipeline for predicting California housing prices using various regression models. The project is structured as a modular system with clear separation of concerns between data preparation, model training, and evaluation.

# 1. Project Overview

## 1.1 Purpose
This project demonstrates a complete data science workflow for predicting California housing prices using machine learning. It includes:
- Data preprocessing and feature engineering
- Multiple model implementations and comparison
- Model evaluation and testing
- Visualization of results
- Production-ready model deployment

## 1.2 Dataset
The project uses the California Housing dataset from scikit-learn, which contains:
- 8 features describing housing characteristics
- Target variable: Median house value
- 20,640 samples
- Features include:
  - MedInc: Median income
  - HouseAge: House age
  - AveRooms: Average rooms
  - AveBedrms: Average bedrooms
  - Population: Neighborhood population
  - AveOccup: Average occupancy
  - Latitude: Location latitude
  - Longitude: Location longitude

# 2. Project Structure

## 2.1 Core Scripts

### data_prep.py
- **Purpose**: Data loading and initial preprocessing
- **Key Functions**:
  - Loads California Housing dataset
  - Sets up display options
  - Creates initial DataFrame
- **Dependencies**: pandas, numpy, matplotlib, seaborn, scikit-learn

### ml_models.py
- **Purpose**: Model training and evaluation
- **Key Functions**:
  - `load_data()`: Loads and prepares dataset
  - `preprocess_data()`: Handles data splitting and scaling
  - `train_and_evaluate_models()`: Trains multiple models and evaluates performance
  - `plot_model_comparison()`: Visualizes model performance
- **Models Implemented**:
  - Linear Regression
  - Random Forest
  - Support Vector Regression (SVR)
  - XGBoost
- **Outputs**:
  - Trained models (saved as .joblib files)
  - Performance metrics
  - Comparison plots
  - Feature importance visualization

### test_model.py
- **Purpose**: Model testing and prediction
- **Key Functions**:
  - `load_model_and_scaler()`: Loads trained model and scaler
  - `prepare_new_data()`: Generates realistic test data
  - `predict_house_values()`: Makes predictions and analyzes results
- **Features**:
  - Realistic data generation
  - Comprehensive prediction analysis
  - Detailed results output
  - Error handling

## 2.2 Directory Structure
```
project/
├── scripts/
│   ├── data_prep.py
│   ├── ml_models.py
│   ├── test_model.py
│   └── README.md
├── figures/
│   ├── model_comparison.png
│   └── feature_importance.png
└── results/
    └── test_predictions.csv
```

# 3. Usage Guide

## 3.1 Setup
1. Install required dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost joblib
```

2. Create necessary directories:
```bash
mkdir -p models figures results
```

## 3.2 Running the Pipeline

1. Data Preparation:
```bash
python scripts/data_prep.py
```

2. Model Training:
```bash
python scripts/ml_models.py
```

3. Model Testing:
```bash
python scripts/test_model.py
```

## 3.3 Expected Outputs
- Trained models in `models/` directory
- Performance visualizations in `figures/` directory
- Prediction results in `results/` directory

# 4. Technical Details

## 4.1 Data Preprocessing
- Feature scaling using StandardScaler
- Train-test split (80-20)
- Data validation and cleaning

## 4.2 Model Implementation
- Multiple regression models for comparison
- Cross-validation for robust evaluation
- Hyperparameter tuning (where applicable)
- Feature importance analysis

## 4.3 Evaluation Metrics
- Mean Squared Error (MSE)
- R-squared (R²) score
- Visual comparison of model performance
- Feature importance visualization

# 5. Best Practices

## 5.1 Code Organization
- Modular design
- Clear function documentation
- Proper error handling
- Logging implementation

## 5.2 Model Management
- Model versioning
- Scalability considerations
- Production readiness
- Performance optimization

## 5.3 Documentation
- Comprehensive docstrings
- Clear README
- Usage examples
- Maintenance guidelines

# 6. Future Improvements

## 6.1 Planned Enhancements
- Hyperparameter optimization
- Additional model architectures
- Advanced feature engineering
- API integration
- Real-time prediction capabilities

## 6.2 Scalability Considerations
- Batch processing support
- Distributed computing
- Cloud deployment
- Model monitoring

# 7. Contributing

## 7.1 Guidelines
- Follow PEP 8 style guide
- Write unit tests
- Document all changes
- Use meaningful commit messages

## 7.2 Development Workflow
1. Create feature branch
2. Implement changes
3. Run tests
4. Update documentation
5. Create pull request

# 8. License
This project is licensed under the MIT License - see the LICENSE file for details.

## Model Files
- `ml_models.py`: Contains functions for training and evaluating machine learning models
- `test_model.py`: Contains functions for testing trained models on new data

Note: Models are not stored in the repository. They can be trained using the provided scripts. 