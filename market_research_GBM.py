

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform as sp_uniform
from scipy.stats import randint as sp_randint
import warnings

warnings.filterwarnings('ignore')


#Load Data
def load_data(train_path='./data/train.csv', test_path='./data/test.csv'):
    """Load and prepare the training and test datasets."""
    print("Loading data...")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print("\nTraining data preview:")
    print(train_data.head())
    
    return train_data, test_data


# Exploratory Data Analysis
def explore_data(train_data, save_plots=True):
    """Perform exploratory data analysis and visualise correlations."""
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Scatter plot: C vs Y1
    plt.figure(figsize=(10, 6))
    plt.scatter(train_data['C'], train_data['Y1'], alpha=0.5)
    plt.xlabel('C')
    plt.ylabel('Y1')
    plt.title('Relationship between C and Y1')
    if save_plots:
        plt.savefig('c_vs_y1_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Calculate and display correlation
    correlation = train_data['C'].corr(train_data['Y1'])
    print(f"\nCorrelation between C and Y1: {correlation:.4f}")
    
    # Correlation matrix
    correlation_matrix = train_data.corr()
    plt.figure(figsize=(12, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    if save_plots:
        plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Display Y1 and Y2 statistics
    print(f"\nY1 - Mean: {train_data['Y1'].mean():.4f}, Std: {train_data['Y1'].std():.4f}")
    print(f"Y2 - Mean: {train_data['Y2'].mean():.4f}, Std: {train_data['Y2'].std():.4f}")
    
    return correlation_matrix


# Model Training with hyperparameter tuning
def train_lgbm_model(X_train, Y_train, X_test, Y_test, target_name='Y'):
    """
    Train a LightGBM regressor with RandomizedSearchCV for hyperparameter tuning.
    
    Parameters:
        X_train, Y_train: Training features and target
        X_test, Y_test: Test features and target for evaluation
        target_name: Name of target variable for logging
    
    Returns:
        best_model: Trained model with best hyperparameters
        best_params: Dictionary of best hyperparameters
    """
    print(f"\nTraining LightGBM model for {target_name}...")
    
    # Initial model
    lgbm_regressor = lgb.LGBMRegressor(
        objective='regression',
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=100,
        verbose=-1
    )
    
    # Hyperparameter search space
    param_grid = {
        'n_estimators': sp_randint(50, 1000),
        'max_depth': sp_randint(3, 30),
        'learning_rate': sp_uniform(0.01, 0.5),
        'num_leaves': sp_randint(10, 150)
    }
    
    # Randomized search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=lgbm_regressor,
        param_distributions=param_grid,
        n_iter=100,
        scoring='neg_root_mean_squared_error',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, Y_train)
    
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    # Evaluate on test set
    predictions = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, predictions))
    
    print(f"\nBest parameters for {target_name}: {best_params}")
    print(f"Best CV RMSE: {-random_search.best_score_:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    
    return best_model, best_params


# Main Execution Pipeline
def main():
    """Main execution pipeline."""
    print("="*60)
    print("MARKET RESEARCH PREDICTION PIPELINE")
    print("="*60)
    
    # Load data
    train_data, test_data = load_data()
    
    # Explore data
    correlation_matrix = explore_data(train_data, save_plots=True)
    
    # Prepare features
    features = [col for col in train_data.columns if col not in ['ID', 'id', 'Y1', 'Y2']]
    X = train_data[features]
    Y1 = train_data['Y1']
    Y2 = train_data['Y2']
    
    print(f"\nFeatures used: {features}")
    print(f"Number of features: {len(features)}")
    
    # Split data for Y1
    X_train, X_test, Y1_train, Y1_test = train_test_split(
        X, Y1, test_size=0.2, random_state=42
    )
    
    # Split data for Y2 (same split for consistency)
    _, _, Y2_train, Y2_test = train_test_split(
        X, Y2, test_size=0.2, random_state=42
    )
    
    # Train models
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    best_model_y1, params_y1 = train_lgbm_model(
        X_train, Y1_train, X_test, Y1_test, target_name='Y1'
    )
    
    best_model_y2, params_y2 = train_lgbm_model(
        X_train, Y2_train, X_test, Y2_test, target_name='Y2'
    )
    
    # Generate predictions on test set
    print("\n" + "="*60)
    print("GENERATING PREDICTIONS")
    print("="*60)
    
    test_features = test_data[features]
    prediction_y1 = best_model_y1.predict(test_features)
    prediction_y2 = best_model_y2.predict(test_features)
    
    # Create submission dataframe
    preds = pd.DataFrame({
        'id': test_data['id'],
        'Y1': prediction_y1,
        'Y2': prediction_y2
    })
    
    print("\nPrediction summary:")
    print(preds.describe())
    print(f"\nPredictions shape: {preds.shape}")
    print(preds.head(10))
    
    # Save predictions
    output_file = 'preds.csv'
    preds.to_csv(output_file, index=False)
    print(f"\nPredictions saved to '{output_file}'")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print("\nSubmit preds.csv to: https://quantchallenge.org/dashboard/data/upload-predictions")
    
    return preds, best_model_y1, best_model_y2


# Entry point
if __name__ == "__main__":
    predictions, model_y1, model_y2 = main()
