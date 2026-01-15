#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')


#Data Loading
def load_data(train_path='./data/train.csv', test_path='./data/test.csv'):
    """Load training and test datasets."""
    print("Loading data...")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    print(f"  Training samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")
    print(f"\nTraining data preview:")
    print(train_data.head())
    print(f"\nData info:")
    print(train_data.info())
    
    return train_data, test_data


#Exploratory Data Analysis
def explore_correlations(data, save_plots=True):
    """Analyse correlations between features and targets."""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Separate features and targets
    feature_cols = [col for col in data.columns if col not in ['Y1', 'Y2']]
    X_features = data[feature_cols]
    Y1 = data['Y1']
    Y2 = data['Y2']
    
    # Calculate correlations with Y1
    correlations_y1 = X_features.corrwith(Y1)
    print("\nCorrelations with Y1:")
    print(correlations_y1.sort_values(ascending=False))
    
    # Calculate correlations with Y2
    correlations_y2 = X_features.corrwith(Y2)
    print("\nCorrelations with Y2:")
    print(correlations_y2.sort_values(ascending=False))
    
    if save_plots:
        # Heatmap for Y1 correlations
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        
        sns.heatmap(correlations_y1.to_frame(name='Y1'), annot=True, cmap="coolwarm", 
                    ax=axes[0], center=0)
        axes[0].set_title('Feature Correlations with Y1')
        
        sns.heatmap(correlations_y2.to_frame(name='Y2'), annot=True, cmap="coolwarm", 
                    ax=axes[1], center=0)
        axes[1].set_title('Feature Correlations with Y2')
        
        plt.tight_layout()
        plt.savefig('correlation_heatmaps.png', dpi=150, bbox_inches='tight')
        print("\nCorrelation heatmaps saved to 'correlation_heatmaps.png'")
        plt.close()
        
        # Scatter plot example: H vs Y1 (typically high correlation)
        plt.figure(figsize=(10, 6))
        plt.scatter(data['H'], Y1, alpha=0.5, s=10)
        plt.xlabel('H')
        plt.ylabel('Y1')
        plt.title('Relationship between H and Y1')
        plt.grid(True, alpha=0.3)
        plt.savefig('h_vs_y1_scatter.png', dpi=150, bbox_inches='tight')
        print("Scatter plot saved to 'h_vs_y1_scatter.png'")
        plt.close()
    
    return correlations_y1, correlations_y2


#Feature Selection
def select_features_by_correlation(data, correlations_y1, correlations_y2, 
                                   top_n=7, threshold=0.1):
    """
    Select top features based on correlation with targets.
    
    Original notebook used hardcoded indices:
    - Y1 features: columns [8, 14, 13, 3, 10, 5, 7] -> I, N (or O), M (or N), D, K, F, H
    - Y2 features: columns [1, 2, 4, 6, 9, 11, 12] -> A (or B), B (or C), E, G, J, L, M
    
    This function selects features dynamically based on correlation strength.
    """
    print("\n" + "="*60)
    print("FEATURE SELECTION")
    print("="*60)
    
    # Get feature columns (exclude targets)
    feature_cols = [col for col in data.columns if col not in ['Y1', 'Y2']]
    
    # Select top features for Y1 (by absolute correlation)
    top_features_y1 = correlations_y1.abs().sort_values(ascending=False).head(top_n).index.tolist()
    
    # Select top features for Y2 (by absolute correlation)
    top_features_y2 = correlations_y2.abs().sort_values(ascending=False).head(top_n).index.tolist()
    
    print(f"\nTop {top_n} features for Y1: {top_features_y1}")
    print(f"Top {top_n} features for Y2: {top_features_y2}")
    
    return top_features_y1, top_features_y2


#Model Training
def train_random_forest(X_train, y_train, X_test, y_test, 
                        n_estimators=200, max_depth=None, random_state=42,
                        target_name='Y'):
    """
    Train a Random Forest Regressor and evaluate performance.
    
    Parameters:
        X_train, y_train: Training data
        X_test, y_test: Test data for evaluation
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees (None for unlimited)
        random_state: Random seed for reproducibility
        target_name: Name of target variable for logging
    
    Returns:
        model: Trained RandomForestRegressor
        metrics: Dictionary of evaluation metrics
    """
    print(f"\nTraining Random Forest for {target_name}...")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1  # Use all CPU cores
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"  {target_name} - R² Score: {r2:.4f}")
    print(f"  {target_name} - RMSE: {rmse:.4f}")
    
    # Feature importances
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n  Feature importances for {target_name}:")
    print(feature_importance.to_string(index=False))
    
    metrics = {'r2': r2, 'rmse': rmse}
    
    return model, metrics


#Main Execution Pipeline
def main():
    """Main execution pipeline."""
    print("="*60)
    print("RANDOM FOREST MARKET RESEARCH PREDICTION")
    print("="*60)
    
    # Load data
    train_data, test_data = load_data()
    
    # Explore correlations
    correlations_y1, correlations_y2 = explore_correlations(train_data, save_plots=True)
    
    # Shuffle data
    train_data_shuffled = shuffle(train_data, random_state=42)
    
    # Select features
    features_y1, features_y2 = select_features_by_correlation(
        train_data_shuffled, correlations_y1, correlations_y2, top_n=7
    )
    
    # Prepare feature matrices
    X_y1 = train_data_shuffled[features_y1]
    X_y2 = train_data_shuffled[features_y2]
    Y1 = train_data_shuffled['Y1']
    Y2 = train_data_shuffled['Y2']
    
    # Train/test split
    X_y1_train, X_y1_test, Y1_train, Y1_test = train_test_split(
        X_y1, Y1, test_size=0.2, random_state=42
    )
    X_y2_train, X_y2_test, Y2_train, Y2_test = train_test_split(
        X_y2, Y2, test_size=0.2, random_state=42
    )
    
    # Train models
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    model_y1, metrics_y1 = train_random_forest(
        X_y1_train, Y1_train, X_y1_test, Y1_test,
        n_estimators=200, target_name='Y1'
    )
    
    model_y2, metrics_y2 = train_random_forest(
        X_y2_train, Y2_train, X_y2_test, Y2_test,
        n_estimators=200, target_name='Y2'
    )
    
    # Generate predictions on test set
    print("\n" + "="*60)
    print("GENERATING PREDICTIONS")
    print("="*60)
    
    # Prepare test features
    X_test_y1 = test_data[features_y1]
    X_test_y2 = test_data[features_y2]
    
    # Predict
    predictions_y1 = model_y1.predict(X_test_y1)
    predictions_y2 = model_y2.predict(X_test_y2)
    
    # Create submission dataframe
    preds = pd.DataFrame({
        'id': test_data['id'],
        'Y1': predictions_y1,
        'Y2': predictions_y2
    })
    
    print("\nPrediction summary:")
    print(preds.describe())
    print(f"\nFirst 10 predictions:")
    print(preds.head(10))
    
    # Save predictions
    output_file = 'preds.csv'
    preds.to_csv(output_file, index=False)
    print(f"\nPredictions saved to '{output_file}'")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"\nY1 Model - R²: {metrics_y1['r2']:.4f}, RMSE: {metrics_y1['rmse']:.4f}")
    print(f"Y2 Model - R²: {metrics_y2['r2']:.4f}, RMSE: {metrics_y2['rmse']:.4f}")
    print("\nSubmit preds.csv to: https://quantchallenge.org/dashboard/data/upload-predictions")
    
    return preds, model_y1, model_y2

#Entry point
if __name__ == "__main__":
    predictions, model_y1, model_y2 = main()
