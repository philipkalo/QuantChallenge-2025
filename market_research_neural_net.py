#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, LeakyReLU,
    Input, Add, Concatenate, GaussianNoise, LayerNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    LearningRateScheduler, TensorBoard
)
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras import backend as K

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


#Configuration
class Config:
    """Central configuration for the pipeline."""
    # Paths
    TRAIN_PATH = './data/train.csv'
    TEST_PATH = './data/test.csv'
    OUTPUT_DIR = './outputs'
    
    # Training
    EPOCHS = 500
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    MIN_LR = 1e-7
    
    # Cross-validation
    N_FOLDS = 5
    USE_KFOLD = True
    
    # Ensemble
    N_ENSEMBLE_MODELS = 3
    USE_ENSEMBLE = True
    
    # Feature engineering
    USE_POLYNOMIAL_FEATURES = True
    POLYNOMIAL_DEGREE = 2
    USE_FEATURE_SELECTION = True
    TOP_K_FEATURES = 30
    
    # Architecture
    ARCHITECTURE = 'residual'  # Options: 'simple', 'deep', 'residual', 'wide'
    DROPOUT_RATE = 0.3
    L2_REG = 1e-4
    
    # Early stopping
    PATIENCE = 30
    MIN_DELTA = 1e-5


#Data Loading and Feature Engineering
def load_data(config):
    """Load training and test data."""
    print("Loading data...")
    train_data = pd.read_csv(config.TRAIN_PATH)
    test_data = pd.read_csv(config.TEST_PATH)
    
    print(f"  Training samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")
    
    return train_data, test_data


def engineer_features(X, feature_names, config, fit=True, poly_features=None):
    """
    Create engineered features:
    - Polynomial features
    - Interaction terms
    - Statistical aggregations
    """
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Basic statistical features (row-wise)
    X_df['row_mean'] = X_df.mean(axis=1)
    X_df['row_std'] = X_df.std(axis=1)
    X_df['row_max'] = X_df.max(axis=1)
    X_df['row_min'] = X_df.min(axis=1)
    X_df['row_range'] = X_df['row_max'] - X_df['row_min']
    X_df['row_skew'] = X_df[feature_names].skew(axis=1)
    
    # Polynomial features (selected interactions)
    if config.USE_POLYNOMIAL_FEATURES:
        from sklearn.preprocessing import PolynomialFeatures
        
        if fit:
            poly = PolynomialFeatures(
                degree=config.POLYNOMIAL_DEGREE,
                include_bias=False,
                interaction_only=True  # Only interactions, not powers
            )
            # Use subset of features for polynomial to avoid explosion
            top_features = feature_names[:min(8, len(feature_names))]
            poly_data = poly.fit_transform(X_df[top_features])
            poly_feature_names = poly.get_feature_names_out(top_features)
            
            # Store for transform
            poly_features = {
                'poly': poly,
                'top_features': top_features,
                'names': poly_feature_names
            }
        else:
            poly = poly_features['poly']
            top_features = poly_features['top_features']
            poly_feature_names = poly_features['names']
            poly_data = poly.transform(X_df[top_features])
        
        # Add polynomial features
        poly_df = pd.DataFrame(poly_data, columns=poly_feature_names, index=X_df.index)
        # Only add new interaction columns (skip original features)
        new_poly_cols = [c for c in poly_df.columns if ' ' in c]
        X_df = pd.concat([X_df, poly_df[new_poly_cols]], axis=1)
    
    return X_df.values, list(X_df.columns), poly_features


def select_features(X, y, feature_names, config):
    """Select top features using mutual information."""
    print(f"\nSelecting top {config.TOP_K_FEATURES} features...")
    
    mi_scores = mutual_info_regression(X, y, random_state=SEED)
    mi_df = pd.DataFrame({
        'feature': feature_names,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    print("Top 10 features by mutual information:")
    print(mi_df.head(10).to_string(index=False))
    
    top_features = mi_df.head(config.TOP_K_FEATURES)['feature'].tolist()
    top_indices = [feature_names.index(f) for f in top_features]
    
    return top_indices, top_features


#Model Architectures
def build_simple_model(input_dim, config):
    """Simple feedforward network."""
    model = Sequential([
        Dense(128, input_shape=(input_dim,), kernel_regularizer=l2(config.L2_REG)),
        LeakyReLU(negative_slope=0.01),
        BatchNormalization(),
        Dropout(config.DROPOUT_RATE),
        
        Dense(64, kernel_regularizer=l2(config.L2_REG)),
        LeakyReLU(negative_slope=0.01),
        BatchNormalization(),
        Dropout(config.DROPOUT_RATE),
        
        Dense(32, kernel_regularizer=l2(config.L2_REG)),
        LeakyReLU(negative_slope=0.01),
        
        Dense(1, activation='linear')
    ], name='simple_model')
    
    return model


def build_deep_model(input_dim, config):
    """Deeper network with more layers."""
    model = Sequential([
        Dense(512, input_shape=(input_dim,), kernel_regularizer=l2(config.L2_REG)),
        LeakyReLU(negative_slope=0.01),
        BatchNormalization(),
        Dropout(config.DROPOUT_RATE),
        
        Dense(256, kernel_regularizer=l2(config.L2_REG)),
        LeakyReLU(negative_slope=0.01),
        BatchNormalization(),
        Dropout(config.DROPOUT_RATE),
        
        Dense(128, kernel_regularizer=l2(config.L2_REG)),
        LeakyReLU(negative_slope=0.01),
        BatchNormalization(),
        Dropout(config.DROPOUT_RATE),
        
        Dense(64, kernel_regularizer=l2(config.L2_REG)),
        LeakyReLU(negative_slope=0.01),
        BatchNormalization(),
        Dropout(config.DROPOUT_RATE * 0.5),
        
        Dense(32, kernel_regularizer=l2(config.L2_REG)),
        LeakyReLU(negative_slope=0.01),
        
        Dense(1, activation='linear')
    ], name='deep_model')
    
    return model


def build_residual_model(input_dim, config):
    """Network with residual/skip connections."""
    inputs = Input(shape=(input_dim,), name='input')
    
    # Initial projection
    x = Dense(256, kernel_regularizer=l2(config.L2_REG))(inputs)
    x = LeakyReLU(negative_slope=0.01)(x)
    x = BatchNormalization()(x)
    
    # Residual block 1
    residual = x
    x = Dense(256, kernel_regularizer=l2(config.L2_REG))(x)
    x = LeakyReLU(negative_slope=0.01)(x)
    x = BatchNormalization()(x)
    x = Dropout(config.DROPOUT_RATE)(x)
    x = Dense(256, kernel_regularizer=l2(config.L2_REG))(x)
    x = Add()([x, residual])
    x = LeakyReLU(negative_slope=0.01)(x)
    
    # Downsample
    x = Dense(128, kernel_regularizer=l2(config.L2_REG))(x)
    x = LeakyReLU(negative_slope=0.01)(x)
    x = BatchNormalization()(x)
    
    # Residual block 2
    residual = x
    x = Dense(128, kernel_regularizer=l2(config.L2_REG))(x)
    x = LeakyReLU(negative_slope=0.01)(x)
    x = BatchNormalization()(x)
    x = Dropout(config.DROPOUT_RATE)(x)
    x = Dense(128, kernel_regularizer=l2(config.L2_REG))(x)
    x = Add()([x, residual])
    x = LeakyReLU(negative_slope=0.01)(x)
    
    # Final layers
    x = Dense(64, kernel_regularizer=l2(config.L2_REG))(x)
    x = LeakyReLU(negative_slope=0.01)(x)
    x = BatchNormalization()(x)
    
    x = Dense(32, kernel_regularizer=l2(config.L2_REG))(x)
    x = LeakyReLU(negative_slope=0.01)(x)
    
    outputs = Dense(1, activation='linear', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='residual_model')
    return model


def build_wide_deep_model(input_dim, config):
    """Wide & Deep architecture - combines memorisation and generalisation."""
    inputs = Input(shape=(input_dim,), name='input')
    
    # Wide component (linear)
    wide = Dense(1, kernel_regularizer=l2(config.L2_REG), name='wide')(inputs)
    
    # Deep component
    deep = Dense(256, kernel_regularizer=l2(config.L2_REG))(inputs)
    deep = LeakyReLU(negative_slope=0.01)(deep)
    deep = BatchNormalization()(deep)
    deep = Dropout(config.DROPOUT_RATE)(deep)
    
    deep = Dense(128, kernel_regularizer=l2(config.L2_REG))(deep)
    deep = LeakyReLU(negative_slope=0.01)(deep)
    deep = BatchNormalization()(deep)
    deep = Dropout(config.DROPOUT_RATE)(deep)
    
    deep = Dense(64, kernel_regularizer=l2(config.L2_REG))(deep)
    deep = LeakyReLU(negative_slope=0.01)(deep)
    deep = BatchNormalization()(deep)
    
    deep = Dense(32, kernel_regularizer=l2(config.L2_REG))(deep)
    deep = LeakyReLU(negative_slope=0.01)(deep)
    
    deep = Dense(1, name='deep_output')(deep)
    
    # Combine wide and deep
    combined = Add()([wide, deep])
    
    model = Model(inputs=inputs, outputs=combined, name='wide_deep_model')
    return model


def get_model(input_dim, config, architecture=None):
    """Factory function to get model by architecture type."""
    arch = architecture or config.ARCHITECTURE
    
    builders = {
        'simple': build_simple_model,
        'deep': build_deep_model,
        'residual': build_residual_model,
        'wide': build_wide_deep_model
    }
    
    if arch not in builders:
        raise ValueError(f"Unknown architecture: {arch}. Choose from {list(builders.keys())}")
    
    return builders[arch](input_dim, config)


#Learning Rate Scheduling
def cosine_annealing_with_warmup(epoch, lr, warmup_epochs=10, total_epochs=500, min_lr=1e-7):
    """Cosine annealing with linear warmup."""
    if epoch < warmup_epochs:
        return lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + 0.5 * (lr - min_lr) * (1 + np.cos(np.pi * progress))


def get_lr_scheduler(config):
    """Get learning rate scheduler callback."""
    def scheduler(epoch, lr):
        return cosine_annealing_with_warmup(
            epoch, config.LEARNING_RATE,
            warmup_epochs=10,
            total_epochs=config.EPOCHS,
            min_lr=config.MIN_LR
        )
    return LearningRateScheduler(scheduler, verbose=0)


#Training Functions
def get_callbacks(config, model_name='model'):
    """Get training callbacks."""
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config.PATIENCE,
            min_delta=config.MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=15,
            min_lr=config.MIN_LR,
            verbose=1
        ),
        get_lr_scheduler(config)
    ]
    return callbacks


def train_single_model(X_train, y_train, X_val, y_val, config, 
                       architecture=None, model_idx=0, verbose=1):
    """Train a single model."""
    # Scale features
    scaler = RobustScaler()  # More robust to outliers than StandardScaler
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Build model
    input_dim = X_train_scaled.shape[1]
    model = get_model(input_dim, config, architecture)
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    
    # Train
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=get_callbacks(config),
        verbose=verbose
    )
    
    return model, scaler, history


def train_with_kfold(X, y, config, target_name='Y'):
    """Train with K-Fold cross-validation and return ensemble of models."""
    print(f"\n{'='*60}")
    print(f"K-FOLD TRAINING FOR {target_name} ({config.N_FOLDS} folds)")
    print(f"{'='*60}")
    
    kfold = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=SEED)
    
    models = []
    scalers = []
    histories = []
    fold_scores = []
    oof_predictions = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\n--- Fold {fold + 1}/{config.N_FOLDS} ---")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model, scaler, history = train_single_model(
            X_train, y_train, X_val, y_val, config,
            model_idx=fold, verbose=0
        )
        
        # Evaluate fold
        X_val_scaled = scaler.transform(X_val)
        val_pred = model.predict(X_val_scaled, verbose=0).flatten()
        oof_predictions[val_idx] = val_pred
        
        fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        fold_r2 = r2_score(y_val, val_pred)
        fold_scores.append({'rmse': fold_rmse, 'r2': fold_r2})
        
        print(f"  Fold {fold + 1} - RMSE: {fold_rmse:.4f}, R²: {fold_r2:.4f}")
        
        models.append(model)
        scalers.append(scaler)
        histories.append(history)
    
    # Overall OOF score
    oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
    oof_r2 = r2_score(y, oof_predictions)
    
    print(f"\n{target_name} Overall OOF Results:")
    print(f"  RMSE: {oof_rmse:.4f}")
    print(f"  R²:   {oof_r2:.4f}")
    print(f"  Mean Fold RMSE: {np.mean([s['rmse'] for s in fold_scores]):.4f} "
          f"(± {np.std([s['rmse'] for s in fold_scores]):.4f})")
    
    return models, scalers, histories, oof_predictions


def train_ensemble(X, y, config, target_name='Y'):
    """Train an ensemble of models with different architectures."""
    print(f"\n{'='*60}")
    print(f"ENSEMBLE TRAINING FOR {target_name} ({config.N_ENSEMBLE_MODELS} models)")
    print(f"{'='*60}")
    
    architectures = ['residual', 'deep', 'wide'][:config.N_ENSEMBLE_MODELS]
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    
    models = []
    scalers = []
    weights = []
    
    for i, arch in enumerate(architectures):
        print(f"\n--- Training {arch} model ({i + 1}/{len(architectures)}) ---")
        
        model, scaler, history = train_single_model(
            X_train, y_train, X_val, y_val, config,
            architecture=arch, model_idx=i, verbose=0
        )
        
        # Evaluate and compute weight based on validation performance
        X_val_scaled = scaler.transform(X_val)
        val_pred = model.predict(X_val_scaled, verbose=0).flatten()
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_r2 = r2_score(y_val, val_pred)
        
        print(f"  {arch} - RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
        
        # Weight inversely proportional to RMSE
        weight = 1.0 / (val_rmse + 1e-6)
        
        models.append(model)
        scalers.append(scaler)
        weights.append(weight)
    
    # Normalise weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    print(f"\nEnsemble weights: {dict(zip(architectures, weights.round(3)))}")
    
    return models, scalers, weights


#Prediction
def predict_kfold(models, scalers, X_test):
    """Make predictions using K-Fold ensemble (average of all fold models)."""
    predictions = []
    
    for model, scaler in zip(models, scalers):
        X_test_scaled = scaler.transform(X_test)
        pred = model.predict(X_test_scaled, verbose=0).flatten()
        predictions.append(pred)
    
    # Average predictions
    return np.mean(predictions, axis=0)


def predict_ensemble(models, scalers, weights, X_test):
    """Make weighted predictions using ensemble."""
    predictions = []
    
    for model, scaler in zip(models, scalers):
        X_test_scaled = scaler.transform(X_test)
        pred = model.predict(X_test_scaled, verbose=0).flatten()
        predictions.append(pred)
    
    # Weighted average
    predictions = np.array(predictions)
    return np.average(predictions, axis=0, weights=weights)


#Visualisation
def plot_training_history(histories, target_name, save_path=None):
    """Plot training histories for all folds/models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for i, history in enumerate(histories):
        axes[0].plot(history.history['loss'], alpha=0.7, label=f'Train (fold {i+1})')
        axes[0].plot(history.history['val_loss'], alpha=0.7, linestyle='--', 
                     label=f'Val (fold {i+1})')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title(f'{target_name} - Training Loss')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    for i, history in enumerate(histories):
        axes[1].plot(history.history['mae'], alpha=0.7, label=f'Train (fold {i+1})')
        axes[1].plot(history.history['val_mae'], alpha=0.7, linestyle='--',
                     label=f'Val (fold {i+1})')
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title(f'{target_name} - Mean Absolute Error')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training plot saved to {save_path}")
    
    plt.close()


def plot_predictions_vs_actual(y_true, y_pred, target_name, save_path=None):
    """Plot predicted vs actual values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=10)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                 'r--', linewidth=2, label='Perfect prediction')
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title(f'{target_name} - Predicted vs Actual')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_pred - y_true
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residual (Predicted - Actual)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{target_name} - Residual Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Prediction plot saved to {save_path}")
    
    plt.close()



#Main Execution Pipeline
def main():
    """Main execution pipeline."""
    print("="*70)
    print("ENHANCED DEEP LEARNING MARKET RESEARCH PREDICTION")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    config = Config()
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Load data
    train_data, test_data = load_data(config)
    
    # Prepare features
    feature_cols = [col for col in train_data.columns if col not in ['id', 'ID', 'Y1', 'Y2']]
    X = train_data[feature_cols].values
    Y1 = train_data['Y1'].values
    Y2 = train_data['Y2'].values
    X_test = test_data[feature_cols].values
    
    print(f"\nOriginal features: {len(feature_cols)}")
    
    # Feature engineering
    print("\n" + "="*60)
    print("FEATURE ENGINEERING")
    print("="*60)
    
    X_eng, eng_feature_names, poly_features = engineer_features(
        X, feature_cols, config, fit=True
    )
    X_test_eng, _, _ = engineer_features(
        X_test, feature_cols, config, fit=False, poly_features=poly_features
    )
    
    print(f"Engineered features: {len(eng_feature_names)}")
    
    # Feature selection for Y1
    if config.USE_FEATURE_SELECTION:
        top_indices_y1, top_features_y1 = select_features(
            X_eng, Y1, eng_feature_names, config
        )
        X_y1 = X_eng[:, top_indices_y1]
        X_test_y1 = X_test_eng[:, top_indices_y1]
        
        top_indices_y2, top_features_y2 = select_features(
            X_eng, Y2, eng_feature_names, config
        )
        X_y2 = X_eng[:, top_indices_y2]
        X_test_y2 = X_test_eng[:, top_indices_y2]
    else:
        X_y1 = X_y2 = X_eng
        X_test_y1 = X_test_y2 = X_test_eng
    
    # Train models
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    if config.USE_KFOLD:
        # K-Fold training
        models_y1, scalers_y1, histories_y1, oof_y1 = train_with_kfold(
            X_y1, Y1, config, target_name='Y1'
        )
        models_y2, scalers_y2, histories_y2, oof_y2 = train_with_kfold(
            X_y2, Y2, config, target_name='Y2'
        )
        
        # Predictions
        pred_y1 = predict_kfold(models_y1, scalers_y1, X_test_y1)
        pred_y2 = predict_kfold(models_y2, scalers_y2, X_test_y2)
        
        # Plot OOF predictions
        plot_predictions_vs_actual(Y1, oof_y1, 'Y1', 
                                   save_path=f'{config.OUTPUT_DIR}/y1_oof_predictions.png')
        plot_predictions_vs_actual(Y2, oof_y2, 'Y2',
                                   save_path=f'{config.OUTPUT_DIR}/y2_oof_predictions.png')
        
    elif config.USE_ENSEMBLE:
        # Ensemble training
        models_y1, scalers_y1, weights_y1 = train_ensemble(
            X_y1, Y1, config, target_name='Y1'
        )
        models_y2, scalers_y2, weights_y2 = train_ensemble(
            X_y2, Y2, config, target_name='Y2'
        )
        
        # Predictions
        pred_y1 = predict_ensemble(models_y1, scalers_y1, weights_y1, X_test_y1)
        pred_y2 = predict_ensemble(models_y2, scalers_y2, weights_y2, X_test_y2)
        
        histories_y1 = histories_y2 = []
    
    else:
        # Simple train/val split
        X_train, X_val, Y1_train, Y1_val, Y2_train, Y2_val = train_test_split(
            X_y1, Y1, Y2, test_size=0.2, random_state=SEED
        )
        
        model_y1, scaler_y1, history_y1 = train_single_model(
            X_train, Y1_train, X_val, Y1_val, config
        )
        model_y2, scaler_y2, history_y2 = train_single_model(
            X_train, Y2_train, X_val, Y2_val, config
        )
        
        # Predictions
        pred_y1 = model_y1.predict(scaler_y1.transform(X_test_y1), verbose=0).flatten()
        pred_y2 = model_y2.predict(scaler_y2.transform(X_test_y2), verbose=0).flatten()
        
        histories_y1 = [history_y1]
        histories_y2 = [history_y2]
    
    # Plot training histories
    if histories_y1:
        plot_training_history(histories_y1, 'Y1', 
                              save_path=f'{config.OUTPUT_DIR}/y1_training_history.png')
    if histories_y2:
        plot_training_history(histories_y2, 'Y2',
                              save_path=f'{config.OUTPUT_DIR}/y2_training_history.png')
    
    # Create submission
    print("\n" + "="*60)
    print("GENERATING PREDICTIONS")
    print("="*60)
    
    preds = pd.DataFrame({
        'id': test_data['id'],
        'Y1': pred_y1,
        'Y2': pred_y2
    })
    
    print("\nPrediction summary:")
    print(preds.describe())
    print(f"\nFirst 10 predictions:")
    print(preds.head(10))
    
    # Save predictions
    output_file = f'{config.OUTPUT_DIR}/preds_enhanced_dl.csv'
    preds.to_csv(output_file, index=False)
    print(f"\nPredictions saved to '{output_file}'")
    
    # Also save to root for easy submission
    preds.to_csv('preds.csv', index=False)
    print("Also saved to 'preds.csv' for submission")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nSubmit to: https://quantchallenge.org/dashboard/data/upload-predictions")
    
    return preds


#Entry Point
if __name__ == "__main__":
    predictions = main()
