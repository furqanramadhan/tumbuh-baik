from pymongo import MongoClient
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import itertools
import warnings
import random
import gc
import os
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ======================================================
# Parameter Categories
# ======================================================
RAINFALL_PARAMS = ["RR", "RR_imputed", "PRECTOTCORR"]
NDVI_PARAMS = ["NDVI", "NDVI_imputed"]
TEMP_PARAMS = ["TAVG", "TMAX", "TMIN", "T2M"]
HUMIDITY_PARAMS = ["RH_AVG", "RH_AVG_preprocessed", 'RH2M']
SOLAR_PARAMS = ["ALLSKY_SFC_SW_DWN", "SRAD", "GHI"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Using device: {device}")

# ======================================================
# Global Configuration
# ======================================================
TRAIN_SPLIT_RATIO = 0.8
FAST_MODE = os.getenv("LSTM_FAST_MODE", "1").lower() in ["1", "true", "yes", "on"]
GRID_EPOCHS = int(os.getenv("LSTM_GRID_EPOCHS", "15" if FAST_MODE else "50"))
FINAL_EPOCHS = int(os.getenv("LSTM_FINAL_EPOCHS", "30" if FAST_MODE else "100"))
GRID_PATIENCE = int(os.getenv("LSTM_GRID_PATIENCE", "5" if FAST_MODE else "15"))
FINAL_PATIENCE = int(os.getenv("LSTM_FINAL_PATIENCE", "8" if FAST_MODE else "20"))
BACKTEST_HORIZON = int(os.getenv("LSTM_BACKTEST_HORIZON", "365"))

# ======================================================
# Transformation Helper Functions
# ======================================================
def is_rainfall_param(param_name):
    return param_name in RAINFALL_PARAMS or any(
        x in param_name.lower() for x in ['rr', 'rain', 'precip', 'hujan']
    )

def is_continuous_weather_param(param_name):
    return (
        param_name in TEMP_PARAMS
        or param_name in HUMIDITY_PARAMS
        or param_name in SOLAR_PARAMS
    )

def prepare_cleaned_series(df, target_column):
    """
    Prepare a continuous daily series for LSTM without re-running full
    preprocessing. Cleaned collections are treated as source of truth; this
    only fills missing calendar rows needed by sequence models.
    """
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found")

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    df = df[~df.index.duplicated(keep='first')]

    series = pd.to_numeric(df[target_column], errors='coerce')
    date_range = pd.date_range(start=series.index[0], end=series.index[-1], freq='D')
    missing_dates = date_range.difference(series.index)
    print(f"   Missing dates after cleaned input: {len(missing_dates)}")

    series = series.reindex(date_range)

    if is_rainfall_param(target_column):
        series = series.fillna(0)
    elif is_continuous_weather_param(target_column):
        series = series.interpolate(method='time', limit_direction='both')
    else:
        series = series.dropna()

    return series.dropna()

def apply_transformation(data, param_name):
    """Apply Log1p transformation for rainfall data specifically."""
    if is_rainfall_param(param_name):
        print(f"   ⚡ Applying Log1p Transformation for {param_name}")
        return np.log1p(data), "log1p"
    
    return data, "none"

def inverse_transformation(data, method):
    """Inverse transformation to original scale."""
    if method == "log1p":
        return np.maximum(0, np.expm1(data))
    return data

def estimate_log_bias_correction(
    actual_physical,
    predicted_transformed,
    transform_method,
    param_name=None,
    wet_day_threshold=1.0
):
    """
    Estimate median residual correction in transformed log space.
    Median correction is intentionally used instead of Duan smearing so rainfall
    zeros and extremes do not create unstable exp residuals.
    """
    default_result = {
        "correction": 0.0,
        "raw_correction": 0.0,
        "mode": "none",
        "sample_count": 0,
        "wet_day_threshold": None,
        "negative_clamped": False,
        "positive_capped": False
    }

    if transform_method != "log1p":
        return default_result

    actual_physical = np.asarray(actual_physical, dtype=float)
    actual_log = np.log1p(np.asarray(actual_physical, dtype=float))
    predicted_log = np.asarray(predicted_transformed, dtype=float)
    mask = np.isfinite(actual_log) & np.isfinite(predicted_log)
    if not mask.any():
        return default_result

    correction_mask = mask
    mode = "median_all_days"
    if param_name and is_rainfall_param(param_name):
        wet_mask = mask & (actual_physical >= wet_day_threshold)
        if wet_mask.sum() >= 10:
            correction_mask = wet_mask
            mode = "median_wet_days"
        else:
            mode = "median_all_days_insufficient_wet_days"

    residuals = np.clip(actual_log[correction_mask] - predicted_log[correction_mask], -5, 5)
    if len(residuals) == 0:
        return default_result

    raw_correction = float(np.median(residuals))
    correction = raw_correction if np.isfinite(raw_correction) else 0.0
    negative_clamped = False
    positive_capped = False

    if param_name and is_rainfall_param(param_name):
        # Do not let dry-day dominance push rainfall forecasts lower for planting rules.
        if correction < 0:
            correction = 0.0
            negative_clamped = True
        if correction > 1.0:
            correction = 1.0
            positive_capped = True

    return {
        "correction": correction if np.isfinite(correction) else 0.0,
        "raw_correction": raw_correction if np.isfinite(raw_correction) else 0.0,
        "mode": mode,
        "sample_count": int(correction_mask.sum()),
        "wet_day_threshold": wet_day_threshold if param_name and is_rainfall_param(param_name) else None,
        "negative_clamped": negative_clamped,
        "positive_capped": positive_capped
    }

def apply_log_bias_correction(values, transform_method, bias_correction=0.0):
    values = np.asarray(values, dtype=float)
    if transform_method == "log1p" and bias_correction:
        return values + float(bias_correction)
    return values

# ======================================================
# Seasonal Decomposition Helper
# ======================================================
def apply_seasonal_decomposition(series, period=365):
    """
    Decompose the series into Trend, Seasonal, and Residual.
    Returns the Deseasonalized series (Trend + Residual) and the Seasonal component.
    """
    if len(series) < (period * 2):
        print("   ⚠️ Data too short for seasonal decomposition. Skipping.")
        return series, None, None

    print(f"   🍂 Applying STL Decomposition (period={period}, robust=True)...")
    decomposition = STL(series, period=period, robust=True).fit()
    
    seasonal = decomposition.seasonal
    # Deseasonalized = Original - Seasonal (Data bersih dari pola tahunan untuk dilatih LSTM)
    deseasonalized = series - seasonal
    
    # Fill NaN just in case
    deseasonalized = deseasonalized.ffill().bfill()
    
    return deseasonalized, seasonal, decomposition

def save_lstm_decomposition(db, decomposition, observed_series, collection_name, target_column, config_id):
    """Persist STL decomposition into temp collection for run_lstm aggregation."""
    if decomposition is None:
        return 0

    saved_count = 0
    for idx in observed_series.index:
        date_str = pd.Timestamp(idx).strftime("%Y-%m-%d")
        param_doc = {
            "observed": float(observed_series.loc[idx]) if pd.notna(observed_series.loc[idx]) else None,
            "trend": float(decomposition.trend.loc[idx]) if pd.notna(decomposition.trend.loc[idx]) else None,
            "seasonal": float(decomposition.seasonal.loc[idx]) if pd.notna(decomposition.seasonal.loc[idx]) else None,
            "resid": float(decomposition.resid.loc[idx]) if pd.notna(decomposition.resid.loc[idx]) else None,
            "source_collection": collection_name,
            "scale": "transformed"
        }

        result = db["decompose-lstm-temp"].update_one(
            {"date": date_str, "config_id": config_id},
            {
                "$set": {
                    "date": date_str,
                    "timestamp": datetime.now().isoformat(),
                    "config_id": config_id,
                    f"parameters.{target_column}": param_doc
                }
            },
            upsert=True
        )
        if result.upserted_id or result.modified_count > 0:
            saved_count += 1

    return saved_count

# ======================================================
# PyTorch Dataset Class
# ======================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def seed_worker(worker_id):
    """Seed worker for DataLoader reproducibility"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ======================================================
# LSTM Model - RECURSIVE (Classic)
# ======================================================
class RecursiveLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size_1=128, hidden_size_2=64, dropout=0.3):
        super(RecursiveLSTM, self).__init__()
        
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size_1, num_layers=1, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_size_1)
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(input_size=hidden_size_1, hidden_size=hidden_size_2, num_layers=1, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_size_2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(hidden_size_2, 64)
        self.dropout_fc = nn.Dropout(dropout * 0.5)
        
        self.fc_out = nn.Linear(64, 1) # Output 1 step ahead
        
    def forward(self, x):
        lstm_out1, _ = self.lstm1(x)
        out1 = self.ln1(lstm_out1)
        out1 = self.dropout1(out1)
        
        lstm_out2, _ = self.lstm2(out1)
        out2 = lstm_out2[:, -1, :]
        out2 = self.ln2(out2)
        out2 = self.dropout2(out2)
        
        out = self.dropout_fc(self.fc1(out2))
        return self.fc_out(out)

# ======================================================
# Create supervised data - ONE STEP AHEAD
# ======================================================
def create_supervised_onestep(data_scaled, lookback):
    if len(data_scaled) <= lookback:
        return np.array([]), np.array([])
    
    n_samples = len(data_scaled) - lookback
    
    indices_X = np.arange(lookback)[None, :] + np.arange(n_samples)[:, None]
    X = data_scaled[indices_X].reshape(n_samples, lookback, 1)
    y = data_scaled[lookback:].reshape(-1, 1)
    
    return X, y

def recursive_forecast(model, initial_window, forecast_horizon, scaler, device):
    """Pure recursive forecast"""
    model.eval()
    
    current_window = initial_window.copy()
    forecast_scaled = []

    with torch.no_grad():
        for step_idx in range(forecast_horizon):
            window_tensor = torch.FloatTensor(current_window).reshape(1, -1, 1).to(device)
            pred = model(window_tensor).cpu().numpy().flatten()[0]
            
            forecast_scaled.append(pred)
            # Update window: remove first, add prediction to end
            current_window = np.append(current_window[1:], pred)
    
    forecast_scaled_array = np.array(forecast_scaled).reshape(-1, 1)
    forecast_unscaled = scaler.inverse_transform(forecast_scaled_array).flatten()
    
    return forecast_unscaled

# ======================================================
# Post-process forecast
# ======================================================
def post_process_forecast(forecast, param_name, historical_data=None):
    """Enhanced post-processing with physical constraints"""
    
    if param_name in RAINFALL_PARAMS:
        forecast = np.clip(forecast, 0, 300)
        forecast[forecast < 0.01] = 0
    elif param_name in HUMIDITY_PARAMS:
        forecast = np.clip(forecast, 0, 100)
    elif param_name in NDVI_PARAMS:
        forecast = np.clip(forecast, -1, 1)
    elif param_name in TEMP_PARAMS:
        forecast = np.clip(forecast, -50, 60)
    elif param_name in SOLAR_PARAMS:
        forecast = np.clip(forecast, 0, 30)
    else:
        param_lower = param_name.lower()
        if any(kw in param_lower for kw in ["rain", "precip", "hujan"]):
            forecast = np.clip(forecast, 0, 400)
            forecast[forecast < 0.01] = 0
        elif any(kw in param_lower for kw in ["hum", "rh", "kelembaban"]):
            forecast = np.clip(forecast, 0, 100)
        elif any(kw in param_lower for kw in ["temp", "suhu", "t2m", "tavg"]):
            forecast = np.clip(forecast, -50, 60)
        elif any(kw in param_lower for kw in ["ndvi", "vegetasi"]):
            forecast = np.clip(forecast, -1, 1)
        elif any(kw in param_lower for kw in ["solar", "rad", "ghi", "srad"]):
            forecast = np.clip(forecast, 0, None)
    
    return forecast

# ======================================================
# Training function
# ======================================================
def train_pytorch_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=20):
    """Train PyTorch LSTM model"""
    
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    best_model_state = None
    
    print(f"\n{'='*70}")
    print(f"🎯 TRAINING STARTED (Recursive LSTM with STL)")
    print(f"{'='*70}\n")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
                break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETED")
    print(f"   Epochs trained: {epoch+1}/{epochs}")
    print(f"   Best validation loss: {best_val_loss:.6f}")
    print(f"{'='*70}\n")
    
    return model, train_losses, val_losses

# ======================================================
# Calculate metrics
# ======================================================
def calculate_metrics(y_true, y_pred, param_name="generic"):
    """Comprehensive metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    param_lower = param_name.lower()
    is_rainfall = any(x in param_lower for x in ['rr', 'rain', 'precip', 'hujan']) or param_name in RAINFALL_PARAMS
    
    if is_rainfall:
        rain_mask = y_true >= 1.0
        if np.sum(rain_mask) > 0:
            mape = float(np.mean(np.abs(
                (y_true[rain_mask] - y_pred[rain_mask]) / y_true[rain_mask]
            )) * 100)
        else:
            mape = 0.0
    else:
        non_zero_mask = np.abs(y_true) > 0.1
        if np.sum(non_zero_mask) > 0:
            mape = float(np.mean(np.abs(
                (y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]
            )) * 100)
        else:
            mape = 0.0

    r2 = r2_score(y_true, y_pred)
    
    if len(y_true) > 1:
        true_dir = np.diff(y_true) > 0
        pred_dir = np.diff(y_pred) > 0
        dir_acc = np.mean(true_dir == pred_dir) * 100
    else:
        dir_acc = None
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mse': mse,
        'mape': mape,
        'r2': r2,
        'directional_accuracy': dir_acc
    }

# ======================================================
# Grid search
# ======================================================
def restore_transformed_scale(values, index, seasonal_component=None):
    values = np.asarray(values, dtype=float)
    if seasonal_component is not None:
        seasonal_values = seasonal_component.reindex(index).ffill().bfill().values
        values = values + seasonal_values
    return values

def restore_physical_scale(values, index, transform_method, seasonal_component=None, bias_correction=0.0):
    values = restore_transformed_scale(values, index, seasonal_component)
    values = apply_log_bias_correction(values, transform_method, bias_correction)
    return inverse_transformation(values, transform_method)

def project_future_seasonality(seasonal_component, forecast_dates):
    """Project seasonal component by calendar day-of-year instead of blind tiling."""
    if seasonal_component is None:
        return None

    seasonal_by_doy = seasonal_component.groupby(seasonal_component.index.dayofyear).mean()
    fallback = float(seasonal_component.mean()) if len(seasonal_component) else 0.0
    future_values = []
    for forecast_date in pd.to_datetime(forecast_dates):
        day_of_year = forecast_date.dayofyear
        if day_of_year in seasonal_by_doy.index:
            future_values.append(float(seasonal_by_doy.loc[day_of_year]))
        elif day_of_year == 366 and 365 in seasonal_by_doy.index:
            future_values.append(float(seasonal_by_doy.loc[365]))
        else:
            future_values.append(fallback)

    return np.asarray(future_values, dtype=float)

def build_horizon_confidence(one_step_rmse, recursive_rmse, forecast_dates):
    """Create lightweight horizon-risk metadata for downstream planting calendar logic."""
    dates = pd.to_datetime(forecast_dates)
    horizon_days = np.arange(1, len(dates) + 1, dtype=float)

    one_step = float(one_step_rmse) if one_step_rmse is not None and np.isfinite(one_step_rmse) else None
    recursive = float(recursive_rmse) if recursive_rmse is not None and np.isfinite(recursive_rmse) else one_step

    if one_step is None or recursive is None or len(horizon_days) == 0:
        return {
            "rmse_proxy": [None] * len(dates),
            "confidence": [None] * len(dates),
            "summary": {
                "one_step_rmse": one_step,
                "recursive_rmse": recursive,
                "rmse_degradation_pct": None,
                "monthly": {}
            }
        }

    max_horizon = max(float(horizon_days[-1]), 1.0)
    rmse_proxy = one_step + (recursive - one_step) * (horizon_days / max_horizon)
    scale = max(recursive, one_step, 1e-6)
    confidence = np.clip(1.0 - (rmse_proxy / (scale * 1.5)), 0.0, 1.0)
    degradation_pct = ((recursive - one_step) / one_step) * 100 if one_step > 0 else None

    monthly = {}
    temp_df = pd.DataFrame({
        "month": dates.strftime("%Y-%m"),
        "rmse_proxy": rmse_proxy,
        "confidence": confidence
    })
    for month, month_df in temp_df.groupby("month"):
        monthly[month] = {
            "rmse_proxy": round(float(month_df["rmse_proxy"].mean()), 4),
            "confidence": round(float(month_df["confidence"].mean()), 4)
        }

    return {
        "rmse_proxy": [round(float(value), 4) for value in rmse_proxy],
        "confidence": [round(float(value), 4) for value in confidence],
        "summary": {
            "one_step_rmse": round(one_step, 4),
            "recursive_rmse": round(recursive, 4),
            "rmse_degradation_pct": round(float(degradation_pct), 2) if degradation_pct is not None else None,
            "monthly": monthly
        }
    }

def grid_search_pytorch(train_data, param_name, transform_method="none",
                        original_data=None, seasonal_component=None):
    """Grid search with transformation support"""
    print(f"\n{'='*70}\nPYTORCH GRID SEARCH: {param_name} (Recursive + STL)\n{'='*70}")
    
    if len(train_data) < 500:
        print(f"❌ Insufficient data: {len(train_data)}")
        return None, None
    
    split_point = int(len(train_data) * TRAIN_SPLIT_RATIO)
    train_split = train_data[:split_point]
    val_split = train_data[split_point:]
    
    print(f"   Train split: {len(train_split)} points")
    print(f"   Val split: {len(val_split)} points")
    
    data_years = len(train_data) / 365
    if data_years >= 10:
        lookback_range = [180, 365, 730]
    elif data_years >= 5:
        lookback_range = [90, 180, 365]
    else:
        lookback_range = [60, 90, 180]
    
    scaler_range = ['standard', 'minmax']
    if FAST_MODE:
        lookback_range = lookback_range[:2]
        scaler_range = ['standard']

    valid_lookbacks = [lb for lb in lookback_range if len(train_split) > lb + 100]
    
    if not valid_lookbacks:
        print("❌ No valid lookbacks")
        return None, None
    
    total_combinations = len(valid_lookbacks) * len(scaler_range)
    print(f"Testing {total_combinations} combinations")
    print(f"Lookbacks: {valid_lookbacks}")
    print(f"Scalers: {scaler_range}")
    
    best_score = float('inf')
    best_result = None
    current_combo = 0
    
    for lookback, scaler_type in itertools.product(valid_lookbacks, scaler_range):
        current_combo += 1
        print(f"\n[{current_combo}/{total_combinations}] Testing: lookback={lookback}, scaler={scaler_type}")
        
        try:
            scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
            
            train_vals = train_split.values.reshape(-1, 1)
            scaler.fit(train_vals)
            
            train_scaled = scaler.transform(train_vals).flatten()
            val_vals = val_split.values.reshape(-1, 1)
            val_scaled = scaler.transform(val_vals).flatten()
            
            X_train, y_train = create_supervised_onestep(train_scaled, lookback)
            
            if len(val_scaled) > lookback:
                X_val, y_val = create_supervised_onestep(val_scaled, lookback)
            else:
                print(f"   ⚠️  Val split too short for lookback={lookback}")
                continue
            
            if len(X_train) < 100:
                print("   ⚠️  Insufficient samples")
                continue
            
            train_dataset = TimeSeriesDataset(X_train, y_train)
            val_dataset = TimeSeriesDataset(X_val, y_val)
            
            g = torch.Generator()
            g.manual_seed(RANDOM_SEED)
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, 
                                    worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
            
            model = RecursiveLSTM(input_size=1, hidden_size_1=128, hidden_size_2=64, dropout=0.3).to(device)
            
            model, _, _ = train_pytorch_model(
                model,
                train_loader,
                val_loader,
                epochs=GRID_EPOCHS,
                lr=0.001,
                patience=GRID_PATIENCE
            )
            
            # Predict
            model.eval()
            val_preds = []
            val_trues = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    outputs = model(X_batch)
                    val_preds.append(outputs.cpu().numpy())
                    val_trues.append(y_batch.numpy())
            
            val_pred = np.concatenate(val_preds, axis=0).flatten()
            val_true = np.concatenate(val_trues, axis=0).flatten()
            
            val_pred_unscaled = scaler.inverse_transform(val_pred.reshape(-1, 1)).flatten()
            val_true_unscaled = scaler.inverse_transform(val_true.reshape(-1, 1)).flatten()
            
            val_metric_index = val_split.index[lookback:lookback + len(val_pred_unscaled)]
            if original_data is not None:
                y_true_physical = original_data.reindex(val_metric_index).values
            else:
                y_true_physical = restore_physical_scale(
                    val_true_unscaled,
                    val_metric_index,
                    transform_method,
                    seasonal_component
                )

            bias_info = {
                "correction": 0.0,
                "raw_correction": 0.0,
                "mode": "none",
                "sample_count": 0,
                "wet_day_threshold": None,
                "negative_clamped": False,
                "positive_capped": False
            }
            bias_correction = 0.0
            if is_rainfall_param(param_name) and transform_method == "log1p":
                val_pred_transformed = restore_transformed_scale(
                    val_pred_unscaled,
                    val_metric_index,
                    seasonal_component
                )
                bias_info = estimate_log_bias_correction(
                    y_true_physical,
                    val_pred_transformed,
                    transform_method,
                    param_name
                )
                bias_correction = bias_info["correction"]

            y_pred_physical = restore_physical_scale(
                val_pred_unscaled,
                val_metric_index,
                transform_method,
                seasonal_component,
                bias_correction
            )
            y_pred_physical = post_process_forecast(y_pred_physical, param_name)

            metrics = calculate_metrics(y_true_physical, y_pred_physical, param_name)
            metrics["log_bias_correction"] = bias_correction
            metrics["log_bias_correction_raw"] = bias_info["raw_correction"]
            metrics["log_bias_correction_mode"] = bias_info["mode"]
            metrics["log_bias_correction_sample_count"] = bias_info["sample_count"]
            metrics["log_bias_correction_wet_day_threshold"] = bias_info["wet_day_threshold"]
            metrics["log_bias_correction_negative_clamped"] = bias_info["negative_clamped"]
            metrics["log_bias_correction_positive_capped"] = bias_info["positive_capped"]

            recursive_horizon = min(BACKTEST_HORIZON, len(val_split))
            recursive_metrics = None
            if recursive_horizon > 0 and len(train_scaled) >= lookback:
                recursive_pred = recursive_forecast(
                    model=model,
                    initial_window=train_scaled[-lookback:],
                    forecast_horizon=recursive_horizon,
                    scaler=scaler,
                    device=device
                )
                recursive_index = val_split.index[:recursive_horizon]
                recursive_pred_physical = restore_physical_scale(
                    recursive_pred,
                    recursive_index,
                    transform_method,
                    seasonal_component,
                    bias_correction
                )
                recursive_pred_physical = post_process_forecast(recursive_pred_physical, param_name)
                if original_data is not None:
                    recursive_true = original_data.reindex(recursive_index).values
                else:
                    recursive_true = restore_physical_scale(
                        val_split.iloc[:recursive_horizon].values,
                        recursive_index,
                        transform_method,
                        seasonal_component
                    )
                recursive_metrics = calculate_metrics(recursive_true, recursive_pred_physical, param_name)
                metrics["recursive_rmse"] = recursive_metrics["rmse"]
                metrics["recursive_mae"] = recursive_metrics["mae"]
                metrics["recursive_mape"] = recursive_metrics["mape"]
            
            print(f"   RMSE (physical one-step): {metrics['rmse']:.3f}, R²: {metrics['r2']:.3f}")
            if recursive_metrics:
                print(f"   Recursive RMSE ({recursive_horizon}d): {recursive_metrics['rmse']:.3f}")
            
            score = metrics.get("recursive_rmse", metrics["rmse"])
            if score < best_score:
                best_score = score
                best_result = {
                    'lookback': lookback,
                    'scaler_type': scaler_type,
                    'metrics': metrics
                }
                print(f"   ✅ New Best: {best_score:.4f}")
            
            del model, train_loader, val_loader, train_dataset, val_dataset
            del X_train, y_train, X_val, y_val
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            if 'model' in locals():
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue
    
    if best_result is None:
        print("\n❌ No valid configuration found!")
        return None, None
    
    print(f"\n{'='*70}")
    print(f"✅ GRID SEARCH COMPLETED")
    print(f"{'='*70}")
    print(f"   Best lookback: {best_result['lookback']} days ({best_result['lookback']/365:.2f} years)")
    print(f"   Best scaler: {best_result['scaler_type']}")
    print(f"   Best selection score: {best_score:.4f}")
    print(f"{'='*70}\n")
    
    best_params = {
        'lookback': best_result['lookback'],
        'lookback_years': round(best_result['lookback']/365, 2),
        'scaler_type': best_result['scaler_type'],
        'hidden_size_1': 128,
        'hidden_size_2': 64,
        'dropout': 0.3,
        'epochs': FINAL_EPOCHS,
        'batch_size': 128,
        'learning_rate': 0.001,
        'log_bias_correction': best_result['metrics'].get('log_bias_correction', 0.0),
        'log_bias_correction_raw': best_result['metrics'].get('log_bias_correction_raw', 0.0),
        'log_bias_correction_mode': best_result['metrics'].get('log_bias_correction_mode', 'none'),
        'log_bias_correction_sample_count': best_result['metrics'].get('log_bias_correction_sample_count', 0),
        'log_bias_correction_wet_day_threshold': best_result['metrics'].get('log_bias_correction_wet_day_threshold'),
        'log_bias_correction_negative_clamped': best_result['metrics'].get('log_bias_correction_negative_clamped', False),
        'log_bias_correction_positive_capped': best_result['metrics'].get('log_bias_correction_positive_capped', False)
    }
    
    return best_params, best_result['metrics']

# ======================================================
# Main pipeline
# ======================================================
def run_lstm_analysis(collection_name, target_column, save_collection="lstm-forecast",
                      config_id=None, append_column_id=True, client=None,
                      forecast_horizon=365, start_date=None, end_date=None):
    """RECURSIVE LSTM FORECASTING WITH SEASONAL DECOMPOSITION"""
    print(f"\n{'='*70}")
    print(f"🔥 RECURSIVE LSTM FORECAST SYSTEM (STL + LSTM)")
    print(f"{'='*70}")
    print(f"📊 Target: {collection_name}.{target_column}")
    print(f"🎯 Forecast horizon: {forecast_horizon} days (RECURSIVE)")
    print(f"⚙️ Fast mode: {'ON' if FAST_MODE else 'OFF'}")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    if client is None:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        MONGO_URI = os.getenv("MONGODB_URI")
        client = MongoClient(MONGO_URI)
        should_close_client = True
    else:
        should_close_client = False
    
    db = client["tugas_akhir"]
    
    try:
        # 1. Fetch data
        print("📥 Step 1: Fetching data...")
        source_data = list(db[collection_name].find().sort("Date", 1))
        print(f"   Retrieved: {len(source_data)} records")
        
        if not source_data:
            raise ValueError(f"No data in {collection_name}")
        
        # 2. Prepare
        print("\n🔧 Step 2: Data preparation...")
        df = pd.DataFrame(source_data)
        
        param_data = prepare_cleaned_series(df, target_column)
        warnings_list = []
        if not collection_name.endswith("_cleaned"):
            warning_msg = (
                f"Collection '{collection_name}' does not end with '_cleaned'. "
                "LSTM expects BMKG/NASA preprocessing output for best consistency."
            )
            print(f"   ⚠️ {warning_msg}")
            warnings_list.append(warning_msg)
        
        if len(param_data) < 500:
            raise ValueError(f"Insufficient data: {len(param_data)}")
        
        data_end_date = param_data.index[-1]
        if end_date:
            end_dt = pd.to_datetime(end_date)
            if end_dt > data_end_date:
                days_needed = (end_dt - data_end_date).days
                forecast_horizon = days_needed
                print(f"   🗓️  Adjusted forecast horizon to {forecast_horizon} days (Target: {end_dt.date()})")
        
        print(f"   Data: {len(param_data)} points ({len(param_data)/365:.1f} years)")
        
        # 3. Apply transformation
        print("\n⚡ Step 3: Data Transformation...")
        transformed_data, transform_method = apply_transformation(param_data, target_column)
        
        # 4. [NEW] Apply Seasonal Decomposition
        print("\n🍂 Step 4: Seasonal Decomposition...")
        deseasonalized_data, seasonal_component, decomposition = apply_seasonal_decomposition(transformed_data)
        decompose_count = save_lstm_decomposition(
            db=db,
            decomposition=decomposition,
            observed_series=transformed_data,
            collection_name=collection_name,
            target_column=target_column,
            config_id=config_id
        )
        if decompose_count:
            print(f"   ✅ Saved {decompose_count} decomposition points to decompose-lstm-temp")
        
        # Use deseasonalized data for training
        training_data = deseasonalized_data
        
        # 5. Grid search
        print("\n🔍 Step 5: Grid search (on Deseasonalized Data)...")
        best_params, val_metrics = grid_search_pytorch(
            training_data,
            target_column,
            transform_method,
            original_data=param_data,
            seasonal_component=seasonal_component
        )
        
        if best_params is None:
            raise ValueError("No valid model found")

        if is_rainfall_param(target_column) and best_params.get("log_bias_correction_negative_clamped"):
            warning_msg = (
                f"{target_column}: negative log bias correction was clamped to 0.0 "
                "to avoid rainfall underestimation in planting-calendar thresholds."
            )
            print(f"   ⚠️ {warning_msg}")
            warnings_list.append(warning_msg)
        
        # 6. Final training
        print("\n🎓 Step 6: Final model training...")
        
        scaler = StandardScaler() if best_params['scaler_type'] == 'standard' else MinMaxScaler()
        
        vals = training_data.values.reshape(-1, 1)
        split_index_raw = int(len(vals) * TRAIN_SPLIT_RATIO)
        train_vals_raw = vals[:split_index_raw]
        
        scaler.fit(train_vals_raw)
        data_scaled = scaler.transform(vals).flatten()
        
        X, y = create_supervised_onestep(data_scaled, best_params['lookback'])
        
        split_idx_seq = split_index_raw - best_params['lookback']
        if split_idx_seq <= 0 or split_idx_seq >= len(X):
            split_idx_seq = max(100, min(len(X) - 100, int(len(X) * TRAIN_SPLIT_RATIO)))

        X_train, X_test = X[:split_idx_seq], X[split_idx_seq:]
        y_train, y_test = y[:split_idx_seq], y[split_idx_seq:]
        
        train_dataset = TimeSeriesDataset(X_train, y_train)
        test_dataset = TimeSeriesDataset(X_test, y_test)
        
        g = torch.Generator()
        g.manual_seed(RANDOM_SEED)
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], 
                                shuffle=True, worker_init_fn=seed_worker, generator=g)
        test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
        
        final_model = RecursiveLSTM(
            input_size=1,
            hidden_size_1=best_params['hidden_size_1'],
            hidden_size_2=best_params['hidden_size_2'],
            dropout=best_params['dropout']
        ).to(device)
        
        final_model, train_losses, val_losses = train_pytorch_model(
            final_model, train_loader, test_loader,
            epochs=best_params['epochs'],
            lr=best_params['learning_rate'],
            patience=FINAL_PATIENCE
        )
        
        # 7. Generate RECURSIVE forecast (Deseasonalized)
        print(f"\n🔮 Step 7: Generating Forecast ({forecast_horizon} days)...")
        
        last_window = data_scaled[-best_params['lookback']:]
        
        # Forecast components (Deseasonalized)
        forecast_deseasonalized = recursive_forecast(
            model=final_model,
            initial_window=last_window,
            forecast_horizon=forecast_horizon,
            scaler=scaler,
            device=device
        )

        forecast_dates = pd.date_range(
            start=data_end_date + pd.Timedelta(days=1),
            periods=forecast_horizon,
            freq='D'
        )
        
        # 8. [NEW] Recombine Seasonality
        print(f"   🍂 Recombining Seasonal Component by day-of-year...")
        log_bias_correction = best_params.get("log_bias_correction", 0.0)
        if seasonal_component is not None:
            future_seasonality = project_future_seasonality(seasonal_component, forecast_dates)
            forecast_transformed = forecast_deseasonalized + future_seasonality
        else:
            forecast_transformed = forecast_deseasonalized

        forecast_transformed = apply_log_bias_correction(
            forecast_transformed,
            transform_method,
            log_bias_correction
        )
        
        # Inverse transformation (Log1p -> Original)
        print(f"   🔄 Inverting transformation ({transform_method})...")
        forecast_final = inverse_transformation(forecast_transformed, transform_method)
        
        forecast_final = post_process_forecast(forecast_final, target_column, param_data.values)
        
        print(f"   ✅ Forecast completed")
        print(f"   Range: {forecast_final.min():.2f} to {forecast_final.max():.2f}")
        print(f"   Mean: {forecast_final.mean():.2f}")
        
        print("\n💾 Step 8: Saving to database...")

        horizon_confidence = build_horizon_confidence(
            val_metrics.get("rmse"),
            val_metrics.get("recursive_rmse"),
            forecast_dates
        )
        
        forecast_docs = []
        start_dt_filter = pd.to_datetime(start_date) if start_date else None

        for i, (forecast_date, forecast_value) in enumerate(zip(forecast_dates, forecast_final)):
            if start_dt_filter and forecast_date < start_dt_filter:
                continue
            
            if np.isnan(forecast_value) or np.isinf(forecast_value):
                continue
            
            doc = {
                "forecast_date": forecast_date.to_pydatetime(),
                "timestamp": datetime.now().isoformat(),
                "source_collection": collection_name,
                "config_id": config_id,
                "parameters": {
                    target_column: {
                        "forecast_value": float(forecast_value),
                        "day_ahead": i + 1,
                        "model_metadata": {
                            "framework": "PyTorch",
                            "model": "LSTM_Recursive_STL",
                            "version": "3.1_stl_bias_confidence",
                            "transform": transform_method,
                            "forecast_method": "recursive_stl",
                            "lookback_days": best_params['lookback'],
                            "test_rmse": val_metrics['rmse'],
                            "test_mae": val_metrics['mae'],
                            "test_mape": val_metrics['mape'],
                            "recursive_rmse": val_metrics.get('recursive_rmse'),
                            "recursive_mae": val_metrics.get('recursive_mae'),
                            "recursive_mape": val_metrics.get('recursive_mape'),
                            "log_bias_correction": log_bias_correction,
                            "log_bias_correction_raw": best_params.get("log_bias_correction_raw"),
                            "log_bias_correction_mode": best_params.get("log_bias_correction_mode"),
                            "log_bias_correction_sample_count": best_params.get("log_bias_correction_sample_count"),
                            "log_bias_correction_wet_day_threshold": best_params.get("log_bias_correction_wet_day_threshold"),
                            "log_bias_correction_negative_clamped": best_params.get("log_bias_correction_negative_clamped"),
                            "log_bias_correction_positive_capped": best_params.get("log_bias_correction_positive_capped"),
                            "horizon_rmse_proxy": horizon_confidence["rmse_proxy"][i],
                            "horizon_confidence": horizon_confidence["confidence"][i],
                            "rmse_degradation_pct": horizon_confidence["summary"].get("rmse_degradation_pct"),
                            "warnings": warnings_list
                        }
                    }
                }
            }
            
            if append_column_id:
                doc["column_id"] = f"{collection_name}_{target_column}"
            
            forecast_docs.append(doc)
        
        # Upsert
        upsert_count = 0
        for doc in forecast_docs:
            set_fields = {
                f"parameters.{target_column}": doc["parameters"][target_column],
                "timestamp": doc["timestamp"],
                "source_collection": doc["source_collection"],
                "config_id": doc["config_id"],
            }

            if append_column_id and "column_id" in doc:
                set_fields["column_id"] = doc["column_id"]

            result = db[save_collection].update_one(
                {"forecast_date": doc["forecast_date"], "config_id": config_id},
                {"$set": set_fields},
                upsert=True
            )
            if result.upserted_id or result.modified_count > 0:
                upsert_count += 1
        
        print(f"   ✅ Saved {upsert_count}/{len(forecast_docs)} documents")
        
        # 9. Summary
        print(f"\n{'='*70}")
        print(f"✅ RECURSIVE FORECAST (STL) COMPLETED")
        print(f"{'='*70}")
        print(f"   Transform: {transform_method}")
        print(f"   Seasonal Mode: Additive (Period=365)")
        print(f"{'='*70}\n")
        
        return {
            "status": "success",
            "framework": "PyTorch",
            "forecast_method": "recursive_stl",
            "transform": transform_method,
            "target_column": target_column,
            "error_metrics": val_metrics,
            "horizon_confidence": horizon_confidence["summary"],
            "warnings": warnings_list,
            "metric_context": {
                "scale": "physical_original_units",
                "one_step_metrics": ["mae", "mse", "rmse", "mape", "r2", "directional_accuracy"],
                "recursive_metrics": ["recursive_mae", "recursive_rmse", "recursive_mape"],
                "selection_metric": "recursive_rmse_when_available"
            },
            "forecast": {
                "values": forecast_final.tolist(),
                "dates": [d.strftime('%Y-%m-%d') for d in forecast_dates]
            },
            "model_config": best_params
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}
    
    finally:
        if should_close_client:
            client.close()

# ======================================================
# Run
# ======================================================
if __name__ == "__main__":
    result = run_lstm_analysis(
        collection_name="bmkg-data",
        target_column="RR"
    )
    
    print(f"\nFinal status: {result['status']}")
