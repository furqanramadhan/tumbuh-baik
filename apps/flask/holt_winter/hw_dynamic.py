from pymongo import MongoClient
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

import itertools
import warnings
warnings.filterwarnings('ignore')

PARAM_MODEL_CONFIG = {
    "TAVG": {
        "transform": None,
        "trend": "add",
        "seasonal": "add",
        "damped_trend": False,
    },
    "RH_AVG": {
        "transform": None,
        "trend": "add",
        "seasonal": "add",
        "damped_trend": False,
    },
    "RR": {
        "transform": None,
        "trend": "add",
        "seasonal": "add",
        "damped_trend": False,
    },
    "RR_imputed": {
        "transform": None,
        "trend": "add",
        "seasonal": "add",
        "damped_trend": False,
    },
    "RH2M": {
        "transform": None,
        "trend": "add",
        "seasonal": "add",
        "damped_trend": False,
    },
    "PRECTOTCORR": {
        "transform": None,
        "trend": "add",
        "seasonal": "add",
        "damped_trend": False,
    },
    "T2M": {
        "transform": None,
        "trend": "add",
        "seasonal": "add",
        "damped_trend": False,
    },
    "ALLSKY_SFC_SW_DWN": {
        "transform": None,
        "trend": "add",
        "seasonal": "add",
        "damped_trend": True,
        "damping_trend": 0.90,
    },
}

DEFAULT_PARAM_CONFIG = {
    "transform": None,
    "trend": "add",
    "seasonal": "add",
    "damped_trend": False,
}

def get_param_model_config(param_name):
    """Return Holt-Winters behavior for a BMKG or NASA POWER parameter."""
    return PARAM_MODEL_CONFIG.get(param_name, DEFAULT_PARAM_CONFIG)

def transform_series(values, method):
    """
    Transform data for model fitting while preserving a Series index when present.
    """
    transformed = np.asarray(values, dtype=float)
    if method == "log1p":
        transformed = np.log1p(transformed)

    if isinstance(values, pd.Series):
        return pd.Series(transformed, index=values.index, name=values.name)
    return transformed

def inverse_transform(values, method):
    """
    Invert model output back to the physical unit scale.
    """
    inverted = np.asarray(values, dtype=float)
    if method == "log1p":
        inverted = np.expm1(inverted)
    return inverted

def inverse_transform_forecast(values, method, correction_factor=1.0):
    """
    Invert forecast output and optionally apply log1p smearing correction.
    """
    values = np.asarray(values, dtype=float)
    if method == "log1p":
        return np.exp(values) * correction_factor - 1
    return inverse_transform(values, method)

def calculate_smearing_factor(actual_values, forecast_transformed, method):
    """
    Estimate Duan smearing factor from validation residuals.
    """
    if method != "log1p":
        return 1.0

    actual_transformed = transform_series(actual_values, method)
    residuals = np.asarray(actual_transformed, dtype=float) - np.asarray(forecast_transformed, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) == 0:
        return 1.0

    residuals = np.clip(residuals, -20, 20)
    factor = float(np.mean(np.exp(residuals)))
    if not np.isfinite(factor) or factor <= 0:
        return 1.0
    return factor

def fit_robust_model(data, best_params, param_name):
    """
    Fit model dengan error handling yang lebih baik
    """
    config = get_param_model_config(param_name)
    transformed_data = transform_series(data, config.get("transform"))

    try:
        model_kwargs = {
            "trend": best_params.get("trend", config.get("trend", "add")),
            "seasonal": best_params.get("seasonal", config.get("seasonal", "add")),
            "seasonal_periods": best_params.get('seasonal_periods', 365),
            "initialization_method": best_params.get('initialization_method', 'heuristic'),
        }

        if config.get("damped_trend"):
            model_kwargs["damped_trend"] = True

        fit_kwargs = {
            "smoothing_level": best_params['alpha'],
            "smoothing_trend": best_params['beta'],
            "smoothing_seasonal": best_params['gamma'],
            "optimized": False,
        }

        if config.get("damped_trend"):
            fit_kwargs["damping_trend"] = config.get("damping_trend", 0.90)

        model = ExponentialSmoothing(
            transformed_data,
            **model_kwargs
        ).fit(**fit_kwargs)
        return model
    except Exception as e:
        print(f"Model fitting failed: {e}")
        # Fallback ke simple exponential smoothing
        try:
            model = ExponentialSmoothing(
                transformed_data,
                trend=None,
                seasonal=None,
                initialization_method='heuristic'
            ).fit(smoothing_level=0.3)
            return model
        except:
            return None

def post_process_forecast(forecast, param_name):
    """
    Post-processing untuk memastikan forecast masuk akal
    """
    if param_name in ["RR", "RR_imputed"]:  # Curah Hujan BMKG
        forecast = np.maximum(forecast, 0)
        forecast = np.minimum(forecast, 300)
        
    elif param_name == "RH_AVG":  # Kelembapan
        # Harus dalam range 0-100%
        forecast = np.clip(forecast, 0, 100)

    elif param_name == "RH2M":  # Kelembapan NASA POWER
        forecast = np.clip(forecast, 0, 100)

    elif param_name == "PRECTOTCORR":  # Presipitasi NASA POWER
        forecast = np.maximum(forecast, 0)
        forecast = np.minimum(forecast, 300)

    elif param_name == "T2M":  # Suhu NASA POWER
        forecast = np.clip(forecast, -50, 60)

    elif param_name == "ALLSKY_SFC_SW_DWN":  # MJ/m2/day
        forecast = np.clip(forecast, 0, 30)
    
    elif param_name == "NDVI":  # Normalized Difference Vegetation Index
        # NDVI harus dalam range -1 to 1
        forecast = np.clip(forecast, -1, 1)
    
    elif "Suhu" in param_name or "Temperature" in param_name:  # Suhu
        forecast = np.clip(forecast, -50, 60)
        # Batasi suhu dalam range yang masuk akal (-50°C to 60°C)
    
    return forecast

def grid_search_hw_params(train_data, param_name):
    """
    Grid search disesuaikan untuk pola curah hujan Indonesia
    """
    print(f"\n--- Grid Search for Indonesian Rainfall Pattern: {param_name} ---")
    config = get_param_model_config(param_name)
    
    if len(train_data) < 365:  # Minimal 1 tahun
        print("❌ Insufficient data (need at least 1 year)")
        return None, None
    
    # Parameter grid yang lebih konservatif untuk rainfall
    alpha_range = config.get("alpha_range", [0.1, 0.2, 0.3, 0.5])
    beta_range = config.get("beta_range", [0.05, 0.1, 0.2])
    gamma_range = config.get("gamma_range", [0.1, 0.2, 0.3])
    
    seasonal_periods_options = []
    if len(train_data) >= 365*2:  
        seasonal_periods_options.append(365)  
    if len(train_data) >= 180*2:  
        seasonal_periods_options.append(180) 
    if len(train_data) >= 90*2:  
        seasonal_periods_options.append(90)   
    if len(train_data) >= 30*3:  
        seasonal_periods_options.append(30)  
    
    if not seasonal_periods_options:
        seasonal_periods_options = [7]  
    
    print(f"Testing seasonal periods: {seasonal_periods_options}")
    
    best_score = float('inf')
    best_params = None
    best_metrics = None
    valid_models = 0
    
   
    val_days = min(365*2, int(len(train_data) * 0.2))  
    split_point = len(train_data) - val_days
    
    train_split = train_data[:split_point]
    val_split = train_data[split_point:]
    train_split_model = transform_series(train_split, config.get("transform"))
    
    print(f"Train: {len(train_split)} days, Validation: {len(val_split)} days")
    
    for seasonal_periods in seasonal_periods_options:
        if len(train_split_model) < seasonal_periods * 2:
            continue

        for alpha in alpha_range:
            for beta in beta_range:
                for gamma in gamma_range:
                    try:
                        model_kwargs = {
                            "trend": config.get("trend", "add"),
                            "seasonal": config.get("seasonal", "add"),
                            "seasonal_periods": seasonal_periods,
                            "initialization_method": "heuristic",
                        }

                        if config.get("damped_trend"):
                            model_kwargs["damped_trend"] = True

                        fit_kwargs = {
                            "smoothing_level": alpha,
                            "smoothing_trend": beta,
                            "smoothing_seasonal": gamma,
                            "optimized": False,
                        }

                        if config.get("damped_trend"):
                            fit_kwargs["damping_trend"] = config.get("damping_trend", 0.90)

                        model = ExponentialSmoothing(
                            train_split_model,
                            **model_kwargs
                        ).fit(**fit_kwargs)

                        forecast_model_scale = model.forecast(len(val_split))
                        correction_factor = 1.0
                        if config.get("bias_correction") == "smearing":
                            correction_factor = calculate_smearing_factor(
                                val_split,
                                forecast_model_scale,
                                config.get("transform")
                            )

                        forecast = inverse_transform_forecast(
                            forecast_model_scale,
                            config.get("transform"),
                            correction_factor
                        )
                        forecast = np.maximum(forecast, 0)

                        mae = mean_absolute_error(val_split, forecast)
                        mse = mean_squared_error(val_split, forecast)
                        rmse = np.sqrt(mse)
                        mape = np.mean(np.abs((val_split - forecast) / np.where(val_split != 0, val_split, 1))) * 100

                        score = mae * 0.7 + rmse * 0.3

                        if score < best_score:
                            best_score = score
                            best_params = {
                                'alpha': alpha,
                                'beta': beta,
                                'gamma': gamma,
                                'seasonal_periods': seasonal_periods,
                                'initialization_method': 'heuristic',
                                'transform': config.get('transform'),
                                'trend': config.get('trend', 'add'),
                                'seasonal': config.get('seasonal', 'add'),
                                'damped_trend': config.get('damped_trend', False),
                                'damping_trend': config.get('damping_trend'),
                                'bias_correction': config.get('bias_correction'),
                                'smearing_factor': correction_factor
                            }
                            best_metrics = {
                                'mae': mae,
                                'rmse': rmse,
                                'mape': mape,
                                'mse': mse,
                                'valid_models': valid_models + 1
                            }
                            valid_models += 1

                    except Exception as e:
                        print(
                            "❌ Grid search fit failed for "
                            f"{param_name} | seasonal_periods={seasonal_periods}, "
                            f"alpha={alpha}, beta={beta}, gamma={gamma}: "
                            f"{type(e).__name__}: {e}"
                        )
                        continue
    
    return best_params, best_metrics

def run_optimized_hw_analysis(
    collection_name,
    target_column,
    save_collection="holt-winter",
    config_id=None,
    append_column_id=True,
    client=None,
    start_date=None,
    end_date=None
):
    """
    Fungsi Holt-Winter yang dinamis berdasarkan parameter dari forecast_config
    """
    print(f"=== Start Holt-Winter Analysis for {collection_name}.{target_column} ===")
    
    # MongoDB connection
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
        # Fetch data dari collection yang ditentukan
        source_data = list(db[collection_name].find().sort("Date", 1))
        print(f"Fetched {len(source_data)} records from {collection_name}")
        
        if not source_data:
            raise ValueError(f"No data found in collection {collection_name}")
        
        # Prepare DataFrame
        df = pd.DataFrame(source_data)
        
        # Cek apakah ada kolom Date/timestamp
        date_column = None
        for col in ['Date', 'date', 'timestamp', 'Timestamp']:
            if col in df.columns:
                date_column = col
                break
        
        if date_column is None:
            raise ValueError(f"No date column found in {collection_name}")
        
        # Set timestamp index
        df['timestamp'] = pd.to_datetime(df[date_column])
        df.set_index('timestamp', inplace=True)

        # Pastikan indeks harian tanpa duplikasi
        df = df[~df.index.duplicated(keep='first')]
        date_range = pd.date_range(start=df.index[0], end=df.index[-1], freq='D')
        missing_dates = date_range.difference(df.index)
        print(f"Missing dates: {missing_dates}")

        # Cek apakah target column ada
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in {collection_name}")

        df = df.reindex(date_range)

        if target_column in ["RR", "RR_imputed", "PRECTOTCORR"]:
            df[target_column] = df[target_column].fillna(0)
        elif target_column in ["RH2M", "T2M", "ALLSKY_SFC_SW_DWN"]:
            df[target_column] = df[target_column].interpolate(
                method="time",
                limit_direction="both"
            )
        else:
            df[target_column] = df[target_column].fillna(0)

        print(f"Data range: {df.index[0]} to {df.index[-1]}")

        config = get_param_model_config(target_column)
        
        # Get data (tanpa preprocessing karena data sudah bersih)
        param_data = df[target_column].dropna()
        
        if len(param_data) < 100:
            raise ValueError(f"Insufficient data for {target_column}")
        
        # Grid search
        best_params, error_metrics = grid_search_hw_params(param_data, target_column)
        
        if best_params is None:
            raise ValueError(f"No valid model found for {target_column}")

        print(f"Selected HW params for {target_column}: {best_params}")
        
        # Fit final model
        final_model = fit_robust_model(param_data, best_params, target_column)
        
        if final_model is None:
            raise ValueError(f"Failed to fit final model for {target_column}")
        
        if start_date is not None and end_date is not None:
            forecast_start_date = pd.to_datetime(start_date)
            forecast_end_date = pd.to_datetime(end_date)
        else:
            forecast_start_date = df.index[-1] + pd.Timedelta(days=1)
            forecast_end_date = forecast_start_date + pd.Timedelta(days=364)

        forecast_dates = pd.date_range(
            start=forecast_start_date,
            end=forecast_end_date,
            freq='D'
        )
        forecast_days = len(forecast_dates)
        
        print(f"Forecast horizon: {forecast_days} days")
        
        # Generate forecast
        print(f"Generating forecast for {target_column}...")
        try:
            forecast = final_model.forecast(steps=forecast_days)
            
            if forecast is None or len(forecast) == 0:
                raise ValueError("Forecast result is empty")
            
            if hasattr(forecast, 'values'):
                forecast = forecast.values
            
            forecast = np.array(forecast).flatten()
            
            if np.isnan(forecast).any() or np.isinf(forecast).any():
                raise ValueError("Forecast contains NaN or infinite values")

            forecast = inverse_transform_forecast(
                forecast,
                config.get("transform"),
                best_params.get("smearing_factor", 1.0)
            )

            if np.isnan(forecast).any() or np.isinf(forecast).any():
                raise ValueError("Forecast contains NaN or infinite values after inverse transform")
            
            # Post-process forecast
            forecast = post_process_forecast(forecast, target_column)
            
            print(f"✓ {target_column} forecast completed")
            print(f"  Forecast range: {forecast.min():.3f} to {forecast.max():.3f}")
            
        except Exception as e:
            raise ValueError(f"Forecast generation failed: {str(e)}")
        
        # Prepare forecast documents
        forecast_docs = []
        
        try:
            for i, forecast_date in enumerate(forecast_dates):
                forecast_date_only = datetime.strptime(forecast_date.strftime('%Y-%m-%d'), '%Y-%m-%d')
                
                forecast_value = float(forecast[i])
                if np.isnan(forecast_value) or np.isinf(forecast_value):
                    print(f"Warning: Skipping invalid forecast value at index {i}")
                    continue

                doc = {
                    "forecast_date": forecast_date_only,
                    "timestamp": datetime.now().isoformat(),
                    "source_collection": collection_name,
                    "config_id": config_id,
                    "parameters": {
                        target_column: {
                            "forecast_value": forecast_value,
                            "model_metadata": {
                                "alpha": best_params["alpha"],
                                "beta": best_params["beta"],
                                "gamma": best_params["gamma"],
                                "use_seasonal": best_params.get("use_seasonal", True),
                                "seasonal_periods": best_params.get("seasonal_periods", 365),
                                "initialization_method": best_params.get("initialization_method", "heuristic"),
                                "transform": best_params.get("transform", config.get("transform")),
                                "damped_trend": best_params.get("damped_trend", config.get("damped_trend", False)),
                                "damping_trend": best_params.get("damping_trend", config.get("damping_trend")),
                                "bias_correction": best_params.get("bias_correction", config.get("bias_correction")),
                                "smearing_factor": best_params.get("smearing_factor"),
                                "trend": best_params.get("trend", config.get("trend")),
                                "seasonal": best_params.get("seasonal", config.get("seasonal"))
                            }
                        }
                    }
                }

                if append_column_id:
                    doc["column_id"] = f"{collection_name}_{target_column}"

                forecast_docs.append(doc)
            
        except Exception as e:
            raise ValueError(f"Error preparing forecast documents: {str(e)}")
        
        # Upsert ke collection
        upsert_count = 0
        for doc in forecast_docs:
            set_fields = {
                f"parameters.{target_column}": doc["parameters"][target_column],
                "timestamp": doc["timestamp"],
                "source_collection": doc["source_collection"],
                "config_id": doc["config_id"]
            }

            if append_column_id and "column_id" in doc:
                set_fields["column_id"] = doc["column_id"]

            result = db[save_collection].update_one(
                {
                    "forecast_date": doc["forecast_date"],
                    "config_id": doc["config_id"]
                },
                {"$set": set_fields},
                upsert=True
            )
            if result.upserted_id or result.modified_count > 0:
                upsert_count += 1
        
        print(f"✓ Upserted {upsert_count} forecast documents to {save_collection}")
        
        result_summary = {
            "collection_name": collection_name,
            "target_column": target_column,
            "forecast_days": len(forecast_docs),
            "documents_processed": upsert_count,
            "save_collection": save_collection,
            "model_params": best_params,
            "error_metrics": error_metrics,  # Menambahkan MAE, RMSE, MAPE, MSE
            "forecast_range": {
                "start": forecast_start_date.strftime("%Y-%m-%d"),
                "end": forecast_end_date.strftime("%Y-%m-%d"),
                "min": float(forecast.min()),
                "max": float(forecast.max())
            }
        }
        
        print(f"✓ Analysis completed for {collection_name}.{target_column}")
        return result_summary
        
    except Exception as e:
        print(f"❌ Error in Holt-Winter analysis: {str(e)}")
        raise e
    
    finally:
        if should_close_client:
            client.close()

if __name__ == "__main__":
    run_optimized_hw_analysis(
        collection_name="bmkg-data",
        target_column="RR"
    )
