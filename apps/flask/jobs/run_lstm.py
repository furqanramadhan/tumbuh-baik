from pymongo import MongoClient
from pymongo import ReturnDocument
from datetime import datetime
import time
import traceback
import os
from flask import jsonify
from dotenv import load_dotenv
import pandas as pd
from bson import ObjectId


load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
if not MONGO_URI:
    raise ValueError("No MONGODB_URI set in environment variables!")

DB_NAME = os.getenv("MONGODB_DB_NAME", "tugas_akhir")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

def convert_objectid(obj):
    """Convert ObjectId to string for JSON serialization"""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, list):
        return [convert_objectid(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_objectid(value) for key, value in obj.items()}
    elif hasattr(obj, '__dict__'):
        return convert_objectid(obj.__dict__)
    else:
        return obj

def is_valid_column(collection_name, column_name, client):
    """Check if column is suitable for forecasting (not ID, date, etc.)"""
    forbidden_keywords = ["id", "date", "time", "timestamp", "_id"]
    if any(keyword in column_name.lower() for keyword in forbidden_keywords):
        return False, f"Column {column_name} contains forbidden keyword for forecasting"
    
    try:
        sample_doc = client["tugas_akhir"][collection_name].find_one({column_name: {"$exists": True}})
        if sample_doc and not isinstance(sample_doc[column_name], (int, float)):
            return False, f"Column {column_name} contains non-numeric data"
        return True, None
    except Exception as e:
        return False, f"Error checking column {column_name}: {str(e)}"

def run_lstm_from_config_core(config_id=None, mongo_uri=None, db_name=None):
    """
    Run one pending LSTM config without depending on Flask response context.
    This is safe to call from a background child process because it creates
    its own MongoClient instead of reusing the Gunicorn parent connection.
    """
    local_client = MongoClient(mongo_uri or MONGO_URI)
    local_db = local_client[db_name or DB_NAME]
    config = None

    try:
        from lstm.lstm_dynamic_2 import run_lstm_analysis

        # Kosongkan collection temp-lstm dan decompose di awal
        local_db["temp-lstm"].delete_many({})
        local_db["decompose-lstm-temp"].delete_many({})  # ✅ TAMBAH

        if config_id:
            try:
                config_object_id = ObjectId(config_id)
            except Exception:
                return {"error": f"Invalid LSTM config_id: {config_id}"}, 400
            config_filter = {"_id": config_object_id, "status": {"$in": ["pending", "running"]}}
        else:
            config_filter = {"status": "pending"}

        config = local_db.lstm_configs.find_one_and_update(
            config_filter,
            {"$set": {"status": "running", "startedAt": datetime.now(), "updatedAt": datetime.now()}},
            return_document=ReturnDocument.AFTER
        )
        
        if not config:
            return {"message": "No pending LSTM config found."}, 404

        config_id = str(config["_id"])
        
        # Ambil info kolom yang akan dianalisis
        name = config.get("name", f"lstm_forecast_{int(time.time())}")
        columns = config.get("columns", [])
        forecast_coll = config.get("forecastResultCollection", "lstm-forecast")
        config_id = str(config["_id"])

        start_date_str = config.get("startDate")
        end_date_str = config.get("endDate")

        if not start_date_str or not end_date_str:
            error_msg = "startDate and endDate must be specified in the config."
            local_db.lstm_configs.update_one(
                {"_id": config["_id"]},
                {"$set": {"status": "failed", "error_message": error_msg, "updatedAt": datetime.now()}}
            )
            return {"error": error_msg}, 400
        
        try:
            start_date= pd.to_datetime(start_date_str).date()
            end_date = pd.to_datetime(end_date_str).date()
            print(f"[INFO] Using startDate: {start_date}, endDate: {end_date}")
        except Exception as e:
            error_msg = f"Invalid date format for startDate or endDate: {str(e)}"
            local_db.lstm_configs.update_one(
                {"_id": config["_id"]},
                {"$set": {"status": "failed", "error_message": error_msg, "updatedAt": datetime.now()}}
            )
            return {"error": error_msg}, 400

        # Validasi kolom
        for item in columns:
            collection = item["collectionName"]
            column = item["columnName"]
            is_valid, error_msg = is_valid_column(collection, column, local_client)
            if not is_valid:
                local_db.lstm_configs.update_one(
                    {"_id": config["_id"]},
                    {"$set": {"status": "failed", "error_message": error_msg, "updatedAt": datetime.now()}}
                )
                return {"error": error_msg}, 400
        
        results = []
        forecast_data = {}
        decompose_data = {}  # ✅ TAMBAH untuk decompose
        error_metrics_list = []

        for item in columns:
            collection = item["collectionName"]
            column = item["columnName"]
            
            print(f"[INFO] Processing LSTM for {collection} - {column}")
            
            try:
                # Jalankan analisis LSTM
                result = run_lstm_analysis(
                    collection_name=collection,
                    target_column=column,
                    save_collection="temp-lstm",
                    config_id=config_id,
                    append_column_id=True,
                    client=local_client,
                    start_date=start_date,
                    end_date=end_date
                )
                results.append(result)

                if result.get("status") == "error":
                    raise ValueError(result.get("error", "Unknown LSTM error"))

                # Simpan metrik evaluasi untuk kolom ini
                if result.get("error_metrics"):
                    metrics = result["error_metrics"]
                    error_metrics_list.append({
                        "collectionName": collection,
                        "columnName": column,
                        "metrics_lstm": {
                            "mae": metrics.get("mae"),
                            "mse": metrics.get("mse"),
                            "rmse": metrics.get("rmse"),
                            "mape": metrics.get("mape"),
                            "r2": metrics.get("r2"),
                            "directional_accuracy": metrics.get("directional_accuracy"),
                            "recursive_mae": metrics.get("recursive_mae"),
                            "recursive_rmse": metrics.get("recursive_rmse"),
                            "recursive_mape": metrics.get("recursive_mape"),
                            "log_bias_correction": metrics.get("log_bias_correction"),
                            "log_bias_correction_raw": metrics.get("log_bias_correction_raw"),
                            "log_bias_correction_mode": metrics.get("log_bias_correction_mode"),
                            "log_bias_correction_sample_count": metrics.get("log_bias_correction_sample_count"),
                            "log_bias_correction_wet_day_threshold": metrics.get("log_bias_correction_wet_day_threshold"),
                            "log_bias_correction_negative_clamped": metrics.get("log_bias_correction_negative_clamped"),
                            "log_bias_correction_positive_capped": metrics.get("log_bias_correction_positive_capped"),
                            "rmse_degradation_pct": result.get("horizon_confidence", {}).get("rmse_degradation_pct"),
                            "horizon_confidence": result.get("horizon_confidence"),
                            "metric_context": result.get("metric_context"),
                            "warnings": result.get("warnings", []),
                        }
                    })
                
                # Ambil hasil forecast untuk digabung
                temp_forecasts = list(local_db["temp-lstm"].find({"config_id": config_id}))
                
                for forecast_doc in temp_forecasts:
                    forecast_date = pd.to_datetime(forecast_doc["forecast_date"]).strftime("%Y-%m-%d")
                    
                    if forecast_date not in forecast_data:
                        forecast_data[forecast_date] = {
                            "forecast_date": forecast_date,
                            "timestamp": datetime.now().isoformat(),
                            "config_id": config_id,
                            "model_type": "LSTM",
                            "parameters": {}
                        }
                    
                    # Tambahkan parameter ke struktur gabungan
                    if "parameters" in forecast_doc:
                        forecast_data[forecast_date]["parameters"].update(
                            forecast_doc["parameters"]
                        )
                
                # ✅ TAMBAH: Ambil hasil decompose untuk digabung
                temp_decompose = list(local_db["decompose-lstm-temp"].find({"config_id": config_id}))
                
                for decompose_doc in temp_decompose:
                    date_str = decompose_doc["date"]
                    
                    if date_str not in decompose_data:
                        decompose_data[date_str] = {
                            "date": date_str,
                            "timestamp": datetime.now().isoformat(),
                            "config_id": config_id,
                            "parameters": {}
                        }
                    
                    if "parameters" in decompose_doc:
                        decompose_data[date_str]["parameters"].update(
                            decompose_doc["parameters"]
                        )
                
            except Exception as e:
                error_msg = f"LSTM failed for {collection}:{column} → {str(e)}"
                local_db.lstm_configs.update_one(
                    {"_id": config["_id"]},
                    {"$set": {"status": "failed", "error_message": error_msg, "updatedAt": datetime.now()}}
                )
                traceback.print_exc()
                return {"error": error_msg}, 500
        
        # Simpan hasil gabungan forecast ke collection final
        if forecast_data:
            # Hapus data lama untuk config ini
            local_db["lstm-forecast"].delete_many({"config_id": config_id})
            
            # Insert data gabungan
            combined_docs = list(forecast_data.values())
            local_db["lstm-forecast"].insert_many(combined_docs)
            
            print(f"✓ Inserted {len(combined_docs)} combined LSTM forecast documents")
        
        # ✅ TAMBAH: Simpan hasil gabungan decompose ke collection final
        if decompose_data:
            local_db["decompose-lstm"].delete_many({"config_id": config_id})
            combined_decompose = list(decompose_data.values())
            local_db["decompose-lstm"].insert_many(combined_decompose)
            print(f"✓ Inserted {len(combined_decompose)} combined decompose documents")
        
        # Bersihkan collection temporary
        local_db["temp-lstm"].delete_many({})
        local_db["decompose-lstm-temp"].delete_many({})  # ✅ TAMBAH
        
        # Update status config dan simpan error metrics
        update_data = {
            "status": "done",
            "error_metrics": error_metrics_list,
            "finishedAt": datetime.now(),
            "updatedAt": datetime.now()
        }

        local_db.lstm_configs.update_one(
            {"_id": config["_id"]},
            {"$set": update_data}
        )

        return {
            "message": f"LSTM Forecasting completed for config: {name}",
            "config_id": config_id,
            "forecastResultCollection": forecast_coll,
            "model_type": "LSTM",
            "results": convert_objectid(results),
            "total_forecast_dates": len(forecast_data),
            "total_decompose_dates": len(decompose_data),  # ✅ TAMBAH
            "error_metrics": error_metrics_list
        }, 200
        
    except Exception as e:
        error_msg = f"Internal server error: {str(e)}"
        if config is not None:
            local_db.lstm_configs.update_one(
                {"_id": config["_id"]},
                {"$set": {"status": "failed", "error_message": error_msg, "updatedAt": datetime.now()}}
            )
        traceback.print_exc()
        return {"error": error_msg}, 500
    finally:
        local_client.close()

def run_lstm_from_config(config_id=None):
    payload, status_code = run_lstm_from_config_core(config_id=config_id)
    return jsonify(payload), status_code

def run_lstm_background_worker(config_id=None, mongo_uri=None, db_name=None):
    """Entry point for the detached LSTM worker process."""
    try:
        import torch
        safe_threads = int(os.getenv("LSTM_TORCH_THREADS", "2"))
        safe_threads = max(1, safe_threads)
        torch.set_num_threads(safe_threads)
        torch.set_num_interop_threads(safe_threads)
        print(f"[INFO] LSTM background worker using {safe_threads} torch CPU threads")
    except Exception as e:
        print(f"[WARN] Failed to configure torch CPU threads: {e}")

    payload, status_code = run_lstm_from_config_core(
        config_id=config_id,
        mongo_uri=mongo_uri,
        db_name=db_name
    )
    print(f"[INFO] LSTM background worker finished with status {status_code}: {payload}")

def get_lstm_status(config_id):
    """Get LSTM execution status from lstm_configs collection"""
    try:
        from bson import ObjectId
        status_doc = db["lstm_configs"].find_one({"_id": ObjectId(config_id)})
        if status_doc:
            return convert_objectid(status_doc)
        return None
    except Exception as e:
        print(f"❌ Error getting LSTM status: {str(e)}")
        return None

def get_all_lstm_statuses():
    """Get all LSTM execution statuses from lstm_configs collection"""
    try:
        statuses = list(db["lstm_configs"].find().sort("updatedAt", -1))
        return convert_objectid(statuses)
    except Exception as e:
        print(f"❌ Error getting all LSTM statuses: {str(e)}")
        return []

# Function untuk testing langsung
def test_lstm_forecast():
    """Function untuk testing LSTM forecast secara langsung"""
    try:
        # Test dengan data dummy config
        test_config = {
            "_id": "test_lstm_config",
            "name": "Test LSTM Forecast",
            "columns": [
                {"collectionName": "bmkg-data", "columnName": "RR"},
                {"collectionName": "bmkg-data", "columnName": "RH_AVG"}
            ],
            "forecastResultCollection": "lstm-forecast"
        }

        # Insert test config ke lstm_configs
        db.lstm_configs.delete_many({"_id": "test_lstm_config"})
        db.lstm_configs.insert_one({**test_config, "status": "pending"})

        # Run LSTM forecast
        result = run_lstm_from_config()
        print("LSTM Test Result:", result)
        
        # Check final status
        final_status = get_lstm_status("test_lstm_config")
        print("Final LSTM Status:", final_status)
        
        return result
        
    except Exception as e:
        print(f"Error in LSTM test: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Untuk testing langsung
    print("🧪 Starting LSTM test...")
    result = test_lstm_forecast()
    if result:
        print("✅ LSTM test completed successfully!")
    else:
        print("❌ LSTM test failed!")
    print("🏁 Test finished.")
