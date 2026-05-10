from pymongo import MongoClient
from flask import Flask, jsonify, request, Blueprint, Response, stream_with_context, current_app
from flask_cors import CORS
import os
from helpers.objectid_converter import convert_objectid
from jobs.run_forecast_from_config import run_forecast_from_config
from jobs.run_lstm import run_lstm_from_config
from bson import ObjectId
import json
from bson.json_util import dumps
import requests
from dotenv import load_dotenv
from datetime import datetime
import logging
import traceback
import tempfile
import shutil
from queue import Queue
import threading
import time
from typing import Dict, Any
from preprocessing.buoys.preprocessing_buoys import (
    DataValidator,
    BuoyPreprocessor,
    MongoDataSaver,
    PreprocessingError
)
from preprocessing.nasa.preprocessing_nasa import (
    NasaPreprocessor,
    NasaDataValidator,
    NasaPreprocessingError,
)
from preprocessing.bmkg.preprocessing_bmkg import (
    BmkgPreprocessor,
    BmkgDataValidator,
    BmkgPreprocessingError,
)

from preprocessing.convert.xlsx_to_csv import convert_single_xlsx
from preprocessing.convert.xlsx_merge_csv import merge_multiple_xlsx
from routes.scheduler_routes import scheduler_bp
from middleware.auth_middleware import require_auth
from models.scheduler_logs import SchedulerLog

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)

cors_origins = os.getenv("CORS_ORIGINS", "").split(",")
CORS(app, 
     origins=cors_origins,
     supports_credentials=True,
     allow_headers=[
         "Content-Type",
         "Authorization",
         "Cookie",
         "ngrok-skip-browser-warning",
     ],
     expose_headers=["Content-Type"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

mongo_uri = os.getenv("MONGODB_URI")
db_name = os.getenv("MONGODB_DB_NAME", "tugas_akhir")
mongo_client = MongoClient(mongo_uri)
db = mongo_client[db_name]
app.config['MONGO_DB'] = db

preprocessing_bp = Blueprint('preprocessing', __name__, url_prefix="/api/v1")

# === Routes ===
@app.route("/")
def home():
    return jsonify({"message": "Flask Holt-Winter API is running!"})


@app.route("/run-forecast", methods=["POST"])
def run_forecast():
    return run_forecast_from_config()

@app.route("/run-lstm", methods=["POST"])
def run_lstm():
    print("Received request to run LSTM forecast")
    return run_lstm_from_config()

@app.route("/run-lstm" , methods=["GET"])
def run_lstm_get():
    print("Received GET request to run LSTM forecast")
    return jsonify({"message": "Use POST method to run LSTM forecast"}), 200

@app.route("/check-mongodb")
def check_mongodb():
    try:
        collections = db.list_collection_names()
        return jsonify({
            "status": "connected",
            "collections": collections
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Dataset routes
@app.route("/api/v1/datasets", methods=["GET"])
def get_all_datasets():
    """Get all datasets metadata"""
    try:
        # Check if DatasetMeta collection exists
        if "dataset_meta" in db.list_collection_names():
            datasets = list(db["dataset_meta"].find().sort("uploadDate", -1))
        elif "DatasetMeta" in db.list_collection_names():
            datasets = list(db["DatasetMeta"].find().sort("uploadDate", -1))
        else:
            # If no metadata collection exists, create a simple one based on available collections
            all_collections = db.list_collection_names()
            # Filter out system collections
            exclude_prefixes = ['system.', 'user', 'session', 'account']
            filtered_collections = [coll for coll in all_collections 
                                   if not any(coll.startswith(prefix) for prefix in exclude_prefixes)]
            
            datasets = []
            for coll in filtered_collections:
                # Create a simple metadata entry for each collection
                collection_type = "buoy" if coll.startswith("buoys_") else "general"
                datasets.append({
                    "collectionName": coll,
                    "title": coll.replace("_", " ").title(),
                    "description": f"Data collection for {coll}",
                    "dataType": collection_type,
                    "uploadDate": datetime.now(),
                    "generated": False
                })
        
        # Convert ObjectId to string for JSON serialization
        return json.loads(dumps({"data": datasets})), 200
    except Exception as e:
        return jsonify({"message": f"Server error: {str(e)}"}), 500
    
@app.route("/api/v1/dataset/<collection_name>", methods=["GET"])
def get_dataset_by_collection(collection_name):
    """Get dataset by collection name with pagination"""
    try:
        # Get query parameters
        page = int(request.args.get("page", 1))
        page_size = int(request.args.get("pageSize", 10))
        sort_by = request.args.get("sortBy", "Date") 
        sort_order = request.args.get("sortOrder", "desc")
        
        # Check if collection exists
        if collection_name not in db.list_collection_names():
            return jsonify({"message": f"Collection '{collection_name}' not found"}), 404
            
        # Get metadata for the collection (if it exists)
        meta = None
        if "dataset_meta" in db.list_collection_names():
            meta = db["dataset_meta"].find_one({"collectionName": collection_name})
        elif "DatasetMeta" in db.list_collection_names():
            meta = db["DatasetMeta"].find_one({"collectionName": collection_name})
            
        # If no metadata, create a simple one
        if not meta:
            # Determine data type based on collection name
            data_type = "buoy" if collection_name.startswith("buoys_") else "general"
            
            # Extract more info for buoy collections
            location = None
            if data_type == "buoy":
                location_parts = collection_name.split("_")
                if len(location_parts) >= 3:
                    location = location_parts[-1]
            
            meta = {
                "collectionName": collection_name,
                "title": collection_name.replace("_", " ").title(),
                "description": f"Data collection for {collection_name}",
                "dataType": data_type,
                "location": location
            }
            
        # Create sort query
        sort_direction = -1 if sort_order == "desc" else 1
        # Ensure the sort field exists in the collection
        sample_doc = db[collection_name].find_one()
        if sample_doc and sort_by not in sample_doc and sort_by != "_id":
            # If the specified sort_by doesn't exist, default to _id
            sort_by = "_id"
            
        sort_query = [(sort_by, sort_direction)]
        
        # Count total documents
        total_data = db[collection_name].count_documents({})
        
        # Get paginated data
        data = list(db[collection_name].find()
                   .sort(sort_query)
                   .skip((page - 1) * page_size)
                   .limit(page_size))
        
        # Get collection stats
        stats = {
            "name": collection_name,
            "total_records": total_data,
            "fields": list(sample_doc.keys()) if sample_doc else []
        }
        
        # Prepare response
        response = {
            "message": "Success",
            "data": {
                "meta": meta,
                "stats": stats,
                "items": data,
                "total": total_data,
                "currentPage": page,
                "totalPages": (total_data + page_size - 1) // page_size,
                "pageSize": page_size,
                "sortBy": sort_by,
                "sortOrder": sort_order
            }
        }
        
        # Convert ObjectId to string for JSON serialization
        return json.loads(dumps(response)), 200
        
    except Exception as e:
        print(f"Error fetching dataset: {str(e)}")
        return jsonify({"message": f"Server error: {str(e)}"}), 500
    

@app.route("/api/v1/datasets/<collection_name>/<object_id>", methods=["GET"])
def get_dataset_document_by_id(collection_name, object_id):
    """Get a specific document from a dataset collection by its ObjectID"""
    try:
        # Validate ObjectId
        if not ObjectId.is_valid(object_id):
            return jsonify({"message": "Invalid ObjectID format"}), 400
            
        # Get the document
        document = db[collection_name].find_one({"_id": ObjectId(object_id)})
        
        if not document:
            return jsonify({"message": "Document not found"}), 404
            
        # Convert ObjectId to string for JSON serialization
        return json.loads(dumps({"data": document})), 200
        
    except Exception as e:
        return jsonify({"message": f"Server error: {str(e)}"}), 500
    
@preprocessing_bp.route("/preprocess/<collection_name>", methods=["POST"])
def preprocess_dataset(collection_name):
    """Preprocess a dataset using the new class-based system"""
    db = request.app.config['MONGO_DB']
    
    try:
        # Check if collection exists
        if collection_name not in db.list_collection_names():
            return jsonify({"message": f"Collection '{collection_name}' not found"}), 404
        
        # Generate a job ID to track this preprocessing task
        job_id = str(ObjectId())
        
        # Create a status document to track progress
        preprocessing_status = {
            "job_id": job_id,
            "collection": collection_name,
            "status": "processing",
            "startTime": datetime.now(),
            "steps": [
                {"name": "started", "completed": True, "timestamp": datetime.now()}
            ]
        }
        
        # save status to MongoDB
        db["preprocessing_jobs"].insert_one(preprocessing_status)
        
        # Get preprocessing options from request body (FIXED)
        options = request.get_json(silent=True) or {}
        
        # Extract location code from collection name if available
        location_code = None
        if collection_name.startswith("buoys_"):
            location_parts = collection_name.split("_")
            if len(location_parts) >= 3:
                location_code = location_parts[-1]
        
        try:
            # Update status to indicate validation started
            db["preprocessing_jobs"].update_one(
                {"job_id": job_id},
                {"$push": {"steps": {"name": "validating_data", "completed": True, "timestamp": datetime.now()}}}
            )
            
            # Validate the dataset
            validator = DataValidator()
            validation_result = validator.validate_dataset(db, collection_name)
            
            if validation_result and not validation_result.get('valid', True):
                return jsonify({
                    "message": "Dataset validation failed",
                    "job_id": job_id,
                    "errors": validation_result.get('errors', []),
                    "warnings": validation_result.get('warnings', [])
                }), 400
            
            # Update status
            db["preprocessing_jobs"].update_one(
                {"job_id": job_id},
                {"$push": {"steps": {"name": "preprocessing_started", "completed": True, "timestamp": datetime.now()}}}
            )
            
            # Initialize preprocessor
            preprocessor = BuoyPreprocessor(db, collection_name, location_code)
            
            # Run preprocessing
            preprocessing_results = preprocessor.preprocess(options)
            
            # Update status
            db["preprocessing_jobs"].update_one(
                {"job_id": job_id},
                {"$push": {"steps": {"name": "preprocessing_completed", "completed": True, "timestamp": datetime.now()}}}
            )
            
            # Save processed data back to MongoDB
            saver = MongoDataSaver()
            save_results = saver.save_processed_data(db, preprocessing_results, collection_name)
            
            # Update job status
            db["preprocessing_jobs"].update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "status": "completed", 
                        "endTime": datetime.now(), 
                        "metadata": save_results
                    },
                    "$push": {"steps": {"name": "results_saved", "completed": True, "timestamp": datetime.now()}}
                }
            )
            
            # Return success response
            return jsonify({
                "message": "Preprocessing completed successfully",
                "job_id": job_id,
                "collection": collection_name,
                "processedCollections": save_results.get("processedCollections", []),
                "recordCounts": save_results.get("recordCounts", {}),
                "status": "preprocessed",
                "warnings": preprocessing_results.get("warnings", [])
            }), 200
            
        except PreprocessingError as pe:
            error_details = str(pe)
            
            # Update status with error
            db["preprocessing_jobs"].update_one(
                {"job_id": job_id},
                {
                    "$set": {"status": "failed", "endTime": datetime.now(), "error": error_details},
                    "$push": {"steps": {"name": "processing_failed", "completed": False, "timestamp": datetime.now(), "error": error_details}}
                }
            )
            
            return jsonify({
                "message": f"Preprocessing error: {error_details}",
                "job_id": job_id,
                "status": "failed"
            }), 500
            
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"message": f"Server error: {str(e)}"}), 500

@preprocessing_bp.route("/preprocess/status/<job_id>", methods=["GET"])
def get_preprocessing_status(job_id):
    """Get status of a preprocessing job"""
    db = request.app.config['MONGO_DB']
    
    try:
        # Find job in database
        job = db["preprocessing_jobs"].find_one({"job_id": job_id})
        
        if not job:
            return jsonify({"message": f"Job with ID '{job_id}' not found"}), 404
            
        # Format response
        response = {
            "job_id": job["job_id"],
            "collection": job["collection"],
            "status": job["status"],
            "started": job.get("startTime"),
            "completed": job.get("endTime"),
            "steps": job.get("steps", []),
        }
        
        # Add error details if job failed
        if job["status"] == "failed":
            response["error"] = job.get("error")
            
        # Add metadata if job completed
        if job["status"] == "completed":
            response["metadata"] = job.get("metadata")
            
        return jsonify({"message": "Success", "data": response}), 200
        
    except Exception as e:
        logger.error(f"Error fetching job status: {str(e)}")
        return jsonify({"message": f"Server error: {str(e)}"}), 500


@preprocessing_bp.route("/preprocess/nasa/<collection_name>", methods=["POST"])
def preprocess_nasa_dataset(collection_name):
    """Preprocessing a NASA POWER dataset"""
    db = current_app.config['MONGO_DB']
    
    try:
        # Check if collection exists
        if collection_name not in db.list_collection_names():
            return jsonify({"message": f"Collection '{collection_name}' not found"}), 404
        
        # Get total records
        total_records = db[collection_name].count_documents({})
        logger.info(f"Starting preprocessing for NASA dataset '{collection_name}' with {total_records} records")
        
        # Generate a job ID to track this preprocessing task
        job_id = str(ObjectId())
        
        # Create a status document to track progress
        preprocessing_status = {
            "job_id": job_id,
            "collection": collection_name,
            "type": "nasa-power",
            "status": "processing",
            "startTime": datetime.now(),
            "totalRecords": total_records,
            "processedRecords": 0,
            "steps": [
                {"name": "started", "completed": True, "timestamp": datetime.now()}
            ]
        }
        
        # Save status to MongoDB
        db["preprocessing_jobs"].insert_one(preprocessing_status)
        
        # GET preprocessing options from request body (FIXED)
        options = request.get_json(silent=True) or {}
        
        try:
            # Update status to indicate validation started
            db['preprocessing_jobs'].update_one(
                {"job_id": job_id},
                {"$push": {"steps": {"name": "validating_data", "completed": True, "timestamp": datetime.now()}}}
            )
            # Validate the dataset
            validator = NasaDataValidator()
            validation_result = validator.validate_dataset(db, collection_name)
            
            if validation_result and not validation_result.get('valid', True):
                db['preprocessing_jobs'].update_one(
                    {"job_id": job_id},
                    {"$set": {"status": "failed", "error": "Validation failed"}}
                )
                return jsonify({
                    "message": "Dataset validation failed",
                    "job_id": job_id,
                    "errors": validation_result.get('errors', []),
                    "warnings": validation_result.get('warnings', [])
                }), 400
                
            # Update status
            db['preprocessing_jobs'].update_one(
                {"job_id": job_id},
                {
                    "$push": {"steps": {"name": "loading_data", "completed": True, "timestamp": datetime.now()}},
                    "$set": {"status": "processing", "currentStep": "loading_data"}
                }
            )
            
            # Initialize preprocessor
            preprocessor = NasaPreprocessor(db, collection_name)
            
            # Update status for preprocessing started
            db["preprocessing_jobs"].update_one(
                {"job_id": job_id},
                {
                    "$push": {"steps": {"name": "preprocessing_started", "completed": True, "timestamp": datetime.now()}},
                    "$set": {"status": "processing", "currentStep": "preprocessing"}
                }
            )
            # Run preprocessing
            preprocessing_results = preprocessor.preprocess(options)
            
            # Update status for preprocessing completion
            db["preprocessing_jobs"].update_one(
                {"job_id": job_id},
                {
                    "$push": {"steps": {"name": "preprocessing_completed", "completed": True, "timestamp": datetime.now()}},
                    "$set": {
                        "processedRecords": preprocessing_results.get("recordCount", total_records),
                        "status": "saving_results",
                        "currentStep": "saving_results"
                    }
                }
            )
            # Final status update
            db["preprocessing_jobs"].update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "status": "completed", 
                        "endTime": datetime.now(), 
                        "metadata": preprocessing_results
                    },
                    "$push": {"steps": {"name": "results_saved", "completed": True, "timestamp": datetime.now()}}
                }
            )
            # Return success response
            return jsonify({
                "message": "NASA POWER preprocessing completed successfully",
                "job_id": job_id,
                "collection": collection_name,
                "cleanedCollection": preprocessing_results.get("cleanedCollection"),
                "preprocessedCollection": f"{collection_name}_cleaned",
                "status": "preprocessed",
                "recordCount": preprocessing_results.get("recordCount", 0),
                "warnings": preprocessing_results.get("warnings", [])
            }), 200
        except NasaPreprocessingError as pe:
            error_details = str(pe)
            logger.error(f"NasaPreprocessingError: {error_details}")
            
            # Update status with error
            db["preprocessing_jobs"].update_one(
                {"job_id": job_id},
                {
                    "$set": {"status": "failed", "endTime": datetime.now(), "error": error_details},
                    "$push": {"steps": {"name": "processing_failed", "completed": False, "timestamp": datetime.now(), "error": error_details}}
                }
            )
            
            return jsonify({
                "message": f"Preprocessing error: {error_details}",
                "job_id": job_id,
                "status": "failed"
            }), 500
    
    except Exception as e:
        logger.error(f"Error in NASA preprocessing: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"message": f"Server error: {str(e)}"}), 500

@preprocessing_bp.route("/preprocess/nasa/<collection_name>/stream", methods=["GET"])
def preprocess_nasa_stream(collection_name):
    """
    SSE endpoint for real-time preprocessing logs
    Stream preprocessing progress to frontend
    """
    def generate():
        # Create log queue for this session
        log_queue = Queue()
        session_id = f"{collection_name}_{int(time.time())}"
        
        # Setup custom log handler that captures logs
        class SSELogHandler(logging.Handler):
            def emit(self, record):
                try:
                    log_entry = self.format(record)
                    # Parse progress information
                    if "PROGRESS:" in log_entry:
                        try:
                            # Extract PROGRESS:percentage:stage:message
                            progress_part = log_entry.split("PROGRESS:")[1]
                            parts = progress_part.split(":", 2)  # Split into max 3 parts
                            
                            if len(parts) >= 3:
                                # Parse percentage with error handling
                                try:
                                    percentage_str = parts[0].strip()
                                    percentage = int(percentage_str) if percentage_str.isdigit() else 0
                                except (ValueError, AttributeError):
                                    percentage = 0
                                    
                                log_queue.put({
                                    'type': 'progress',
                                    'percentage': percentage,
                                    'stage': parts[1].strip() if len(parts) > 1 else 'processing',
                                    'message': parts[2].strip() if len(parts) > 2 else 'Processing...'
                                })
                            else:
                                # If parsing fails, treat as regular log
                                log_queue.put({
                                    'type': 'log',
                                    'level': record.levelname,
                                    'message': log_entry,
                                    'timestamp': time.time()
                                })
                        except (ValueError, IndexError) as e:
                            # If parsing fails, treat as regular log
                            log_queue.put({
                                'type': 'log',
                                'level': record.levelname,
                                'message': log_entry,
                                'timestamp': time.time()
                            })
                    else:
                        log_queue.put({
                            'type': 'log',
                            'level': record.levelname,
                            'message': log_entry,
                            'timestamp': time.time()
                        })
                except Exception as e:
                    logger.error(f"SSE Log Handler error: {str(e)}")
        
        # Add handler to preprocessing logger
        sse_handler = SSELogHandler()
        sse_handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
        
        preprocessing_logger = logging.getLogger('preprocessing.nasa.preprocessing_nasa')
        preprocessing_logger.addHandler(sse_handler)
        preprocessing_logger.setLevel(logging.INFO)
        
        try:
            # Send initial connection message
            yield f"data: {json.dumps({'type': 'connected', 'session_id': session_id, 'collection': collection_name})}\n\n"
            
            # Get db instance from current request context
            db_instance = current_app.config['MONGO_DB']
            
            # Start preprocessing in background thread
            preprocessing_result = {'status': None, 'data': None, 'error': None}
            
            def run_preprocessing():
                try:
                    # Send starting message
                    log_queue.put({
                        'type': 'progress',
                        'stage': 'starting',
                        'percentage': 0,
                        'message': 'Initializing NASA POWER preprocessing...'
                    })
                    
                    # Run actual preprocessing
                    preprocessor = NasaPreprocessor(db_instance, collection_name)
                    result = preprocessor.preprocess()
                    
                    def sanitize_numpy(obj):
                        if isinstance(obj, dict):
                            return {key: sanitize_numpy(value) for key, value in obj.items()}
                        elif isinstance(obj, list):
                            return [sanitize_numpy(item) for item in obj]
                        elif hasattr(obj, 'item'):  # numpy scalar
                            return obj.item()
                        elif type(obj).__module__ == 'numpy':
                            if hasattr(obj, 'tolist'):
                                return obj.tolist()
                            return obj.item()
                        else:
                            return obj
                    
                    safe_result = sanitize_numpy(result)
                    
                    # Debug detailed logging result 
                    logger.info(f"Raw preprocessing result: {result}")
                    logger.info(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                    
                    # Store result
                    preprocessing_result['status'] = 'success'
                    preprocessing_result['data'] = safe_result
                    
                    safe_result = {
                        'recordCount': int(result['recordCount']) if 'recordCount' in result and result['recordCount'] is not None else 0,
                        'originalRecordCount': int(result['originalRecordCount']) if 'originalRecordCount' in result and result['originalRecordCount'] is not None else 0,
                        'cleanedCollection': result.get('cleanedCollection'),
                        'collection': result.get('collection'),
                        'message': result.get('message'),
                        'preprocessedCollections': result.get('preprocessedCollections', []),
                        'preprocessing_report': {
                            'outliers': {
                                'total_outliers': int(result.get('preprocessing_report', {}).get('outliers', {}).get('total_outliers', 0))
                            },
                            'quality_metrics': {
                                'completeness_percentage': float(result.get('preprocessing_report', {}).get('quality_metrics', {}).get('completeness_percentage', 0))
                            }
                        } if result.get('preprocessing_report') else None
                    }
                    
                    logger.info(f"📤 Sending safe result: {safe_result}")
                    
                    log_queue.put({
                        'type': 'complete',
                        'status': 'success',
                        'result': safe_result
                    })
                    
                except Exception as e:
                    logger.error(f"❌ Preprocessing error: {str(e)}")
                    logger.error(f"❌ Traceback: {traceback.format_exc()}")
                    
                    preprocessing_result['status'] = 'error'
                    preprocessing_result['error'] = str(e)
                    
                    log_queue.put({
                        'type': 'error',
                        'message': str(e),
                        'traceback': traceback.format_exc()
                    })
                finally:
                    logger.info("🏁 Preprocessing thread completed, stream will close naturally")
                    preprocessing_result['thread_complete'] = True
            
            # Start preprocessing thread
            thread = threading.Thread(target=run_preprocessing)
            thread.daemon = True
            thread.start()
            
            while True:
                try:
                    log_data = log_queue.get(timeout=1)
                    
                    # Send log to client as SSE
                    yield f"data: {json.dumps(log_data)}\n\n"
                    
                    # ✅ FIXED: After sending completion, wait briefly then close
                    if log_data.get('type') == 'complete':
                        
                        yield f": completion-sent\n\n"
                        time.sleep(1.0)  # 1 second delay
                        logger.info("🎯 Closing stream after completion processed")
                        break
                        
                except Exception:
                    if preprocessing_result.get('thread_complete'):
                        logger.info("🏁 Thread completed, closing stream")
                        break
                        
                    # Timeout - send keepalive
                    yield ": keepalive\n\n"
                    continue
        except GeneratorExit:
            # Client disconnected
            logger.info(f"Client disconnected from preprocessing stream: {collection_name}")
        finally:
            # Cleanup
            try:
                preprocessing_logger.removeHandler(sse_handler)
                logger.info(f"Cleaned up SSE handler for session {session_id}")
            except Exception as e:
                logger.error(f"Error during SSE cleanup: {str(e)}")
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
        }
    )

@preprocessing_bp.route("/preprocess/bmkg/<collection_name>", methods=["POST"])
def preprocess_bmkg_dataset(collection_name):
    """Preprocessing a BMKG dataset"""
    db = current_app.config['MONGO_DB']
    
    try:
        # Check if collection exists
        if collection_name not in db.list_collection_names():
            return jsonify({"message": f"Collection '{collection_name}' not found"}), 404
        
        # Get total records
        total_records = db[collection_name].count_documents({})
        logger.info(f"Starting preprocessing for BMKG dataset '{collection_name}' with {total_records} records")
        
        # Generate a job ID to track this preprocessing task
        job_id = str(ObjectId())
        
        # Create a status document to track progress
        preprocessing_status = {
            "job_id": job_id,
            "collection": collection_name,
            "type": "bmkg",
            "status": "processing",
            "startTime": datetime.now(),
            "totalRecords": total_records,
            "processedRecords": 0,
            "steps": [
                {"name": "started", "completed": True, "timestamp": datetime.now()}
            ]
        }
        
        # Save status to MongoDB
        db["preprocessing_jobs"].insert_one(preprocessing_status)
        
        # GET preprocessing options from request body (FIXED)
        options = request.get_json(silent=True) or {}
        
        try:
            # Update status to indicate validation started
            db['preprocessing_jobs'].update_one(
                {"job_id": job_id},
                {"$push": {"steps": {"name": "validating_data", "completed": True, "timestamp": datetime.now()}}}
            )
            
            # Validate the dataset
            validator = BmkgDataValidator()
            validation_result = validator.validate_dataset(db, collection_name)
            
            if validation_result and not validation_result.get('valid', True):
                db['preprocessing_jobs'].update_one(
                    {"job_id": job_id},
                    {"$set": {"status": "failed", "error": "Validation failed"}}
                )
                return jsonify({
                    "message": "Dataset validation failed",
                    "job_id": job_id,
                    "errors": validation_result.get('errors', []),
                    "warnings": validation_result.get('warnings', [])
                }), 400
                
            # Update status
            db['preprocessing_jobs'].update_one(
                {"job_id": job_id},
                {
                    "$push": {"steps": {"name": "loading_data", "completed": True, "timestamp": datetime.now()}},
                    "$set": {"status": "processing", "currentStep": "loading_data"}
                }
            )
            
            # Initialize preprocessor
            preprocessor = BmkgPreprocessor(db, collection_name)
            
            # Update status for preprocessing started
            db["preprocessing_jobs"].update_one(
                {"job_id": job_id},
                {
                    "$push": {"steps": {"name": "preprocessing_started", "completed": True, "timestamp": datetime.now()}},
                    "$set": {"status": "processing", "currentStep": "preprocessing"}
                }
            )
            
            # Run preprocessing
            preprocessing_results = preprocessor.preprocess(options)
            
            # Update status for preprocessing completion
            db["preprocessing_jobs"].update_one(
                {"job_id": job_id},
                {
                    "$push": {"steps": {"name": "preprocessing_completed", "completed": True, "timestamp": datetime.now()}},
                    "$set": {
                        "processedRecords": preprocessing_results.get("recordCount", total_records),
                        "status": "saving_results",
                        "currentStep": "saving_results"
                    }
                }
            )
            
            # Final status update
            db["preprocessing_jobs"].update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "status": "completed", 
                        "endTime": datetime.now(), 
                        "metadata": preprocessing_results
                    },
                    "$push": {"steps": {"name": "results_saved", "completed": True, "timestamp": datetime.now()}}
                }
            )
            
            # Return success response
            return jsonify({
                "message": "BMKG preprocessing completed successfully",
                "job_id": job_id,
                "collection": collection_name,
                "cleanedCollection": preprocessing_results.get("cleanedCollection"),
                "preprocessedCollection": f"{collection_name}_cleaned",
                "status": "preprocessed",
                "recordCount": preprocessing_results.get("recordCount", 0),
                "warnings": preprocessing_results.get("warnings", [])
            }), 200
            
        except BmkgPreprocessingError as pe:
            error_details = str(pe)
            logger.error(f"BmkgPreprocessingError: {error_details}")
            
            # Update status with error
            db["preprocessing_jobs"].update_one(
                {"job_id": job_id},
                {
                    "$set": {"status": "failed", "endTime": datetime.now(), "error": error_details},
                    "$push": {"steps": {"name": "processing_failed", "completed": False, "timestamp": datetime.now(), "error": error_details}}
                }
            )
            
            return jsonify({
                "message": f"Preprocessing error: {error_details}",
                "job_id": job_id,
                "status": "failed"
            }), 500
    
    except Exception as e:
        logger.error(f"Error in BMKG preprocessing: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"message": f"Server error: {str(e)}"}), 500

@preprocessing_bp.route("/preprocess/bmkg/<collection_name>/stream", methods=["GET"])
def preprocess_bmkg_stream(collection_name):
    """
    SSE endpoint for real-time BMKG preprocessing logs
    Stream preprocessing progress to frontend
    """
    def generate():
        # Create log queue for this session
        log_queue = Queue()
        session_id = f"{collection_name}_{int(time.time())}"
        
        # Setup custom log handler that captures logs
        class SSELogHandler(logging.Handler):
            def emit(self, record):
                try:
                    log_entry = self.format(record)
                    
                    # Parse progress information
                    if "PROGRESS:" in log_entry:
                        try:
                            # Extract PROGRESS:percentage:stage:message
                            progress_part = log_entry.split("PROGRESS:")[1]
                            parts = progress_part.split(":", 2)  # Split into max 3 parts
                            
                            if len(parts) >= 3:
                                # Parse percentage with error handling
                                try:
                                    percentage_str = parts[0].strip()
                                    percentage = int(percentage_str) if percentage_str.isdigit() else 0
                                except (ValueError, AttributeError):
                                    percentage = 0
                                    
                                log_queue.put({
                                    'type': 'progress',
                                    'percentage': percentage,
                                    'stage': parts[1].strip() if len(parts) > 1 else 'processing',
                                    'message': parts[2].strip() if len(parts) > 2 else 'Processing...'
                                })
                            else:
                                # If parsing fails, treat as regular log
                                log_queue.put({
                                    'type': 'log',
                                    'level': record.levelname,
                                    'message': log_entry,
                                    'timestamp': time.time()
                                })
                        except (ValueError, IndexError) as e:
                            # If parsing fails, treat as regular log
                            log_queue.put({
                                'type': 'log',
                                'level': record.levelname,
                                'message': log_entry,
                                'timestamp': time.time()
                            })
                    else:
                        log_queue.put({
                            'type': 'log',
                            'level': record.levelname,
                            'message': log_entry,
                            'timestamp': time.time()
                        })
                except Exception as e:
                    logger.error(f"SSE Log Handler error: {str(e)}")
        
        # Add handler to preprocessing logger
        sse_handler = SSELogHandler()
        sse_handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
        
        preprocessing_logger = logging.getLogger('preprocessing.bmkg.preprocessing_bmkg')
        preprocessing_logger.addHandler(sse_handler)
        preprocessing_logger.setLevel(logging.INFO)
        
        try:
            # Send initial connection message
            yield f"data: {json.dumps({'type': 'connected', 'session_id': session_id, 'collection': collection_name})}\n\n"
            
            # Get db instance from current request context
            db_instance = current_app.config['MONGO_DB']
            
            # Start preprocessing in background thread
            preprocessing_result = {'status': None, 'data': None, 'error': None}
            
            def run_preprocessing():
                result = None
                try:
                    # Send starting message
                    log_queue.put({
                        'type': 'progress',
                        'stage': 'starting',
                        'percentage': 0,
                        'message': 'Initializing BMKG preprocessing...'
                    })
                    
                    # Run actual preprocessing
                    preprocessor = BmkgPreprocessor(db_instance, collection_name)
                    result = preprocessor.preprocess()
                    
                    def sanitize_numpy(obj):
                        if isinstance(obj, dict):
                            return {key: sanitize_numpy(value) for key, value in obj.items()}
                        elif isinstance(obj, list):
                            return [sanitize_numpy(item) for item in obj]
                        elif hasattr(obj, 'item'):  # numpy scalar
                            return obj.item()
                        elif type(obj).__module__ == 'numpy':
                            if hasattr(obj, 'tolist'):
                                return obj.tolist()
                            return obj.item()
                        else:
                            return obj
                    
                    safe_result = sanitize_numpy(result)
                    
                    # Debug detailed logging result 
                    logger.info(f"Raw preprocessing result: {result}")
                    logger.info(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                    
                    # Store result
                    preprocessing_result['status'] = 'success'
                    preprocessing_result['data'] = safe_result
                    
                    # Send completion
                    safe_result = {
                        'recordCount': int(result.get('recordCount', 0)) if result.get('recordCount') else 0,
                        'originalRecordCount': int(result.get('originalRecordCount', 0)) if result.get('originalRecordCount') else 0,
                        'cleanedCollection': result.get('cleanedCollection'),
                        'collection': result.get('collection'),
                        'message': result.get('message'),
                        'preprocessing_report': result.get('preprocessing_report')
                    }
                    
                    logger.info(f"📤 Sending safe result: {safe_result}")
                    
                    log_queue.put({
                        'type': 'complete',
                        'status': 'success',
                        'result': safe_result
                    })
                    
                except Exception as e:
                    logger.error(f"❌ Preprocessing error: {str(e)}")
                    logger.error(f"❌ Traceback: {traceback.format_exc()}")
                    
                    preprocessing_result['status'] = 'error'
                    preprocessing_result['error'] = str(e)
                    
                    log_queue.put({
                        'type': 'error',
                        'message': str(e),
                        'traceback': traceback.format_exc()
                    })
                finally:
                    # Signal completion
                    logger.info("🏁 Preprocessing thread completed, stream will close naturally")
                    preprocessing_result['thread_complete'] = True
            
            # Start preprocessing thread
            thread = threading.Thread(target=run_preprocessing)
            thread.daemon = True
            thread.start()
            
            # Stream logs to client
            while True:
                try:
                    # Get log from queue (timeout to check if client disconnected)
                    log_data = log_queue.get(timeout=1)
                    
                    # ✅ FIXED: Send log to client FIRST before checking completion
                    yield f"data: {json.dumps(log_data)}\n\n"
                    
                    # Check if done AFTER sending the message
                    if log_data.get('type') == 'complete':
                        yield f": completion-sent\n\n"
                        time.sleep(1.0)  # 1 second delay
                        logger.info("🎯 Closing stream after completion processed")
                        break
                    
                except Exception:
                    if preprocessing_result.get('thread_complete'):
                        logger.info("🏁 Thread completed, closing stream")
                        break
                    # Timeout - send keepalive
                    yield ": keepalive\n\n"
                    continue
                    
        except GeneratorExit:
            # Client disconnected
            logger.info(f"Client disconnected from preprocessing stream: {collection_name}")
        finally:
            # Cleanup
            try:
                preprocessing_logger.removeHandler(sse_handler)
                logger.info(f"Cleaned up SSE handler for session {session_id}")
            except Exception as e:
                logger.error(f"Error during SSE cleanup: {str(e)}")
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
        }
    )


@app.route('/api/v1/convert/xlsx-to-csv', methods=['POST'])
def convert_xlsx_to_csv():
    """Convert single XLSX file to CSV format"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.xlsx'):
            return jsonify({'error': 'Only XLSX files are supported'}), 400
        
        # Check file size (16MB limit)
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > 16 * 1024 * 1024:  # 16MB
            return jsonify({'error': 'File size exceeds 16MB limit'}), 400
        
        # Read file content
        file_buffer = file.read()
        
        # Convert XLSX to CSV
        result = convert_single_xlsx(file_buffer, file.filename)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in XLSX conversion: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/convert/xlsx-merge-csv', methods=['POST'])
def convert_multi_xlsx_to_csv():
    """Convert and merge multiple XLSX files to single CSV"""
    try:
        files = request.files.getlist('files')
        
        if not files:
            return jsonify({'error': 'No files provided'}), 400
        
        if len(files) > 50:
            return jsonify({'error': 'Maximum 50 files allowed per batch'}), 400
        
        # Validate and prepare file data
        files_data = []
        total_size = 0
        
        for file in files:
            if file.filename == '':
                continue
                
            if not file.filename.lower().endswith('.xlsx'):
                return jsonify({'error': f'File {file.filename} is not XLSX format'}), 400
            
            # Check individual file size
            file.seek(0, 2)
            file_size = file.tell()
            file.seek(0)
            
            if file_size > 16 * 1024 * 1024:  # 16MB per file
                return jsonify({'error': f'File {file.filename} exceeds 16MB limit'}), 400
            
            total_size += file_size
            
            # Check total batch size (200MB limit for batch)
            if total_size > 200 * 1024 * 1024:
                return jsonify({'error': 'Total batch size exceeds 200MB limit'}), 400
            
            file_buffer = file.read()
            files_data.append({
                'buffer': file_buffer,
                'filename': file.filename
            })
        
        if not files_data:
            return jsonify({'error': 'No valid XLSX files found'}), 400
        
        # Process and merge files
        result = merge_multiple_xlsx(files_data)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in multi-XLSX conversion: {str(e)}")
        return jsonify({'error': str(e)}), 500
    

# @preprocessing_bp.route("/dataset-meta/<slug>/decomposition", methods=["GET"])
# def get_decomposition_data(slug):
#     """
#     Get seasonal decomposition data for a specific parameter
    
#     Query Parameters:
#         - parameter: Parameter name (e.g., T2M, RH2M, ALLSKY_SFC_SW_DWN)
    
#     Returns:
#         JSON with decomposition time series (original, trend, seasonal, residual)
#     """
#     try:
#         db = current_app.config['MONGO_DB']
        
#         # Get parameter from query string
#         parameter = request.args.get('parameter')
#         if not parameter:
#             return jsonify({"message": "Parameter name is required"}), 400
        
#         # Find dataset metadata to get decomposition collection name
#         meta_collection = "dataset_meta" if "dataset_meta" in db.list_collection_names() else "DatasetMeta"
        
#         dataset_meta = db[meta_collection].find_one({"collectionName": slug})
        
#         if not dataset_meta:
#             return jsonify({"message": f"Dataset '{slug}' not found"}), 404
        
#         # Check if dataset is preprocessed
#         if dataset_meta.get("status") != "preprocessed":
#             return jsonify({
#                 "message": "Dataset must be preprocessed before viewing decomposition",
#                 "current_status": dataset_meta.get("status", "unknown")
#             }), 400
        
#         # Get decomposition collection name
#         decomp_collection_name = f"{slug}_decomposition"
        
#         if decomp_collection_name not in db.list_collection_names():
#             return jsonify({
#                 "message": "Decomposition data not found. Dataset may need reprocessing.",
#                 "expected_collection": decomp_collection_name
#             }), 404
        
#         # Query decomposition data for the specific parameter
#         decomp_data = list(db[decomp_collection_name].find(
#             {"parameter": parameter},
#             {"_id": 0}  # Exclude MongoDB _id
#         ).sort("Date", 1))  # Sort by date ascending
        
#         if not decomp_data:
#             return jsonify({
#                 "message": f"No decomposition data found for parameter '{parameter}'",
#                 "available_parameters": db[decomp_collection_name].distinct("parameter")
#             }), 404
        
#         # Extract decomposition components
#         dates = [record["Date"].isoformat() if isinstance(record["Date"], datetime) else record["Date"] for record in decomp_data]
#         original = [float(record["original"]) if record["original"] is not None else None for record in decomp_data]
#         trend = [float(record["trend"]) if record["trend"] is not None else None for record in decomp_data]
#         seasonal = [float(record["seasonal"]) if record["seasonal"] is not None else None for record in decomp_data]
#         residual = [float(record["residual"]) if record["residual"] is not None else None for record in decomp_data]
        
#         # Prepare response
#         response = {
#             "collectionName": slug,
#             "parameter": parameter,
#             "decomposition": {
#                 "dates": dates,
#                 "original": original,
#                 "trend": trend,
#                 "seasonal": seasonal,
#                 "residual": residual
#             },
#             "metadata": {
#                 "model": "additive",
#                 "period": 365,
#                 "dataPoints": len(decomp_data),
#                 "dateRange": {
#                     "start": dates[0] if dates else None,
#                     "end": dates[-1] if dates else None
#                 }
#             }
#         }
        
#         return jsonify(response), 200
        
#     except Exception as e:
#         logger.error(f"Error fetching decomposition data: {str(e)}")
#         logger.error(traceback.format_exc())
#         return jsonify({"message": f"Server error: {str(e)}"}), 500

@preprocessing_bp.route("/dataset-meta/<slug>/decomposition", methods=["GET"])
def get_decomposition_data(slug):
    """
    Get seasonal decomposition data for both NASA and BMKG datasets
    
    Query Parameters:
        - parameter: Parameter name (e.g., T2M, RH2M for NASA; RR, TX, TN for BMKG)
    
    Returns:
        JSON with decomposition time series (original, trend, seasonal, residual)
        
    Supports:
        - NASA: Separate decomposition collection with full time series
        - BMKG: Embedded decomposition in preprocessing_report with sample data
    """
    try:
        db = current_app.config['MONGO_DB']
        
        # Get parameter from query string
        parameter = request.args.get('parameter')
        if not parameter:
            return jsonify({"message": "Parameter name is required"}), 400
        
        # Find dataset metadata to determine dataset type
        meta_collection = "dataset_meta" if "dataset_meta" in db.list_collection_names() else "DatasetMeta"
        dataset_meta = db[meta_collection].find_one({"collectionName": slug})
        
        if not dataset_meta:
            return jsonify({"message": f"Dataset '{slug}' not found"}), 404
        
        # Check if dataset is preprocessed
        if dataset_meta.get("status") != "preprocessed":
            return jsonify({
                "message": "Dataset must be preprocessed before viewing decomposition",
                "current_status": dataset_meta.get("status", "unknown")
            }), 400
        
        # 🔧 NEW: Determine dataset type from metadata or collection name patterns
        dataset_type = dataset_meta.get("dataType", "").lower()
        
        # Auto-detect if not specified in metadata
        if not dataset_type:
            if any(param in ['RR', 'TX', 'TN', 'TAVG', 'RH_AVG', 'FF_X', 'FF_AVG', 'DDD_X', 'SS'] for param in [parameter]):
                dataset_type = "bmkg"
            elif any(param in ['T2M', 'RH2M', 'ALLSKY_SFC_SW_DWN', 'WS10M', 'PRECTOTCORR'] for param in [parameter]):
                dataset_type = "nasa"
            else:
                # Try both approaches
                dataset_type = "auto-detect"
        
        # 🎯 APPROACH 1: Try NASA-style separate collection first
        decomp_collection_name = f"{slug}_decomposition"
        
        if decomp_collection_name in db.list_collection_names():
            logger.info(f"Found NASA-style decomposition collection: {decomp_collection_name}")
            
            # Query decomposition data for the specific parameter
            decomp_data = list(db[decomp_collection_name].find(
                {"parameter": parameter},
                {"_id": 0}  # Exclude MongoDB _id
            ).sort("Date", 1))  # Sort by date ascending
            
            if decomp_data:
                # Extract decomposition components (NASA format)
                dates = [record["Date"].isoformat() if isinstance(record["Date"], datetime) else record["Date"] for record in decomp_data]
                original = [float(record["original"]) if record["original"] is not None else None for record in decomp_data]
                trend = [float(record["trend"]) if record["trend"] is not None else None for record in decomp_data]
                seasonal = [float(record["seasonal"]) if record["seasonal"] is not None else None for record in decomp_data]
                residual = [float(record["residual"]) if record["residual"] is not None else None for record in decomp_data]
                
                # Prepare NASA-style response
                response = {
                    "collectionName": slug,
                    "parameter": parameter,
                    "dataset_type": "nasa",
                    "data_source": "separate_collection",
                    "decomposition": {
                        "dates": dates,
                        "original": original,
                        "trend": trend,
                        "seasonal": seasonal,
                        "residual": residual
                    },
                    "metadata": {
                        "model": "additive",
                        "period": 365,
                        "dataPoints": len(decomp_data),
                        "dateRange": {
                            "start": dates[0] if dates else None,
                            "end": dates[-1] if dates else None
                        }
                    }
                }
                
                return jsonify(response), 200
        
        # 🎯 APPROACH 2: Try BMKG-style embedded in preprocessing_report
        report = db["preprocessing_report"].find_one({
            "$or": [
                {"collection_name": slug},
                {"original_collection_name": slug}
            ]
        })
        
        if report:
            logger.info(f"Found preprocessing report for: {slug}")
            
            # Extract decomposition data from embedded report
            decomposition_data = report.get("preprocessing_details", {}).get("decomposition", {})
            
            if not decomposition_data:
                # Try alternative path
                decomposition_data = report.get("report_data", {}).get("decomposition", {})
            
            if decomposition_data:
                # Check if parameter exists in decomposition data
                param_decomp = decomposition_data.get("decomposition_data", {}).get(parameter)
                
                if param_decomp:
                    # Extract components from embedded sample data (BMKG format)
                    components_sample = param_decomp.get("components_sample", {})
                    statistics = param_decomp.get("statistics", {})
                    seasonal_analysis = param_decomp.get("seasonal_analysis", {})
                    
                    # Convert sample data to time series format
                    trend_data = components_sample.get("trend", {})
                    seasonal_data = components_sample.get("seasonal", {})
                    residual_data = components_sample.get("residual", {})
                    
                    # Convert timestamps and values
                    if trend_data:
                        dates = sorted(trend_data.keys())
                        trend_values = [trend_data.get(date) for date in dates]
                        seasonal_values = [seasonal_data.get(date) for date in dates]
                        residual_values = [residual_data.get(date) for date in dates]
                        
                        # Reconstruct original from components (trend + seasonal + residual)
                        original_values = []
                        for i in range(len(trend_values)):
                            if trend_values[i] is not None and seasonal_values[i] is not None and residual_values[i] is not None:
                                original_values.append(trend_values[i] + seasonal_values[i] + residual_values[i])
                            else:
                                original_values.append(None)
                    else:
                        dates = []
                        trend_values = []
                        seasonal_values = []
                        residual_values = []
                        original_values = []
                    
                    # Prepare BMKG-style response
                    response = {
                        "collectionName": slug,
                        "parameter": parameter,
                        "dataset_type": "bmkg",
                        "data_source": "embedded_report",
                        "decomposition": {
                            "dates": dates,
                            "original": original_values,
                            "trend": trend_values,
                            "seasonal": seasonal_values,
                            "residual": residual_values
                        },
                        "statistics": statistics,
                        "indonesian_seasonal_analysis": seasonal_analysis,
                        "metadata": {
                            "model": param_decomp.get("method", "STL_robust"),
                            "period": param_decomp.get("period", 365),
                            "robust": True,
                            "total_data_points": statistics.get("data_points", 0),
                            "trend_strength": statistics.get("trend_strength", 0),
                            "seasonal_strength": statistics.get("seasonal_strength", 0),
                            "trend_direction": statistics.get("trend_direction", "unknown"),
                            "dataPoints": len(dates),
                            "dateRange": {
                                "start": dates[0] if dates else None,
                                "end": dates[-1] if dates else None
                            },
                            "note": "Sample data (last 90 days) from embedded STL decomposition"
                        }
                    }
                    
                    return jsonify(response), 200
                else:
                    # Parameter not found in decomposition data
                    available_params = list(decomposition_data.get("decomposition_data", {}).keys())
                    return jsonify({
                        "message": f"No decomposition data found for parameter '{parameter}' in embedded report",
                        "available_parameters": available_params,
                        "dataset_type": "bmkg"
                    }), 404
        
        # 🚫 NO DATA FOUND - Provide helpful error message
        return jsonify({
            "message": "Decomposition data not found",
            "details": {
                "checked_nasa_collection": decomp_collection_name,
                "checked_embedded_report": "preprocessing_report",
                "dataset_type_detected": dataset_type,
                "parameter_requested": parameter
            },
            "suggestions": [
                "Dataset may need reprocessing with decomposition enabled",
                f"For NASA datasets, expected collection: {decomp_collection_name}",
                "For BMKG datasets, decomposition should be embedded in preprocessing_report",
                "Verify parameter name matches dataset type (NASA: T2M, RH2M; BMKG: RR, TX, TN)"
            ]
        }), 404
        
    except Exception as e:
        logger.error(f"Error fetching decomposition data: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"message": f"Server error: {str(e)}"}), 500

app.register_blueprint(preprocessing_bp)
app.register_blueprint(scheduler_bp)

@app.route("/api/v1/scheduler/health", methods=["GET"])
def scheduler_health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5001))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
    # # Jangan pakai debug di production
    # app.run(host="0.0.0.0", port=5001)