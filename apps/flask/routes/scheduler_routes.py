import os
import sys
import subprocess
import logging
import threading
from datetime import datetime, timezone, timedelta
from flask import Blueprint, jsonify, request
from models.scheduler_logs import SchedulerLog # dynamic import map
from models.scheduler_logs import SchedulerLog
from middleware.auth_middleware import require_auth
from pymongo import MongoClient
from bson import ObjectId
import time

logger = logging.getLogger(__name__)

# Registration Example for app.py:
# from routes.scheduler_routes import scheduler_bp
# app.register_blueprint(scheduler_bp)

scheduler_bp = Blueprint('scheduler', __name__, url_prefix="/api/v1/scheduler")

# Simple in-memory rate limiting tracker for manual triggers
_trigger_history = []

def get_next_run() -> str:
    """Calculate next run from cron expression (daily 02:00 WIB)."""
    # 02:00 WIB is 19:00 UTC previous day.
    now = datetime.now(timezone.utc)
    # Next WIB 02:00:
    # Convert now to WIB (+7)
    now_wib = now + timedelta(hours=7)
    
    # Tambahkan tzinfo agar object ini menjadi offset-aware
    next_wib_run = datetime(now_wib.year, now_wib.month, now_wib.day, 2, 0, 0, tzinfo=timezone.utc)
    
    if now_wib >= next_wib_run:
        next_wib_run += timedelta(days=1)
        
    next_utc_run = next_wib_run - timedelta(hours=7)
    return next_utc_run.replace(tzinfo=timezone.utc).isoformat()


def validate_api_key(req) -> bool:
    """Validate API key via Header Authorization."""
    auth_header = req.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return False
    
    token = auth_header.split(" ")[1]
    expected_token = os.getenv("API_KEY", "dev_secret_key")
    return token == expected_token

def run_scheduler_async(log_id: str, mode: str, tasks: list, datasets: dict):
    """
    Execute daily_scheduler.py using subprocess to ensure it runs independently.
    """
    try:
        logger.info(f"Running scheduler async for log: {log_id}. Mode: {mode}")
        
        script_path = os.path.join(os.path.dirname(__file__), '..', 'scheduler', 'daily_scheduler.py')
        cmd = [sys.executable, script_path, "--manual", "--log-id", str(log_id)]
        
        if mode == "custom":
            if datasets.get("nasa_refresh"):
                cmd.extend(["--nasa-refresh", ",".join(datasets["nasa_refresh"])])
            if datasets.get("nasa_preprocess"):
                cmd.extend(["--nasa-preprocess", ",".join(datasets["nasa_preprocess"])])
            if datasets.get("bmkg_preprocess"):
                cmd.extend(["--bmkg-preprocess", ",".join(datasets["bmkg_preprocess"])])
        else:
            if tasks and "all" not in tasks:
                # Assuming quick run single task
                cmd.extend(["--tasks", tasks[0]])
            else:
                cmd.extend(["--tasks", "all"])

        # Execute detached process
        subprocess.Popen(cmd)
        
    except Exception as e:
        logger.error(f"Async scheduler execution failed: {e}")
        db = SchedulerLog._get_db()
        db.scheduler_logs.update_one(
            {"_id": ObjectId(log_id)},
            {"$set": {"status": "failed", "completedAt": datetime.now(timezone.utc)}}
        )
@scheduler_bp.route("/status", methods=["GET"])
def get_status():
    """Get Scheduler Status"""
    try:
        latest_log = SchedulerLog.get_latest()
        
        # Calculate statistics
        db = SchedulerLog._get_db()
        collection = db["scheduler_logs"]
        total_execs = collection.count_documents({})
        failed_execs = collection.count_documents({"status": "failed"})
        
        success_rate = SchedulerLog.get_success_rate(days=30)
        
        # Calculate Average Duration
        durations = list(collection.aggregate([
            {"$match": {"status": "success", "duration": {"$ne": None}}},
            {"$group": {"_id": None, "avgDuration": {"$avg": "$duration"}}}
        ]))
        avg_duration = durations[0]["avgDuration"] if durations else 0

        # === PERBAIKAN: BACA DARI DATABASE CONFIG, BUKAN DARI .ENV ===
        config = db.scheduler_config.find_one({})
        
        if config and config.get("enabled") is True:
            is_active = True
            computed_runs = calculate_next_runs(
                config.get("frequency", "weekly"),
                config.get("executionTime", "02:00"),
                config.get("dayOfWeek", 0),
                config.get("daysOfWeek"),
                count=1
            )
            next_run = computed_runs[0] if computed_runs else None
        else:
            is_active = False
            next_run = None

        data = {
            "lastRun": latest_log,
            "nextRun": next_run,
            "isActive": is_active,
            "statistics": {
                "successRate": success_rate,
                "avgDuration": round(avg_duration, 2),
                "totalExecutions": total_execs,
                "failedExecutions": failed_execs
            }
        }
        
        return jsonify({"success": True, "data": data}), 200

    except Exception as e:
        logger.error(f"Error fetching scheduler status: {e}")
        return jsonify({
            "success": False, 
            "error": {"code": "INTERNAL_ERROR", "message": "Failed to retrieve status."}
        }), 500

@scheduler_bp.route("/logs", methods=["GET"])
def get_logs():
    """Get Scheduler Logs with Pagination"""
    try:
        limit = int(request.args.get("limit", 10))
        offset = int(request.args.get("offset", 0))
        status = request.args.get("status")
        
        # Cap limit
        limit = min(limit, 50)
        
        logs, total = SchedulerLog.get_logs(limit=limit, offset=offset, status_filter=status)
        
        has_more = (offset + limit) < total
        
        return jsonify({
            "success": True,
            "data": {
                "logs": logs,
                "pagination": {
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "hasMore": has_more,
                    "nextOffset": offset + limit if has_more else None
                }
            }
        }), 200

    except ValueError as e:
        return jsonify({"success": False, "error": {"code": "VALIDATION_ERROR", "message": str(e)}}), 400
    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        return jsonify({"success": False, "error": {"code": "INTERNAL_ERROR"}}), 500
    

def calculate_next_runs(
    frequency,
    exec_time, 
    day_of_week=0,
    days_of_week=None,
    count=6
):
    """Simple calculation for next runs UTC based on UTC+7 """
    try:
        hour, minute = map(int, exec_time.split(':'))
    except: 
        hour, minute = 2, 0
    
    now = datetime.now(timezone.utc)
    runs = []
    
    current_wib = now + timedelta(hours=7)
    base_date = current_wib.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if current_wib > base_date:
        base_date += timedelta(days=1)
        
    date_cursor = base_date
    while len(runs) < count:
        # Konversi format hari Python ke format JavaScript (Minggu=0, Senin=1, dst)
        js_weekday = (date_cursor.weekday() + 1) % 7
        if frequency == "weekly" and js_weekday != day_of_week:
            date_cursor += timedelta(days=1)
            continue
        elif frequency == "biweekly" and days_of_week and js_weekday not in days_of_week:
            date_cursor += timedelta(days=1)
            continue
        utc_run = date_cursor - timedelta(hours=7)
        runs.append(utc_run.isoformat())
        date_cursor += timedelta(days=1)
        
    return runs
    
    
@scheduler_bp.route("/config", methods=["GET"])
# @require_auth, matikan sementara untuk demo
def get_automation_config():
    try:
        db = SchedulerLog._get_db()
        config = db.scheduler_config.find_one({})
        
        if not config:
            config = {
                "enabled": False,
                "frequency": "weekly",
                "executionTime": "02:00",
                "dayOfWeek": 0,
                "daysOfWeek": [0, 3],
                "selectedDatasets": {"nasa_refresh": [], "nasa_preprocess": [], "bmkg_preprocess": []}
            }
        else:
            config["_id"] = str(config["_id"])
            
        config["nextRuns"] = calculate_next_runs(
            config.get("frequency", "weekly"),
            config.get("executionTime", "02:00"),
            config.get("dayOfWeek", 0),
            config.get("daysOfWeek")
        ) if config.get("enabled", False) else []
            
        return jsonify({"success": True, "data": config})
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        return jsonify({"success": False, "error": {"code": "INTERNAL_ERROR"}}), 500


@scheduler_bp.route("/config", methods=["POST"])
# @require_auth, matikan sementara untuk demo
def save_automation_config():
    try:
        body = request.get_json() or {}
        db = SchedulerLog._get_db()
        
        # Validasi dasar
        if body.get("frequency") not in ["weekly", "biweekly"]:
            return jsonify({"success": False, "error": {"message": "Invalid frequency"}}), 400
            
        update_data = {
            "enabled": body.get("enabled", False),
            "frequency": body.get("frequency", "weekly"),
            "executionTime": body.get("executionTime", "02:00"),
            "dayOfWeek": body.get("dayOfWeek", 0),
            "daysOfWeek": body.get("daysOfWeek", [0, 3]),
            "selectedDatasets": body.get("selectedDatasets", {"nasa_refresh": [], "nasa_preprocess": [], "bmkg_preprocess": []}),
            "updatedAt": datetime.now(timezone.utc)
        }
        
        db.scheduler_config.update_one({}, {"$set": update_data, "$unset": {"lastTriggeredDate": "", "daysOfMonth": ""}}, upsert=True)
        
        next_run = calculate_next_runs(
            update_data["frequency"], update_data["executionTime"], 
            update_data["dayOfWeek"], update_data["daysOfWeek"], count=1
        )
        
        return jsonify({"success": True, "message": "Config saved", "data": {"nextRun": next_run[0] if next_run else None}})
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return jsonify({"success": False, "error": {"code": "INTERNAL_ERROR"}}), 500

@scheduler_bp.route("/trigger", methods=["POST"])
# @require_auth, matikan sementara untuk demo
def trigger_scheduler():
    """Trigger Scheduler Manually"""
    global _trigger_history
    
    # Simple rate limiting: max 5 hits per hour
    now = datetime.now()
    _trigger_history = [t for t in _trigger_history if now - t < timedelta(hours=1)]
    if len(_trigger_history) >= 5:
        return jsonify({"success": False, "error": {"code": "RATE_LIMIT_EXCEEDED", "message": "Max 5 manual triggers per hour."}}), 429
        
    # Check if already running
    latest = SchedulerLog.get_latest()
    if latest and latest.get("status") == "running":
         return jsonify({"success": False, "error": {"code": "CONFLICT", "message": "Scheduler is already running."}}), 409
         
    _trigger_history.append(now)
    
    try:
        body = request.get_json(silent=True) or {}
        mode = body.get("mode", "quick")
        tasks = body.get("tasks", [])
        datasets = body.get("datasets", {})
        is_async = body.get("async", True)
        
        # Biarkan manual script yang create log. Kita asumsikan pending log di sini.
        log_id = SchedulerLog.create_log(triggered_by="manual")
        
        if is_async:
            _thread = threading.Thread(target=run_scheduler_async, args=(log_id, mode, tasks, datasets))
            _thread.start()
            
            return jsonify({
                "success": True,
                "data": {
                    "executionId": log_id,
                    "status": "queued",
                    "startedAt": datetime.now(timezone.utc).isoformat(),
                    "estimatedDuration": 360 # Menit estimasi kasar jika full run
                }
            }), 202
        else:
            # Sync execution    
            run_scheduler_async(log_id, tasks)
            # Fetch completed logic
            completed_log = SchedulerLog.get_by_id(log_id)
            return jsonify({
                 "success": True,
                 "data": completed_log
            }), 200
            
    except Exception as e:
        logger.error(f"Trigger error: {e}")
        return jsonify({"success": False, "error": {"code": "INTERNAL_ERROR"}}), 500
    
def scheduler_daemon():
    """Background thread that checks the schedule every minute."""
    # Baris with dihapus karena Blueprint tidak mendukung context manager 
    try:
        client = MongoClient(os.getenv("MONGODB_URI"))
        db = client[os.getenv("MONGODB_DB_NAME", "tugas_akhir")]
    except Exception as e:
        logger.error(f"Daemon failed to connect to DB: {e}")
        return

    while True:
        try:
            now = datetime.now(timezone.utc)
            now_wib = now + timedelta(hours=7)
            
            config = db.scheduler_config.find_one({})
            
            if config and config.get("enabled"):
                exec_time = config.get("executionTime", "02:00")
                try:
                    t_hour, t_minute = map(int, exec_time.split(':'))
                except ValueError:
                    t_hour, t_minute = 2, 0
                    
                # Check frequency matching
                freq = config.get("frequency", "weekly")
                is_today = False
                
                # Konversi hari agar sama dengan format database/frontend
                js_weekday = (now_wib.weekday() + 1) % 7
                
                if freq == "weekly" and js_weekday == config.get("dayOfWeek", 0):
                    is_today = True
                elif freq == "biweekly" and js_weekday in config.get("daysOfWeek", []):
                    is_today = True

                # Trigger if day and exact minute matches
                if is_today and now_wib.hour == t_hour and now_wib.minute == t_minute:
                    today_str = now_wib.strftime("%Y-%m-%d")
                    
                    # Prevent duplicate runs in the same minute/day
                    if config.get("lastTriggeredDate") != today_str:
                        logger.info(f"Cron Daemon: Time matches ({exec_time} WIB)! Executing automation...")
                        
                        # Lock execution for today
                        db.scheduler_config.update_one({}, {"$set": {"lastTriggeredDate": today_str}})
                        
                        # Prepare and Run
                        datasets = config.get("selectedDatasets", {})
                        
                        # Bypass Flask models & create log directly in MongoDB
                        log_doc = {
                            "status": "running",
                            "triggeredBy": "cron",
                            "executedAt": datetime.now(timezone.utc),
                            "createdAt": datetime.now(timezone.utc),
                            "updatedAt": datetime.now(timezone.utc),
                            "tasks": [],
                            "totalDatasets": sum(len(v) for v in datasets.values() if isinstance(v, list)),
                            "datasetsUpdated": 0,
                            "errors": []
                        }
                        
                        res = db.scheduler_logs.insert_one(log_doc)
                        log_id = str(res.inserted_id)
                        
                        run_scheduler_async(log_id, "custom", [], datasets)

        except Exception as e:
            logger.error(f"Daemon encountered an error: {e}")
            
        # Wait 60 seconds before checking again
        time.sleep(60)
            
# Start the daemon when app initialize
daemon_thread = threading.Thread(target=scheduler_daemon, daemon=True)
daemon_thread.start()