from flask import Flask, request, jsonify
import logging
from logging.handlers import RotatingFileHandler
import threading
import uuid
import time
from main import run
from functools import wraps
from datetime import datetime
import os
from influxdb_client import InfluxDBClient
from dotenv import load_dotenv
from collections import defaultdict

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
handler = RotatingFileHandler('app.log', maxBytes=10000000, backupCount=5)
handler.setFormatter(logging.Formatter(
    '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
))
app.logger.addHandler(handler)

# InfluxDB configuration
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://influxdb.monitoring.svc.cluster.local:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "my-secret-token")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "my-org")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "my-bucket")

def get_influx_client():
    return InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)

# Thread-safe JobStore
class JobStore:
    def __init__(self):
        self.jobs = defaultdict(dict)
        self.lock = threading.Lock()
        
    def add_job(self, job_id, status="processing"):
        with self.lock:
            self.jobs[job_id] = {
                "status": status,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
    
    def update_job(self, job_id, data):
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(data)
                self.jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()
    
    def get_job(self, job_id):
        with self.lock:
            return self.jobs.get(job_id)
    
    def cleanup_old_jobs(self, max_age_hours=24):
        with self.lock:
            current_time = datetime.utcnow()
            to_delete = [job_id for job_id, job_data in self.jobs.items()
                         if (current_time - datetime.fromisoformat(job_data['created_at'])).total_seconds() / 3600 > max_age_hours]
            for job_id in to_delete:
                del self.jobs[job_id]

job_store = JobStore()

# Middleware for request validation
def validate_json():
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return jsonify({"error": "Content-Type must be application/json"}), 415
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def run_multiagent_async(job_id, input_data):
    try:
        app.logger.info(f"Starting job {job_id} with data: {input_data}")
        start_time = time.time()
        result = run(input_data)
        execution_time = time.time() - start_time
        job_store.update_job(job_id, {
            "status": "completed",
            "result": result,
            "execution_time": f"{execution_time:.2f} seconds"
        })
    except Exception as e:
        app.logger.error(f"Error in job {job_id}: {str(e)}", exc_info=True)
        job_store.update_job(job_id, {"status": "failed", "error": str(e)})

def get_influx_data():
    with get_influx_client() as client:
        query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: -1h)
          |> limit(n: 10)
        '''
        query_api = client.query_api()
        tables = query_api.query(query)
        return [{
            "time": record.get_time().isoformat(),
            "measurement": record.get_measurement(),
            "field": record.get_field(),
            "value": record.get_value()
        } for table in tables for record in table.records]

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.utcnow().isoformat()})

@app.route('/run', methods=['POST'])
@validate_json()
def run_multiagent():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Request body cannot be empty"}), 400
        job_id = str(uuid.uuid4())
        job_store.add_job(job_id)
        threading.Thread(target=run_multiagent_async, args=(job_id, data), daemon=True).start()
        return jsonify({"message": "Multi-agent system started", "job_id": job_id}), 202
    except Exception as e:
        app.logger.error(f"Error occurred: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    job_data = job_store.get_job(job_id)
    return jsonify(job_data) if job_data else jsonify({"error": "Job ID not found"}), 404

@app.route('/jobs', methods=['GET'])
def list_jobs():
    with job_store.lock:
        return jsonify({"total_jobs": len(job_store.jobs), "jobs": job_store.jobs})

@app.route('/influx-data', methods=['GET'])
def influx_data():
    try:
        data = get_influx_data()
        print("InfluxDB Data:", data)
        return jsonify(data)
    except Exception as e:
        app.logger.error(f"Error fetching InfluxDB data: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Server Error: {str(error)}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500

def cleanup_task():
    while True:
        try:
            job_store.cleanup_old_jobs()
        except Exception as e:
            app.logger.error(f"Error in cleanup task: {str(e)}")
        time.sleep(3600)

if __name__ == '__main__':
    threading.Thread(target=cleanup_task, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
