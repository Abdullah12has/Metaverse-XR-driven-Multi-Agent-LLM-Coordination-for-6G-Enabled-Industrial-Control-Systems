from flask import Flask, request, jsonify
import logging
from logging.handlers import RotatingFileHandler
import threading
import uuid
import time
from main import run
from functools import wraps
from datetime import datetime

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
handler = RotatingFileHandler('app.log', maxBytes=10000000, backupCount=5)
handler.setFormatter(logging.Formatter(
    '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
))
app.logger.addHandler(handler)

# Dictionary to store job results with cleanup
class JobStore:
    def __init__(self):
        self.jobs = {}
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
            to_delete = []
            for job_id, job_data in self.jobs.items():
                created_at = datetime.fromisoformat(job_data['created_at'])
                age = (current_time - created_at).total_seconds() / 3600
                if age > max_age_hours:
                    to_delete.append(job_id)
            
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
    """Runs the multi-agent process asynchronously and stores the result."""
    try:
        app.logger.info(f"Starting job {job_id} with data: {input_data}")
        
        # Record start time
        start_time = time.time()
        
        # Call the multi-agent function with dynamic inputs
        result = run(input_data)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        job_store.update_job(job_id, {
            "status": "completed",
            "result": result,
            "execution_time": f"{execution_time:.2f} seconds"
        })
        
    except Exception as e:
        app.logger.error(f"Error in job {job_id}: {str(e)}", exc_info=True)
        job_store.update_job(job_id, {
            "status": "failed",
            "error": str(e)
        })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route('/run', methods=['POST'])
@validate_json()
def run_multiagent():
    try:
        data = request.json
        
        # Validate required fields
        if not data:
            return jsonify({"error": "Request body cannot be empty"}), 400
        
        # Generate job ID and store initial status
        job_id = str(uuid.uuid4())
        job_store.add_job(job_id)
        
        # Start the process in a separate thread
        thread = threading.Thread(
            target=run_multiagent_async,
            args=(job_id, data),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            "message": "Multi-agent system started",
            "job_id": job_id
        }), 202
        
    except Exception as e:
        app.logger.error(f"Error occurred: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Check the status of a job."""
    job_data = job_store.get_job(job_id)
    if job_data:
        return jsonify(job_data)
    return jsonify({"error": "Job ID not found"}), 404

@app.route('/jobs', methods=['GET'])
def list_jobs():
    """List all active jobs."""
    with job_store.lock:
        return jsonify({
            "total_jobs": len(job_store.jobs),
            "jobs": job_store.jobs
        })

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Server Error: {str(error)}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500

# Cleanup task
def cleanup_task():
    while True:
        job_store.cleanup_old_jobs()
        time.sleep(3600)  # Run every hour

if __name__ == '__main__':
    # Start cleanup thread
    cleanup_thread = threading.Thread(
        target=cleanup_task,
        daemon=True
    )
    cleanup_thread.start()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)