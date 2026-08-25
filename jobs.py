import uuid
import threading
import traceback
import time


class Job:
    def __init__(self, job_id, kind):
        self.id = job_id
        self.kind = kind
        self.status = "queued"  # queued | running | done | error
        self.progress = 0.0
        self.message = ""
        self.result = None
        self.error = None
        self.created_at = time.time()
        self.log_lines = []

    def to_dict(self):
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "log": self.log_lines[-50:],
        }


JOBS = {}
JOBS_LOCK = threading.Lock()


def create_job(kind):
    job_id = str(uuid.uuid4())[:8]
    job = Job(job_id, kind)
    with JOBS_LOCK:
        JOBS[job_id] = job
    return job


def get_job(job_id):
    with JOBS_LOCK:
        return JOBS.get(job_id)


def run_in_background(job, fn):
    def _runner():
        job.status = "running"
        try:
            result = fn(job)
            job.result = result
            job.status = "done"
            job.progress = 1.0
        except Exception as e:
            job.status = "error"
            job.error = f"{e}\n{traceback.format_exc()}"

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return t
