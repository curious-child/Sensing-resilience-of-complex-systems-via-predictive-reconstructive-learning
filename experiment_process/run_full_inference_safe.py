"""Serial full evaluation with the original batch size and an external RAM watchdog."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "experiment_results" / "inference_run_logs" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PRISM_BATCH_SIZE="8", PRISM_GPU_MEMORY_FRACTION="0.5",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    status = {"batch_size": 8, "gpu_memory_fraction": 0.5, "reserve_ram_gib": 5,
              "jobs": [], "state": "running"}

    def save():
        temp = output / "status.tmp"
        temp.write_text(json.dumps(status, indent=2), encoding="utf-8")
        temp.replace(output / "status.json")

    save()
    print(f"Run logs: {output}", flush=True)
    for task in ("sis_binary", "neuronal_three_class", "neuronal_binary"):
        for method in ("mlp", "knn", "resinf"):
            job = dict(task=task, method=method, state="starting")
            status["jobs"].append(job)
            save()
            if psutil.virtual_memory().available < 7 * 1024**3:
                status["state"] = "stopped_low_available_ram_before_job"
                save()
                return 2
            cmd = [sys.executable, str(ROOT / "experiment_process/generate_resilience_inference_cache.py"),
                   "--generate", "--task", task, "--method", method, "--device", "cuda"]
            with (output / f"{task}_{method}.log").open("w", encoding="utf-8") as log:
                child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                job.update(pid=child.pid, state="running")
                save()
                try:
                    while child.poll() is None:
                        if psutil.virtual_memory().available < 5 * 1024**3:
                            job["state"] = "stopped_low_available_ram"
                            status["state"] = job["state"]
                            save()
                            return 2
                        time.sleep(0.5)
                except BaseException:
                    job["state"] = status["state"] = "interrupted"
                    save()
                    raise
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                job.update(exit_code=child.returncode, state="complete" if child.returncode == 0 else "failed")
                save()
                if child.returncode:
                    status["state"] = "failed"
                    save()
                    return child.returncode
    status["state"] = "complete"
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
