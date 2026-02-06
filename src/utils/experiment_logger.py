import os
import json
import csv
import yaml
import time
import logging
import subprocess
import torch
from datetime import datetime


class ExperimentLogger:
    def __init__(self, base_dir="outputs", exp_name="seg"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, f"{exp_name}_{timestamp}")
        self.viz_dir = os.path.join(self.run_dir, "viz")

        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)

        self.logger = logging.getLogger("experiment")
        self.logger.setLevel(logging.INFO)

        fh = logging.FileHandler(os.path.join(self.run_dir, "train.log"))
        sh = logging.StreamHandler()

        fmt = logging.Formatter("[%(asctime)s] %(message)s")
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)

        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

        self.csv_file = open(os.path.join(self.run_dir, "history.csv"), "w", newline="")
        self.csv_writer = None

        self.metrics_path = os.path.join(self.run_dir, "metrics.jsonl")

        self.logger.info(f"Run directory: {self.run_dir}")

        self._log_system_info()

    def _log_system_info(self):
        info = {
            "time": time.asctime(),
            "python": subprocess.check_output(["python", "--version"]).decode().strip(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }

        with open(os.path.join(self.run_dir, "system.txt"), "w") as f:
            for k, v in info.items():
                f.write(f"{k}: {v}\n")

    def save_config(self, cfg: dict):
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        except Exception:
            commit = "unknown"

        cfg = dict(cfg)
        cfg["git_commit"] = commit

        with open(os.path.join(self.run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f)

        self.logger.info("Saved config.yaml")

    def log_epoch(self, metrics: dict):
        metrics = dict(metrics)
        metrics["time"] = time.time()

        if self.csv_writer is None:
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=metrics.keys())
            self.csv_writer.writeheader()

        self.csv_writer.writerow(metrics)
        self.csv_file.flush()

        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        pretty = " | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                            for k, v in metrics.items())
        self.logger.info(pretty)

    def save_checkpoint(self, model, name="best.pt", **meta):
        path = os.path.join(self.run_dir, name)
        torch.save({"model": model.state_dict(), **meta}, path)
        self.logger.info(f"Saved checkpoint: {name}")

    def get_viz_dir(self):
        return self.viz_dir

    def get_best_checkpoint_path(self):
        return os.path.join(self.run_dir, "best.pt")
    
    def close(self):
        self.csv_file.close()
