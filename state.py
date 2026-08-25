import threading
import torch


class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # dataset
        self.dataset_name = None
        self.train_loader = None
        self.test_loader = None
        self.num_classes = None
        self.in_channels = 3
        self.img_size = 32

        # model
        self.backbone_name = None
        self.backbone = None
        self.enhancement = "none"
        self.model = None
        self.surrogate_model = None  # independent model used for black-box transfer attacks

        # results
        self.clean_history = None
        self.adv_train_history = {}   # method -> history
        self.clean_eval = None        # {accuracy, ci_low, ci_high}
        self.clean_preds = None
        self.clean_labels = None
        self.attack_results = {}      # attack_name -> {accuracy, ci_low, ci_high, n_samples, example_images}
        self.attack_preds = {}        # attack_name -> (preds, labels)
        self.smoothing_result = None
        self.gradcam_results = []
        self.metrics = None
        self.stats = {}
        self.plots = {}
        self.logs = []

    def log(self, msg):
        self.logs.append(msg)
        if len(self.logs) > 500:
            self.logs = self.logs[-500:]

    def summary(self):
        return {
            "device": str(self.device),
            "dataset": self.dataset_name,
            "num_classes": self.num_classes,
            "img_size": self.img_size,
            "in_channels": self.in_channels,
            "backbone": self.backbone_name,
            "enhancement": self.enhancement,
            "has_model": self.model is not None,
            "clean_eval": self.clean_eval,
            "attack_results": {
                k: {kk: vv for kk, vv in v.items() if kk != "example_images_raw"}
                for k, v in self.attack_results.items()
            },
            "smoothing_result": self.smoothing_result,
            "metrics": self.metrics,
            "stats": self.stats,
            "has_clean_history": self.clean_history is not None,
            "adv_train_methods_run": list(self.adv_train_history.keys()),
            "plots": list(self.plots.keys()),
            "gradcam_count": len(self.gradcam_results),
            "recent_logs": self.logs[-20:],
        }


STATE = AppState()
