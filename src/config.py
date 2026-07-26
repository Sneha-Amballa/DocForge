import os
from pathlib import Path
from typing import Union, Dict, Any, List, Optional

class DatasetConfig:
    """Configuration loader for DocForge dataset structure, preprocessing, and model settings.

    Loads configuration parameters from 'config.yaml' and exposes them as attributes.
    Supports environment variable overrides for critical paths.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize configurations by loading from config.yaml.

        Args:
            config_path: Custom path to config.yaml. Defaults to 'config.yaml' at workspace root.
        """
        # Resolve workspace directory
        self.workspace_dir = Path(__file__).resolve().parent.parent

        if config_path:
            self.config_path = Path(config_path).resolve()
        else:
            self.config_path = self.workspace_dir / "config.yaml"

        self.data: Dict[str, Any] = {}
        self.load_config()

        # Extract values
        self._parse_config()

    def load_config(self) -> None:
        """Load configuration dictionary from YAML file with safety fallbacks."""
        if not self.config_path.exists():
            # If config file does not exist, use default dictionary values
            self.data = self._get_defaults()
            return

        with open(self.config_path, "r", encoding="utf-8") as f:
            content = f.read()

        try:
            import yaml
            self.data = yaml.safe_load(content) or {}
        except ImportError:
            # Fallback simple YAML parser for basic nested structures
            self.data = self._parse_simple_yaml(content)

    def _parse_simple_yaml(self, text: str) -> Dict[str, Any]:
        """A simple fallback parser for standard key-value and nested YAML files."""
        res: Dict[str, Any] = {}
        # Keep track of path in dict
        stack: List[tuple[int, dict]] = [(-1, res)]
        
        for line in text.splitlines():
            # Remove comments and strip tail
            line_no_comment = line.split("#")[0].rstrip()
            if not line_no_comment.strip():
                continue
            
            indent = len(line_no_comment) - len(line_no_comment.lstrip())
            content = line_no_comment.strip()
            
            if ":" in content:
                k, v = content.split(":", 1)
                k = k.strip()
                v = v.strip()
                
                # Check stack to see where this indent fits
                while len(stack) > 1 and stack[-1][0] >= indent:
                    stack.pop()
                
                current_dict = stack[-1][1]
                
                if not v:
                    # Nested section
                    new_dict: Dict[str, Any] = {}
                    current_dict[k] = new_dict
                    stack.append((indent, new_dict))
                else:
                    # Single value
                    # Strip quotes
                    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                        val: Any = v[1:-1]
                    elif v.lower() == "true":
                        val = True
                    elif v.lower() == "false":
                        val = False
                    elif v.lower() == "null":
                        val = None
                    elif v.startswith("[") and v.endswith("]"):
                        # Basic list parsing
                        items = [x.strip() for x in v[1:-1].split(",")]
                        parsed_items = []
                        for item in items:
                            if not item:
                                continue
                            if (item.startswith('"') and item.endswith('"')) or (item.startswith("'") and item.endswith("'")):
                                parsed_items.append(item[1:-1])
                            else:
                                try:
                                    parsed_items.append(float(item) if "." in item else int(item))
                                except ValueError:
                                    parsed_items.append(item)
                        val = parsed_items
                    else:
                        try:
                            val = float(v) if "." in v else int(v)
                        except ValueError:
                            val = v
                    
                    current_dict[k] = val
                    
        return res

    def _get_defaults(self) -> Dict[str, Any]:
        """Provide default configurations if YAML is missing."""
        return {
            "dataset": {
                "root_path": "dataset",
                "training_set": "dataset/DocTamperV1-TrainingSet",
                "testing_set": "dataset/DocTamperV1-TestingSet",
                "scd_set": "dataset/DocTamperV1-SCD",
                "fcd_set": "dataset/DocTamperV1-FCD",
            },
            "preprocessing": {
                "image_size": [512, 512],
                "preserve_aspect_ratio": True,
                "padding_enabled": True,
                "normalization_mean": [0.485, 0.456, 0.406],
                "normalization_std": [0.229, 0.224, 0.225]
            },
            "augmentation": {"enabled": False},
            "dataloader": {
                "batch_size": 4,
                "shuffle": True,
                "num_workers": 0,
                "pin_memory": True,
                "persistent_workers": False
            },
            "model": {
                "processor_name": "Qwen/Qwen2-VL-2B-Instruct",
                "prompt_template": "Detect document tampering."
            },
            "split": {
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "test_ratio": 0.1,
                "seed": 42
            },
            "outputs": {
                "root_path": "outputs",
                "overlays_dir": "outputs/overlays",
                "samples_dir": "outputs/samples",
                "reports_dir": "outputs/reports",
                "statistics_dir": "outputs/statistics",
                "checkpoints_dir": "outputs/checkpoints"
            },
            "training": {
                "epochs": 3,
                "learning_rate": 0.0002,
                "weight_decay": 0.01,
                "gradient_accumulation_steps": 2,
                "grad_clip_norm": 1.0,
                "precision": "fp32",
                "scheduler_type": "cosine",
                "warmup_steps": 100,
                "checkpoint_interval": 1,
                "early_stopping_patience": 3,
                "logging_backends": ["tensorboard", "terminal"],
                "dry_run": False
            }
        }

    def _parse_config(self) -> None:
        """Parse raw dictionary config into class attributes."""
        # Dataset Paths
        ds_section = self.data.get("dataset", {})
        ds_root_env = os.getenv("DOCFORGE_DATASET_ROOT")
        self.dataset_root = Path(ds_root_env).resolve() if ds_root_env else self.workspace_dir / ds_section.get("root_path", "dataset")

        self.training_set = self.dataset_root / Path(ds_section.get("training_set", "DocTamperV1-TrainingSet")).name
        self.testing_set = self.dataset_root / Path(ds_section.get("testing_set", "DocTamperV1-TestingSet")).name
        self.scd_set = self.dataset_root / Path(ds_section.get("scd_set", "DocTamperV1-SCD")).name
        self.fcd_set = self.dataset_root / Path(ds_section.get("fcd_set", "DocTamperV1-FCD")).name

        # Preprocessing settings
        pre_section = self.data.get("preprocessing", {})
        self.image_size = tuple(pre_section.get("image_size", [512, 512]))
        self.preserve_aspect_ratio = bool(pre_section.get("preserve_aspect_ratio", True))
        self.padding_enabled = bool(pre_section.get("padding_enabled", True))
        self.normalization_mean = pre_section.get("normalization_mean", [0.485, 0.456, 0.406])
        self.normalization_std = pre_section.get("normalization_std", [0.229, 0.224, 0.225])

        # Augmentation settings
        self.aug_config = self.data.get("augmentation", {})
        self.aug_enabled = bool(self.aug_config.get("enabled", False))

        # Dataloader settings
        dl_section = self.data.get("dataloader", {})
        self.batch_size = int(dl_section.get("batch_size", 4))
        self.shuffle = bool(dl_section.get("shuffle", True))
        self.num_workers = int(dl_section.get("num_workers", 0))
        self.pin_memory = bool(dl_section.get("pin_memory", True))
        self.persistent_workers = bool(dl_section.get("persistent_workers", False))

        # Model settings
        model_section = self.data.get("model", {})
        self.processor_name = str(model_section.get("processor_name", "Qwen/Qwen2-VL-2B-Instruct"))
        self.prompt_template = str(model_section.get("prompt_template", "Detect document tampering."))

        # Split settings
        split_section = self.data.get("split", {})
        self.train_ratio = float(split_section.get("train_ratio", 0.8))
        self.val_ratio = float(split_section.get("val_ratio", 0.1))
        self.test_ratio = float(split_section.get("test_ratio", 0.1))
        self.seed = int(split_section.get("seed", 42))

        # Outputs
        out_section = self.data.get("outputs", {})
        out_root_env = os.getenv("DOCFORGE_OUTPUT_ROOT")
        self.output_root = Path(out_root_env).resolve() if out_root_env else self.workspace_dir / out_section.get("root_path", "outputs")
        self.overlays_dir = self.output_root / Path(out_section.get("overlays_dir", "overlays")).name
        self.samples_dir = self.output_root / Path(out_section.get("samples_dir", "samples")).name
        self.reports_dir = self.output_root / Path(out_section.get("reports_dir", "reports")).name
        self.statistics_dir = self.output_root / Path(out_section.get("statistics_dir", "statistics")).name
        self.checkpoints_dir = self.output_root / Path(out_section.get("checkpoints_dir", "checkpoints")).name

        # Training settings
        train_section = self.data.get("training", {})
        self.epochs = int(train_section.get("epochs", 3))
        self.learning_rate = float(train_section.get("learning_rate", 0.0002))
        self.weight_decay = float(train_section.get("weight_decay", 0.01))
        self.gradient_accumulation_steps = int(train_section.get("gradient_accumulation_steps", 2))
        self.grad_clip_norm = float(train_section.get("grad_clip_norm", 1.0))
        self.precision = str(train_section.get("precision", "fp32"))
        self.scheduler_type = str(train_section.get("scheduler_type", "cosine"))
        self.warmup_steps = int(train_section.get("warmup_steps", 100))
        self.checkpoint_interval = int(train_section.get("checkpoint_interval", 1))
        self.early_stopping_patience = int(train_section.get("early_stopping_patience", 3))
        self.logging_backends = list(train_section.get("logging_backends", ["tensorboard", "terminal"]))
        self.dry_run = bool(train_section.get("dry_run", True))

    def ensure_output_dirs(self) -> None:
        """Create all output directories if they do not exist."""
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.overlays_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.statistics_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def get_subset_path(self, subset_name: str) -> Path:
        """Get the path for a specific dataset subset.

        Args:
            subset_name: Name of the subset ('TrainingSet', 'TestingSet', 'SCD', 'FCD').

        Returns:
            Path: Absolute path to the subset directory.

        Raises:
            ValueError: If the subset name is invalid.
        """
        mapping: Dict[str, Path] = {
            "trainingset": self.training_set,
            "testingset": self.testing_set,
            "scd": self.scd_set,
            "fcd": self.fcd_set,
            "training": self.training_set,
            "testing": self.testing_set
        }
        key = subset_name.lower().replace("-", "").replace("_", "")
        if key not in mapping:
            raise ValueError(
                f"Unknown subset name: {subset_name}. "
                f"Available subsets: TrainingSet, TestingSet, SCD, FCD."
            )
        return mapping[key]

    def __repr__(self) -> str:
        return (
            f"DatasetConfig(\n"
            f"  dataset_root={self.dataset_root},\n"
            f"  training_set={self.training_set},\n"
            f"  testing_set={self.testing_set},\n"
            f"  scd_set={self.scd_set},\n"
            f"  fcd_set={self.fcd_set},\n"
            f"  output_root={self.output_root},\n"
            f"  image_size={self.image_size},\n"
            f"  batch_size={self.batch_size}\n"
            f")"
        )
