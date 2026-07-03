# OmniTrainer package for GH05T3-Omni meta-evolution training
from .omni_trainer import OmniTrainer, OmniTrainerConfig
from .model_registry import (
    load_registry, save_registry, register_model, get_latest_model
)
