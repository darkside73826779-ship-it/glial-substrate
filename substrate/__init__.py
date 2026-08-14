from .layer import SubstrateConfig, SubstrateLinear, GateNet
from .model import TinyTransformer, SubstrateController, Block
from .tasks import AssociativeRecall, ContinualLM

__all__ = [
    "SubstrateConfig", "SubstrateLinear", "GateNet",
    "TinyTransformer", "SubstrateController", "Block",
    "AssociativeRecall", "ContinualLM",
]
__version__ = "0.3.0-alpha1"   # "Read Organ Online" (2026-08-14); see CHANGELOG.md
