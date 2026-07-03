# Phase 6+ modules + fallback for main evolution classes
from .meta_learning import get_meta_learning_engine
from .self_improving_loop_v2 import get_self_improving_loop_v2

# Load RoleEvolutionManager from the sibling evolution.py file via importlib.
# `from ..evolution import` is circular because the package directory shadows
# the module file. We must register the module in sys.modules before exec so
# that @dataclass can resolve cls.__module__ correctly.
import importlib.util as _ilu
import pathlib as _pl
import sys as _sys
try:
    _KEY  = "backend.oss._evolution_base"
    _spec = _ilu.spec_from_file_location(
        _KEY,
        _pl.Path(__file__).parent.parent / "evolution.py",
    )
    _mod = _ilu.module_from_spec(_spec)
    _sys.modules[_KEY] = _mod          # register BEFORE exec so @dataclass works
    _spec.loader.exec_module(_mod)
    RoleEvolutionManager = _mod.RoleEvolutionManager
    EvolutionaryPressureEngine = _mod.EvolutionaryPressureEngine
except Exception:
    RoleEvolutionManager = None
    EvolutionaryPressureEngine = None

__all__ = ["get_meta_learning_engine", "get_self_improving_loop_v2", "RoleEvolutionManager", "EvolutionaryPressureEngine"]
