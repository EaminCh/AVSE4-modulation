"""
Thin shim: identical to new_model.py but backed by new_core_ablation.

Set ABLATION_FREQ_WARP=true or ABLATION_FREQ_MOD=true in the environment
before launching to activate the corresponding frequency experiment.
new_core.py and new_model.py are untouched.
"""
import sys
import new_core_ablation as _abl_core

# Redirect `from new_core import ...` inside new_model.py to the ablation
# classes. Must be done before new_model is imported/loaded.
sys.modules['new_core'] = _abl_core

from new_model import AVSE4BaselineModule  # noqa: F401 -- re-exported
