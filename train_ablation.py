"""
Thin shim: identical to train.py but routes all imports through
new_core_ablation and new_model_ablation.

Set ABLATION_FREQ_WARP=true or ABLATION_FREQ_MOD=true in the environment
before launching. train.py, new_model.py, new_core.py are untouched.
"""
import sys
import new_core_ablation as _abl_core
import new_model_ablation as _abl_model  # also sets sys.modules['new_core']

# Ensure any remaining `from new_model import ...` inside train.py resolves
# to the ablation-backed module.
sys.modules['new_core']  = _abl_core
sys.modules['new_model'] = _abl_model

from train import main  # noqa: F401 -- train.py now sees ablation stack

if __name__ == '__main__':
    main()
