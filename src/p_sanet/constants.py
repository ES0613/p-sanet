"""Fixed release constants for the inference pipeline."""

from __future__ import annotations

FRAMEWORK_NAME = "P-SANET"
DEFAULT_MODELS = ("model_1", "model_2", "model_3", "model_4", "model_5")

N_CONFORMERS = 100
RADIUS_SCALE = 0.856
UFF_MAX_ITERS = 1000
MAX_EMBED_ATTEMPTS = 50

EPSILON_CONSTANT_STARTS_K = (200.0, 300.0, 450.0, 800.0)
EPSILON_MIN_K = 1.0e-8
EPSILON_MAX_K = 5000.0
EPSILON_DENSE = 0.4

PVAP_MIN_PA = 1.0e-12
PVAP_MAX_PA = 1.0e9
FUGACITY_RESIDUAL_TOL = 1.0e-5
PHASE_GAP_TOL = 1.0e-12

SMI_TED_REVISION = "d823faeb77bf4d6ea7f0795d5cebd5ff2fb32344"
SMI_TED_REPOSITORY = "ibm-research/materials.smi-ted"
SMI_TED_FILES = {
    "load.py": "e4fb006a990b4aba5807d64b2dd7f0a17cb4ef79743e1dae130583de2639def9",
    "bert_vocab_curated.txt": "8576b60e838336837f9e894457ef144f440c4d31bc8ceda2315fb5b28a6dfd95",
    "smi-ted-Light_40.pt": "baf252dbc081a00c68d2fd6ed8b08a0db0fa15244cfea442d49f0619a3a65375",
}
SMI_TED_REMOTE_PATHS = {
    "load.py": "smi-ted/inference/smi_ted_light/load.py",
    "bert_vocab_curated.txt": "bert_vocab_curated.txt",
    "smi-ted-Light_40.pt": "smi-ted-Light_40.pt",
}
