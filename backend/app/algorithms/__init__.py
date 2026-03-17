"""
Anomaly detection algorithms and evaluation.

Migrated from the old project (ts-iteration-loop/services/inference/).

Modules:
- iforest, ensemble, wavelet, adtk_hbos: CPU detection algorithms (detect())
- dispatcher: unified entry point for CPU (run) and GPU (run_vllm) methods
- preprocessing: data quality, downsampling, STL, noise detection
- helpers: shared primitives (mask, clustering, wavelet refinement)
- lb_eval: online segment scoring (KS-test, no ground truth needed)
- eval_metrics: offline evaluation (Precision/Recall/F1/MAR/FAR, needs labels)
"""
