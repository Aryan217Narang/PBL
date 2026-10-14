# DEPRECATED: Legacy TensorFlow Prototype Pipeline

> **Notice:** The scripts in this directory (`legacy_tf_pipeline/`) represent the original TensorFlow/Keras CPU prototype.
> 
> **This pipeline is now DEPRECATED** in favor of the canonical, GPU-accelerated PyTorch + ART pipeline located in `code/nids hybrid defence/`.

### Why this pipeline was deprecated:
1. **CPU bottleneck:** TensorFlow on native Windows lacks official GPU acceleration for modern TF versions; training and PGD generation were extremely slow.
2. **Framework fragmentation:** Maintaining two parallel pipelines (TensorFlow and PyTorch) created duplicated code, metric divergence, and confusion.
3. **Canonical Pipeline:** All active development, adversarial defenses, PGD/PIOA training, spatial smoothing, temporal evaluations, and AICC/TCC consistency checks are standardized in `code/nids hybrid defence/`.

To run the unified active pipeline, execute:
```bash
cd "code/nids hybrid defence"
python run_pipeline.py
```
Or run the top-level runner:
```bash
python code/main.py
```
