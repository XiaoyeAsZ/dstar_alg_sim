

## Quick start

```bash
cd ~/dstar_artifact

# Profile and save the sparse-attention mask.
python dstar.py profile-mask --config config/pixart-sigma-1024-sparse-profile.json

# Generate one prompt and write a manageable trace set.
python dstar.py infer --config config/pixart-sigma-1024-trace.json

# Generate 1000 evaluation samples with quantization and the saved mask.
python dstar.py infer --config config/pixart-sigma-1024-all.json

# Evaluate generated files and save a JSON report.
python dstar.py evaluate outputs/pixart-sigma-1024/all --json outputs/quality.json

# Evaluate generated videos with FVD against a real-video directory.
python dstar.py evaluate outputs/latte/all \
  --reference-dir /path/to/real/videos \
  --metrics is fvd \
  --json outputs/latte-quality.json

# Simulate trace files produced by the prompt config.
python dstar.py simulate traces/pixart-sigma-1024 --log-path outputs/simulator.log
```
