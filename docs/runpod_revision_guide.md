# Runpod guide for revision experiments

This guide uses a Runpod PyTorch template, SSH, and a normal Git checkout. It does not require an API key, a custom image, an S3 key, or a container registry credential. Create the Pod only when ready to run: GPU time is billed while it runs, and storage charges may continue after it is stopped. Before starting, choose a spending limit, review current Runpod pricing, and check how to stop the selected Pod. Back up the results before terminating a Pod or deleting storage.

## 1. Before creating a Pod

- Register your public SSH key in Runpod Credentials. Keep the private key on your own computer.
- Verify that GitHub CI passes for the exact commit to be run. Record that commit and review `configs/revision_experiments.json`. The full configuration runs 60 experiments (four datasets, three configurations, five seeds).
- Ensure the verified Parquet files in `/Users/rudyhendrawan/Projects/data/parquet/` and their conversion reports are available. The original CSVs total approximately 823 MB, but processed and resampled data, GAN cache, bootstrap calculations, and model training need substantially more disk and memory. File size is not a RAM estimate. Select storage and RAM with headroom and monitor usage during the pilot.
- Decide where results will be copied locally. A mounted volume may have different persistence and billing rules from the container disk; verify the selected Runpod template and volume settings before relying on them.
- Use a PyTorch template with a working NVIDIA GPU, a Python version compatible with the dependencies, SSH support and direct TCP access if SCP is needed. The installed PyTorch/CUDA and other libraries are not yet pinned by this project; record versions for each actual experiment.

## 2. Connect and check the Pod

Create a Pod with the chosen template and GPU. Copy the connection command from the Pod's **Connect** page and run it on your Mac using your private key if required. Use the exact host, user and port provided by Runpod; proxy SSH and direct TCP have different connection and file-transfer capabilities.

Inside the Pod:

```bash
nvidia-smi
python3 --version
python3 -c "import torch; print('torch:', torch.__version__, 'CUDA build:', torch.version.cuda, 'CUDA available:', torch.cuda.is_available())"
```

If CUDA is unavailable, investigate the template and GPU allocation before continuing. The scripts use a GPU for the PyTorch GAN models when CUDA is available; do not assume scikit-learn models, XGBoost, bootstrapping, or other steps run on the GPU.

## 3. Transfer datasets

If the Pod provides direct TCP SSH/SCP, use its **Connect** page to obtain the host, username and port. On the Mac, replace the placeholders with those exact values:

```bash
ssh -i ~/.ssh/runpod_research -p <PORT> <USER>@<HOST> 'mkdir -p /workspace/data/parquet'
scp -i ~/.ssh/runpod_research -P <PORT> \
  /Users/rudyhendrawan/Projects/data/parquet/*.parquet \
  /Users/rudyhendrawan/Projects/data/parquet/*.parquet.report.json \
  <USER>@<HOST>:/workspace/data/parquet/
```

If only proxy SSH is available, consult the Pod's connection instructions for a supported file-transfer method; do not assume the proxy command can be used directly with SCP. Confirm the files on the Pod with `ls -lh /workspace/data/parquet/`. Verify transferred Parquet SHA-256 checksums against the `parquet_sha256` values in the reports using `sha256sum` on the Pod. For CSV alternatives see [the Parquet data guide](parquet_data.md).

## 4. Install the project without replacing template PyTorch

On the Pod, replace `<COMMIT>` with the reviewed commit SHA:

```bash
cd /workspace
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/rudyhendrawn/data-synth-research.git
cd data-synth-research
git checkout --detach <COMMIT>
git rev-parse HEAD
```

If Git LFS is installed and the repository contains large tracked experiment artifacts, set `GIT_LFS_SKIP_SMUDGE=1` when cloning if those files are not needed; this avoids downloading historical caches. The datasets are transferred separately above.

Create an isolated environment that can see the template's PyTorch, then constrain pip to the installed torch version. If `venv` or pip is missing, install it using the template's documented method. Do not silently replace PyTorch if pip reports a dependency conflict; resolve the conflicting versions first.

```bash
python3 -m venv --system-site-packages /workspace/revision-venv
source /workspace/revision-venv/bin/activate
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pip freeze | grep '^torch==' > /tmp/runpod-torch-constraint.txt
test -s /tmp/runpod-torch-constraint.txt
python -m pip install -c /tmp/runpod-torch-constraint.txt -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available()"
python -m unittest discover -s tests -v
```

If `pip check` reports conflicts from packages inherited from the template, investigate them before proceeding. The current requirements are unpinned, so a successful setup today does not guarantee identical dependency resolution later. Save the resolved environment:

```bash
mkdir -p /workspace/revision-results
python -m pip freeze > /workspace/revision-results/pip-freeze.txt
nvidia-smi > /workspace/revision-results/nvidia-smi.txt
git rev-parse HEAD > /workspace/revision-results/git-commit.txt
```

## 5. Run a small pilot before the full study

Make a separate pilot configuration; do not overwrite the reviewed full configuration:

```bash
python - <<'PY'
import json
from pathlib import Path

config = json.loads(Path('configs/revision_experiments.json').read_text())
config['seeds'] = [42]
config['datasets'] = {'03_fraud_oracle.csv': config['datasets']['03_fraud_oracle.csv'][:1]}
config['baseline_ids'] = {'03_fraud_oracle.csv': config['baseline_ids']['03_fraud_oracle.csv']}
config['benchmark_options']['bootstrap_samples'] = 10
Path('/workspace/pilot_config.json').write_text(json.dumps(config, indent=2))
PY
python scripts/revision_experiments.py /workspace/pilot_config.json \
  --data-root /workspace/data/parquet --data-format parquet --output-dir /workspace/revision-results/pilot
```

This tests the pipeline and file paths with one small dataset and one baseline; it does **not** test GAN GPU training or large-dataset memory. Next, use a separate one-epoch GAN pilot:

```bash
python - <<'PY'
import json
from pathlib import Path

config = json.loads(Path('/workspace/pilot_config.json').read_text())
config['datasets']['03_fraud_oracle.csv'] = [
    {'exp_id': 'gan_gpu_pilot', 'components': {'model': 'LogisticRegression', 'oversampling': 'PytorchGAN'}}
]
config.pop('baseline_ids', None)
config['benchmark_options']['gan_epochs'] = 1
config['benchmark_options']['gan_batch_size'] = 128
Path('/workspace/pilot_gan_config.json').write_text(json.dumps(config, indent=2))
PY
python scripts/revision_experiments.py /workspace/pilot_gan_config.json \
  --data-root /workspace/data/parquet --data-format parquet --output-dir /workspace/revision-results/pilot-gan
```

Check the log says `Using device: cuda`. These pilots are operational checks and must not be reported as final experiments; neither confirms that the largest dataset fits in memory. Monitor `nvidia-smi`, system RAM, disk and elapsed time. Stop and investigate any failed run; the runner writes `runs.csv` and returns a nonzero exit status if a run fails.

## 6. Run the reviewed configuration

After reviewing pilot runtime, capacity, output and available balance, run:

```bash
set -o pipefail
python scripts/revision_experiments.py configs/revision_experiments.json \
  --data-root /workspace/data/parquet --data-format parquet --output-dir /workspace/revision-results/full \
  2>&1 | tee /workspace/revision-results/full-run.log
```

For long jobs use a persistent terminal session such as `tmux` if available. Each invocation creates a new timestamped directory and isolated GAN cache. The current runner checkpoints CSVs after each run but does not resume completed experiments automatically; after an interruption, inspect existing outputs before launching another run.

Check `runs.csv` for every expected run and status, `summary.csv` for successful seeds, `paired_deltas.csv` for matching seed counts, and the per-dataset sensitivity CSVs for whether precision targets were actually met. Existing historical output uses different code: record the commit and software versions when comparing results.

## 7. Retrieve results and stop

From the Mac, with direct TCP SCP and the values provided by Runpod:

```bash
mkdir -p ~/runpod-revision-results
scp -i ~/.ssh/runpod_research -P <PORT> -r \
  <USER>@<HOST>:/workspace/revision-results/ ~/runpod-revision-results/
```

Verify that the configuration, environment record, CSV files and logs are present locally before stopping or terminating the Pod. Check the Runpod dashboard for any remaining compute or storage charges. Stopping a Pod may release the GPU while leaving billable storage; terminating or deleting storage can destroy data.
