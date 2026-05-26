# Delegation Condition Demo

This repository contains a static GitHub Pages version of the delegation-condition image classification experiment.

- Uses the same 10 focal Tiny ImageNet images on every run.
- Shows 5 same-species candidate classes for each task.
- Shows 5 example images for every candidate class.
- Lets the subject classify directly or delegate the task to GoogLeNet Inception v3.
- Shows final accuracy and the precomputed class the AI would have picked for every task.

The deployable site lives in `docs/`.

## Local Preview

```bash
python3 -m http.server 8000 -d docs
```

Then open <http://127.0.0.1:8000>.

## Regenerate Static Data

The export script expects Tiny ImageNet locally and writes only the referenced images into `docs/images/`.

```bash
PYTHONPATH=.deps TORCH_HOME=.torch-cache python3 scripts/export_static.py
```

If `torchvision` is not already available, install dependencies first:

```bash
python3 -m pip install -r requirements.txt
```
