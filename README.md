# CARformer

[MICCAI 2026] Official code for **"CARformer: Class-Aware Representation Learning for Small-Cohort, Imbalanced Psychiatric Disorder Classification in Brain MRI"**.

## Setup

```bash
conda create -n carformer python=3.10 -y
conda activate carformer
pip install torch==2.6.0 torchvision==0.21.0
pip install -r requirements.txt
```

Please install the PyTorch build that matches your CUDA version if the command above is not suitable for your system.

## Data

Prepare a JSON split file with `training`, `validation`, and `test` lists. Each sample should contain an `image` list and an integer `label`.

```json
{
  "training": [
    {"image": ["site_a/control_001/T1.nii.gz"], "label": 0}
  ],
  "validation": [
    {"image": ["site_b/control_002/T1.nii.gz"], "label": 0}
  ],
  "test": [
    {"image": ["site_c/control_003/T1.nii.gz"], "label": 0}
  ]
}
```

Image paths can be absolute or relative to `--data_root`. See `downstream/example_dataset.json`.

## Pretrained Weight

Download the BrainMVP UniFormer pretrained weight from the official BrainMVP repository and place it outside git, for example:

```text
pretrained/BrainMVP_uniformer.pt
```

BrainMVP repo: https://github.com/shaohao011/BrainMVP

## Train

Edit paths in `downstream/do_train.sh`, then run:

```bash
cd downstream
bash do_train.sh
```

Checkpoints are selected by validation macro AUC. By default, the script keeps the top 3 validation-AUC checkpoints.

## Test

Edit the checkpoint path in `downstream/do_test.sh`, then run:

```bash
cd downstream
bash do_test.sh
```

## Citation

Formal proceedings metadata will be updated after the MICCAI 2026 citation is available.

```bibtex
@inproceedings{fu2026carformer,
  title = {CARformer: Class-Aware Representation Learning for Small-Cohort, Imbalanced Psychiatric Disorder Classification in Brain MRI},
  author = {Fu, Xingyue and others},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI},
  year = {2026},
  note = {To appear}
}
```

## Acknowledgement

This implementation uses the BrainMVP pretrained UniFormer weight. Please download the pretrained weight from the official BrainMVP repository and cite BrainMVP when using it.
