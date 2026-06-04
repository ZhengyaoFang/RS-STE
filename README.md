# Recognition-Synergistic Scene Text Editing 🎨✨
[![arXiv](https://img.shields.io/badge/arXiv-2503.08387-b31b1b.svg)](https://arxiv.org/abs/2503.08387)

**CVPR 2025 | Official Implementation**
This is an official implementation of RS-STE proposed by our paper "Recognition-Synergistic Scene Text Editing" (CVPR 2025).

<div style="text-align: center;">
    <img src="docs/examples.png" alt="examples" style="width: 67%;" />
</div>

---

## 0️⃣ Install

**Environment🌄**
You can use uv to create the virtual environment:
```bash
pip install uv # install uv
uv venv # create virtual environment
source .venv/bin/activate
# Our CUDA version is 11.4
uv pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
uv pip install -r requirements.txt
```

Or use conda:
```bash
conda create -n rsste python=3.8
conda activate rsste
# Our CUDA version is 11.4
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

---

## 1️⃣ Dataset

### Synthetic Training Dataset

The synthetic training dataset is available on HuggingFace: [RS-STE-Dataset](https://huggingface.co/datasets/Nineve/RS-STE-Dataset).

Download it to your local directory and follow the dataset preparation steps below.

### Dataset Annotation Format

Organize the annotation file into the following format and save it directly to `data/annotation/`. The annotation files are `.pkl` (pickle) files with the following structure:

**For paired synthetic data (Stage 2 pretraining)**:
```python
import pickle
data = {
    "image1_paths": ['path/to/source_img1.png', ...],   # source images
    "image2_paths": ['path/to/target_img1.png', ...],   # target images (paired ground truth)
    "image1_rec": ['source text 1', ...],                # source image texts
    "image2_rec": ['target text 1', ...]                 # target texts (edited text)
}
with open("data/annotation/train_annotations.pkl", "wb") as f:
    pickle.dump(data, f)
```

**For real data (Stage 3 cycle finetuning)** — note: no `image2_paths` needed:
```python
import pickle
data = {
    "image1_paths": ['path/to/real_img1.png', ...],   # real scene text images
    "image1_rec": ['recognition text 1', ...],          # ground truth recognition
    "image2_rec": ['perturbed text 1', ...]             # target text for editing (can be randomly perturbed)
}
with open("data/annotation/real_train.pkl", "wb") as f:
    pickle.dump(data, f)
```

**For inference**:
```python
import pickle
data = {
    "image1_paths": ['example_data/3.png', 'example_data/47.png', ...],  # source images
    "image2_paths": [],                                                   # (Optional, for paired images)
    "image1_rec": ['MAIL', 'Colchester', ...],                           # (Optional, for training) source texts
    "image2_rec": ['ROYAL', 'Insurance', ...]                            # target texts
}
with open("data/annotation/inference_annotations.pkl", "wb") as f:
    pickle.dump(data, f)
```

### Dataset Preparation Tools

We provide scripts to prepare annotations from common dataset formats.

**For inference demo** (from example data):
```bash
uv run python tools/dataset_prepare.py
```

**For synthetic pair training** (from RS-STE-Dataset):
```bash
uv run python tools/train_annotation_prepare.py \
    --mode synth \
    --data_dir /path/to/RS-STE-Dataset \
    --output data/annotation/train_annotations.pkl
```

**For real data cycle training** (from scene text recognition datasets like Union14M):
```bash
uv run python tools/train_annotation_prepare.py \
    --mode real \
    --image_dir /path/to/real_images \
    --rec_file /path/to/recognition_labels.txt \
    --output data/annotation/real_train.pkl
```

The recognition label file should have one line per image in the format:
```
image_name.jpg HELLO WORLD
```

---

## 2️⃣ Inference

Download the pretrained checkpoints here: [BaiduNetDisk](https://pan.baidu.com/s/151EXQY5SdpETd3BS62dJYQ?pwd=db8s) or [Google Drive](https://drive.google.com/file/d/1ThBNiVR6JbgGXX1sy_2N9KXJZbmJEkSI/view?usp=drive_link).

Place the checkpoint in the `weights/` directory, then run:
```bash
uv run python inference.py --resume path/to/checkpoint
```

---

## 3️⃣ Training

RS-STE uses a **three-stage training pipeline**:

| Stage | Description | Training Data | Key Losses |
|-------|-------------|---------------|------------|
| **Stage 1** | VQ-VAE pretraining | Unlabeled scene text images (Union14M, Tamper, etc.) | Reconstruction + LPIPS + GAN + Codebook |
| **Stage 2** | Synthetic pair pretraining | Paired synthetic data (HuggingFace RS-STE-Dataset) | Image MSE + Perceptual + Recognition |
| **Stage 3** | Real data cycle finetuning | Unpaired real scene text images | Cycle-consistency + Recognition |

The training flow is:
1. **Stage 1** trains the VQ-VAE to compress scene text images into a compact latent space. Only the **decoder** part is used in later stages (frozen).
2. **Stage 2** trains the editing transformer on paired synthetic data to learn text editing in the VQ-VAE latent space.
3. **Stage 3** finetunes on real images using cycle-consistency to bridge the domain gap.

> **Pretrained checkpoints are available** for all stages. If you just want to run inference, skip to [Section 2️⃣](#2️⃣-inference). If you want to reproduce training from scratch, follow Stage 1 → Stage 2 → Stage 3 below.

---

### Stage 1: VQ-VAE Pretraining

The VQ-VAE compresses 32×128 scene text images into a 8×32×3 latent representation using a codebook of 8192 entries. This stage uses the code in `stage1_vqvae/`.

#### Download Pretrained VQ-VAE Checkpoint

The pretrained VQ-VAE checkpoint will be available on HuggingFace: [RS-STE-Checkpoints](https://huggingface.co/Nineve/RS-STE-Checkpoints).

File: `model_vqd3s8192f4.ckpt` (contains encoder + decoder + codebook)

#### Prepare Training Data

First, create the image list file for training:

```bash
cd stage1_vqvae
# Edit tools/create_data_file.py to point to your scene text image directories
uv run python tools/create_data_file.py
```

This generates `data/train.txt` and `data/test.txt` containing paths to training images.

#### Launch VQ-VAE Training

From the `stage1_vqvae/` directory:

```bash
# Stage 1a: Train with smaller codebook as warmup (optional)
uv run python main.py --base configs/d256_s1024.yaml -t True \
    --name d256_s1024 --gpus 0,1,2,3

# Stage 1b: Train with final codebook (d=3, s=8192), initialized from warmup
uv run python main.py --base configs/d3_s8192_contiguously.yaml -t True \
    --name d3_s8192 --gpus 0,1,2,3 \
    --resume_from_checkpoint pretrained_weights/model_vqd3s8192f4.ckpt
```

The pretrained VQ-VAE checkpoint (`model_vqd3s8192f4.ckpt`) will be downloaded from HuggingFace to `pretrained_weights/`.

**Key configs**:
- `configs/d3_s8192_contiguously.yaml`: Final config with `embed_dim=3, n_embed=8192` (matches `configs/vqgan_decoder.yaml`)
- `configs/d256_s1024.yaml`: Intermediate warmup config with `embed_dim=256, n_embed=1024`

Checkpoints are saved to `logs/<timestamp>_<name>/checkpoints/`.

#### Extract Decoder for RS-STE

After Stage 1 training, the decoder weights are extracted and baked into the RS-STE checkpoint used in Stage 2/3. We provide the extracted decoder as part of the released RS-STE checkpoint.

---

### Stage 2: Synthetic Pair Pretraining

Train the editing transformer on paired synthetic data. Uses the frozen VQ-VAE decoder from Stage 1.

**Prerequisites**: Download the synthetic dataset from [HuggingFace](https://huggingface.co/datasets/Nineve/RS-STE-Dataset).

```bash
# Back in the RS-STE root directory
# Prepare annotations from the downloaded dataset
uv run python tools/train_annotation_prepare.py \
    --mode synth \
    --data_dir /path/to/RS-STE-Dataset \
    --output data/annotation/train_annotations.pkl

# Launch Stage 2 training
uv run python main.py \
    --name stage2_synth_pair \
    --vqgan_config configs/vqgan_decoder.yaml \
    --transformer_config configs/synth_pair.yaml
```

**Key parameters in `configs/synth_pair.yaml`**:
- `model.base_learning_rate`: 4.5e-06 (scaled by GPUs × batch_size × grad_accumulation)
- `data.params.batch_size`: 8 per GPU
- `data.params.train.params.size`: 32 (image height — images are resized to 32×128)

Training logs and checkpoints are saved to `logs/<timestamp>_<name>/`. Checkpoints are saved every 1000 training steps.

---

### Stage 3: Real Data Cycle Finetuning

After Stage 2 pretraining, finetune on real scene text data with cycle-consistency:

```bash
# Prepare real data annotations (e.g., from Union14M-L)
uv run python tools/train_annotation_prepare.py \
    --mode real \
    --image_dir /path/to/real_scene_text_images \
    --rec_file /path/to/recognition_labels.txt \
    --output data/annotation/real_train.pkl

# Launch cycle finetuning from Stage 2 checkpoint
uv run python main.py \
    --name stage3_real_cycle \
    --vqgan_config configs/vqgan_decoder.yaml \
    --transformer_config configs/real_cycle.yaml \
    --resume path/to/stage2_checkpoint.ckpt
```

**Key parameters in `configs/real_cycle.yaml`**:
- `model.training_mode`: "real_cycle" (enables cycle-consistency training)
- `data.params.batch_size`: 4 per GPU (reduced due to cycle forward-backward pass)
- Recognition loss weight is higher (50×) to maintain text editing accuracy

**How cycle training works**: In the `real_cycle` mode, the model performs:
1. Forward pass: image1 + rec2 → edit → image2_pred
2. Cycle pass: image2_pred + rec1 → reconstruct → image1_pred
3. Losses: image1_reconstruction (MSE + Perceptual) + recognition losses on both directions

This ensures the model learns to handle real-world image distributions while preserving text editing capability.

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--name` | Experiment name | `rs-ste` |
| `--vqgan_config` | Path to VQ-VAE decoder config | `configs/vqgan_decoder.yaml` |
| `--transformer_config` | Path to transformer training config | `configs/synth_pair.yaml` |
| `--resume` | Path to checkpoint for resume/finetune | `None` |
| `--logdir` | Directory for logs and checkpoints | `logs` |
| `--seed` | Random seed | `42` |
| `--scale_lr` | Scale learning rate by GPUs×batch | `True` |

### Training Configuration

GPU configuration is set in `configs/lightning.yaml`:
```yaml
trainer:
  accelerator: ddp
  gpus: 0,1,2,3   # Modify according to your setup
```

For Stage 1 VQ-VAE training, GPU config is passed via command line (`--gpus 0,1,2,3`).

### Pretrained Checkpoints

We provide the following pretrained checkpoints on HuggingFace (coming soon) and cloud drives:

| Checkpoint | Stage | File | BaiduNetDisk | Google Drive |
|-----------|-------|------|-------------|--------------|
| VQ-VAE | Stage 1 | `model_vqd3s8192f4.ckpt` | [Link](https://pan.baidu.com/s/12z_jftTVY9bseuvA0Xe2Hw?pwd=auql) | — |
| RS-STE (full) | Stage 2+3 | `model.pth` | [Link](https://pan.baidu.com/s/151EXQY5SdpETd3BS62dJYQ?pwd=db8s) | [Link](https://drive.google.com/file/d/1ThBNiVR6JbgGXX1sy_2N9KXJZbmJEkSI/view?usp=drive_link) |

> **Note**: The RS-STE checkpoint (`model.pth`) is self-contained — it includes both the transformer weights and the VQ-VAE decoder weights. You only need the VQ-VAE checkpoint if you're training from scratch.

---

## 4️⃣ Evaluation

We provide evaluation scripts for both image quality metrics and recognition accuracy.

### Image Quality Metrics (PSNR / SSIM / MSE / FID)

For synthetic test sets where ground truth target images are available (e.g., Tamper-Syn2k):

```bash
uv run python evaluation/evaluation_TamperSyn.py \
    --target_path path/to/generated_images \
    --gt_path path/to/ground_truth_images
```

This computes PSNR, SSIM, MSE, and FID between the generated and ground truth images.

### Recognition Accuracy

For real scene text editing evaluation (e.g., Tamper-Scene):

1. First generate edited images using inference:
```bash
uv run python inference.py \
    --resume path/to/checkpoint \
    --target_path output/edited_images
```

2. Then evaluate recognition accuracy:
```bash
uv run python evaluation/eval_real.py \
    --image_folder output/edited_images \
    --gt_file path/to/ground_truth_texts.txt \
    --saved_model pretrained_model/TPS-ResNet-BiLSTM-Attn.pth
```

> **Note**: The recognition evaluation requires a pretrained STR (Scene Text Recognition) model. Download the TPS-ResNet-BiLSTM-Attn model from [BaiduNetDisk](https://pan.baidu.com/s/12z_jftTVY9bseuvA0Xe2Hw?pwd=auql) and place it in `pretrained_model/`.

The evaluation script reports:
- **Accuracy**: Percentage of correctly recognized edited text
- **Normalized Edit Distance**: 1 - (edit distance / max length)

---

## 📁 Project Structure

```
RS-STE/
├── configs/
│   ├── synth_pair.yaml            # Stage 2: Synthetic pair training config
│   ├── real_cycle.yaml            # Stage 3: Real data cycle training config
│   ├── vqgan_decoder.yaml         # VQ-VAE decoder model config
│   └── lightning.yaml             # PyTorch Lightning trainer config
├── data/
│   ├── dataset.py                 # Dataset classes (TrainingDataset, InferenceDataset)
│   ├── alphabet/                  # Character alphabet files
│   └── annotation/                # Annotation pickle files (generated)
├── model/
│   ├── model.py                   # RS-STE main model (LightningModule)
│   ├── minGPT.py                  # GPT transformer backbone
│   ├── decoder.py                 # VQ-VAE decoder
│   └── utils.py                   # String label converter, etc.
├── stage1_vqvae/                  # Stage 1: VQ-VAE pretraining
│   ├── main.py                    # VQ-VAE training entry point
│   ├── configs/                   # VQ-VAE training configs
│   ├── model/                     # VQ-VAE model (encoder + decoder + codebook)
│   ├── dataset/                   # VQ-VAE dataset classes
│   └── tools/                     # Data preparation tools
├── evaluation/
│   ├── evaluation_TamperSyn.py    # PSNR/SSIM/MSE/FID evaluation
│   ├── eval_real.py               # Recognition accuracy evaluation
│   ├── eval_utils.py              # Evaluation utilities
│   ├── rec_model/                 # Scene text recognition model
│   └── pytorch_fid/               # FID calculation
├── tools/
│   ├── dataset_prepare.py         # Inference annotation preparation
│   └── train_annotation_prepare.py # Training annotation preparation
├── main.py                        # RS-STE training entry point (Stage 2 & 3)
├── inference.py                   # Inference entry point
├── requirements.txt
└── README.md
```

---

## 🎬 Citation
```
@InProceedings{Fang_2025_CVPR,
    author    = {Fang, Zhengyao and Lyu, Pengyuan and Wu, Jingjing and Zhang, Chengquan and Yu, Jun and Lu, Guangming and Pei, Wenjie},
    title     = {Recognition-Synergistic Scene Text Editing},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {13104-13113}
}
```


## 📄 License
RS-STE is under MIT.
