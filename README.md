# Adapting Autoregressive Models to Diffusion Models

Implementation of adapting pre-trained autoregressive language models to diffusion models, based on the ICLR 2025 paper "Scaling Diffusion Language Models via Adaptation from Autoregressive Models".

## Requirements

```bash
pip install torch transformers modelscope datasets
pip install jupyter matplotlib numpy tqdm accelerate
```

## Data

Download ArXiv dataset and save as `autodl-tmp/arxiv_train-1.jsonl`:

```python
from datasets import load_dataset
dataset = load_dataset('scientific_papers', 'arxiv', split='train')
dataset.to_json('autodl-tmp/arxiv_train-1.jsonl')
```

## Notebooks

### 1. `gpt2_diffusion_full_training__1_.ipynb`

Main training pipeline with attention mask annealing and shift operation.

- **Model**: GPT-2 (124M)
- **Training**: 10 epochs with progressive attention annealing (10,000 steps)
- **Features**: Shift operation enabled, checkpoint saving
- **Runtime**: ~1-2 hours on RTX 5090

### 2. `shift_comparison_training_optimized__1_.ipynb`

Ablation study comparing training with vs. without shift operation.

- **Comparison**: Two models trained in parallel
- **Training**: 1 epoch on subset of data
- **Output**: Comparative statistics and loss curves
- **Runtime**: ~1-2 minutes on RTX 5090

### 3. `diffusion_galactica_complete__2___1_.ipynb`

Adaptation of Galactica model (specialized for scientific text).

- **Model**: Galactica-125M
- **Training**: 3 epochs on ArXiv papers
- **Features**: Numerical stability checks, best model saving
- **Runtime**: ~15-20 minutes on RTX 5090

## Usage

```bash
jupyter notebook [notebook_name].ipynb
```

Run all cells sequentially. Models will be saved in checkpoint directories.

## Key Hyperparameters

- Batch size: 8
- Learning rate: 5e-5
- Sequence length: 512
- Annealing steps: 10,000
- Mixed precision: FP16

## Citation

```bibtex
@inproceedings{diffusion2025,
  title={Scaling Diffusion Language Models via Adaptation from Autoregressive Models},
  booktitle={ICLR},
  year={2025}
}
```
