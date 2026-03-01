#!/usr/bin/env python3
"""
AlphaGenome Fine-tuning Script

Adapted from: https://colab.research.google.com/github/google-deepmind/alphagenome_research/blob/main/notebooks/finetune.ipynb
"""

import argparse
import csv
import os
import sys

# ── Suppress noisy C++ (glog) rendezvous warnings ─────────────────────────
os.environ.setdefault('GLOG_minloglevel', '3')
# ────────────────────────────────────────────────────────────────────────────

# 1. Environment Setup
import tensorflow as tf

# Hide local GPUs/TPUs. TensorFlow only used for data loading.
tf.config.set_visible_devices([], 'GPU')
tf.config.set_visible_devices([], 'TPU')

# 2. Imports
# Import the necessary libraries:
# - `alphagenome_research.finetuning` contains the finetuning utilities
# - `alphagenome_research.model` provides the model architecture and metadata handling
# - `alphagenome.data` provides genomic data utilities

import dataclasses
from datetime import datetime
import os
import pprint

from alphagenome.data import fold_intervals
from alphagenome.data import genome
from alphagenome.visualization import plot_components
from alphagenome_research.finetuning import dataset as dataset_lib
from alphagenome_research.finetuning import finetune
from alphagenome_research.model import dna_model
from alphagenome_research.model.metadata import metadata as metadata_lib
from etils import epath
import huggingface_hub
import haiku as hk
import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np
import optax
import orbax.checkpoint as ocp
import pandas as pd

# 3. Configuration
# Define the key hyperparameters for finetuning:
# - `LEARNING_RATE`: Controls the step size during optimization
# - `MODEL_VERSION`: Which pretrained fold to use (FOLD_0 through FOLD_3)
# - `NUM_TRAIN_STEPS`: Number of training steps for which we optimize the model.
# - `SEQUENCE_LENGTH`: Length of input DNA sequences (1M bp = 2^20). Training requires at least 2**17.
# - `BATCH_SIZE`: Number of samples per device.
# - `ORGANISM`: Target organism for predictions. Set to HOMO_SAPIENS for the official baseline run.

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Fine-tune AlphaGenome baseline run (HOMO_SAPIENS + ENCODE tracks)')
parser.add_argument('--learning-rate', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--num-steps', type=int, default=1000, help='Number of training steps')
parser.add_argument('--batch-size', type=int, default=1, help='Batch size per device')
parser.add_argument('--sequence-length', type=int, default=131072, help='Sequence length (must be >= 2^17)')
parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/fine_tuned', help='Output checkpoint directory')
parser.add_argument('--pretrained-checkpoint', type=str, 
                    default='/home/ray/AG_Finetune/checkpoints/pre_trained/alphagenome-all-folds',
                    help='Path to pretrained checkpoint')
parser.add_argument('--fasta', type=str, default='/home/ray/AG_Finetune/data/resources/hg38.fa',
                    help='Path to hg38 genome FASTA. If omitted, uses remote GCS URL from AlphaGenome.')
parser.add_argument('--windows-bed', type=str,
                    default='/home/ray/AG_Finetune/exports/peak_safe_windows/D1MSN_H3K27ac_merged_peak_safe_windows.bed',
                    help='Path to BED file with training windows (safe windows recommended)')

args = parser.parse_args()

LEARNING_RATE = args.learning_rate
NUM_TRAIN_STEPS = args.num_steps
MODEL_VERSION = dna_model.ModelVersion.FOLD_0
SEQUENCE_LENGTH = args.sequence_length
BATCH_SIZE = args.batch_size
ORGANISM = dna_model.Organism.HOMO_SAPIENS
SAVE_CHECKPOINT_DIR = os.path.abspath(args.checkpoint_dir)
PRETRAINED_CHECKPOINT = args.pretrained_checkpoint
FASTA_PATH = args.fasta
WINDOWS_BED = args.windows_bed

# 4. Track Metadata
# Define which genomic tracks to finetune on. Each track requires:
# - `name`: Human-readable name.
# - `output_type`: Output type of assay (e.g., `RNA_SEQ`, `DNASE`, `CHIP_TF`, `ATAC`). One of `dna_model.OutputType`.
# - `strand`: Strand orientation (`+`, `-`, or `.` for unstranded)
# - `nonzero_mean`: Optional mean of non-zero values (used for normalization).
# - `file_path`: Path to the BigWig file containing the track data.
# Human ENCODE tracks (RNA-seq +/- strands + DNase-seq)
# BigWig files from: https://storage.googleapis.com/alphagenome/reference/encode/hg38/
# These are the same tracks used in the official DeepMind finetuning notebook.
TRACK_METADATA = pd.DataFrame(
    data=[
        [
            'RNA_SEQ',
            'UBERON:0000948 total RNA-seq',
            '+',
            '/home/ray/AG_Finetune/data/encode_hg38/ENCFF018EZY.bigWig',
        ],
        [
            'RNA_SEQ',
            'UBERON:0000948 total RNA-seq',
            '-',
            '/home/ray/AG_Finetune/data/encode_hg38/ENCFF904TSK.bigWig',
        ],
        [
            'DNASE',
            'EFO:0005337 DNase-seq',
            '.',
            '/home/ray/AG_Finetune/data/encode_hg38/ENCFF218CLQ.bigWig',
        ],
    ],
    columns=['output_type', 'name', 'strand', 'file_path'],
)
TRACK_METADATA

# Build Output Metadata
## Convert the track DataFrame into an `AlphaGenomeOutputMetadata` object that configures the model's output heads.
def build_output_metadata(
    track_metadata: pd.DataFrame,
) -> metadata_lib.AlphaGenomeOutputMetadata:
  """Builds AlphaGenomeOutputMetadata from the track metadata DataFrame.

  Args:
    track_metadata: A pandas DataFrame containing metadata for the tracks,
      including 'output_type', 'name', 'strand', and 'file_path'.

  Returns:
    A dict mapping organism to AlphaGenomeOutputMetadata.
  """
  required_cols = {'file_path', 'name', 'output_type', 'strand'}
  if not required_cols.issubset(track_metadata.columns):
    raise ValueError(
        'track_metadata must have columns %s. Missing: %s.',
        required_cols,
        required_cols - set(track_metadata.columns),
    )
  metadata = {}
  for output_type, df_group in track_metadata.groupby('output_type'):
    try:
      output_type = dna_model.OutputType[str(output_type)]
    except KeyError as e:
      raise ValueError(f'Unknown output_type: {output_type}') from e
    metadata[output_type.name.lower()] = df_group
  return metadata_lib.AlphaGenomeOutputMetadata(**metadata)


output_metadata = {
    dna_model.Organism.HOMO_SAPIENS: build_output_metadata(TRACK_METADATA)
}

# 5. Data Pipeline
## Set up the training data iterator. This loads genomic intervals and corresponding track values from the specified BigWig files.
print(f"\n{'='*60}")
print(f"DATA PIPELINE SETUP")
print(f"{'='*60}")
print(f"FASTA: {FASTA_PATH}")
print(f"Training windows: {WINDOWS_BED}")
print(f"Sequence length: {SEQUENCE_LENGTH} bp (2^{int(np.log2(SEQUENCE_LENGTH))})")
print(f"Batch size: {BATCH_SIZE} per device ({BATCH_SIZE * jax.local_device_count()} total)")
print(f"Organism: {ORGANISM.name}")

# NOTE: The DeepMind colab uses fold_intervals.Subset.TRAIN which may not align
# with our safe windows. If you need to use custom intervals from a BED file,
# you'll need to modify alphagenome_research.finetuning.dataset.get_dataset_iterator
# or implement a custom data loader.
print(f"\n⚠️  WARNING: Using AlphaGenome's default fold intervals.")
print(f"    Custom BED file ({WINDOWS_BED}) is specified but not yet integrated.")
print(f"    TODO: Implement custom interval loading to use safe windows.\n")

ds_iter_kwargs = dict(
    batch_size=BATCH_SIZE * jax.local_device_count(),
    sequence_length=SEQUENCE_LENGTH,
    output_metadata=output_metadata[ORGANISM],
    organism=ORGANISM,
    model_version=MODEL_VERSION,
    subset=fold_intervals.Subset.TRAIN,
)
if args.fasta is not None:
    ds_iter_kwargs['fasta_path'] = args.fasta
ds_iter = finetune.get_dataset_iterator(**ds_iter_kwargs)

batch = next(ds_iter)
pprint.pprint(jax.tree.map(np.shape, batch))

# 6. Model Initialization
## Load the pretrained AlphaGenome checkpoint and initialize new output heads for the finetuning tracks.
print(f"\n{'='*60}")
print(f"MODEL INITIALIZATION")
print(f"{'='*60}")
print(f"Loading pretrained checkpoint: {PRETRAINED_CHECKPOINT}")

checkpointer = ocp.StandardCheckpointer()
try:
    params_base, state_base = checkpointer.restore(PRETRAINED_CHECKPOINT)
    print("✓ Pretrained checkpoint loaded successfully")
except Exception as e:
    print(f"✗ Failed to load checkpoint: {e}")
    sys.exit(1)

# Set Up Device Mesh for Data Parallelism
num_devices = jax.local_device_count()
devices = mesh_utils.create_device_mesh((num_devices,))
mesh = Mesh(devices, axis_names=('data',))
data_sharding = P('data')
replicated_sharding = P()

# Initialize New Output Heads
## Create the forward function configured for our finetuning tracks and initialize the new head parameters.
forward_fn = finetune.get_forward_fn(
    output_metadata, jmp_policy='params=float32,compute=float32,output=float32'
)
with jax.set_mesh(mesh):
  batch = jax.device_put(batch, data_sharding)
  params_ft, state_ft = jax.jit(
      forward_fn.init,
      in_shardings=(replicated_sharding, data_sharding),
      out_shardings=replicated_sharding,
  )(jax.random.PRNGKey(0), batch)

# Merge Pretrained Trunk with New Heads
## Perform weight surgery: keep the pretrained trunk parameters and replace the head parameters with the newly initialized ones.
params_ft_head = hk.data_structures.filter(
    lambda module_name, *_: 'head' in module_name, params_ft
)
params_base_no_head = hk.data_structures.filter(
    lambda module_name, *_: 'head' not in module_name, params_base
)
params = hk.data_structures.merge(params_base_no_head, params_ft_head)
state = state_base
optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adam(LEARNING_RATE),
)
opt_state = optimizer.init(params)
train_step = jax.jit(
    finetune.get_train_step(forward_fn.apply, optimizer),
    in_shardings=(
        replicated_sharding,
        replicated_sharding,
        replicated_sharding,
        data_sharding,
    ),
    out_shardings=(
        replicated_sharding,
        replicated_sharding,
        replicated_sharding,
        replicated_sharding,
    ),
)

# 7. Training Loop
## Set up checkpointing and run the finetuning training loop.
print(f"\n{'='*60}")
print(f"TRAINING LOOP")
print(f"{'='*60}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Training steps: {NUM_TRAIN_STEPS}")
print(f"Devices: {jax.local_device_count()}")

# Configure Checkpoint Directory
_ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
checkpoint_dir = epath.Path(SAVE_CHECKPOINT_DIR) / _ts
checkpoint_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir = str(checkpoint_dir)
print(f"Checkpoint directory: {checkpoint_dir}")

# Set up metrics logging
metrics_csv = os.path.join(checkpoint_dir, 'metrics.csv')
with open(metrics_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['timestamp', 'step', 'loss'])
    writer.writeheader()

checkpointer = ocp.StandardCheckpointer()

def save(weights, idx):
  ckpt_path = os.path.join(checkpoint_dir, 'checkpoint_{:05d}'.format(idx))
  print(f'Saving checkpoint to {ckpt_path}')
  checkpointer.save(ckpt_path, weights)
  checkpointer.wait_until_finished()
  return ckpt_path

# Run Training
print(f"\nStarting training...\n")
loss_history = []
step = 0

try:
    for step in range(NUM_TRAIN_STEPS):
        try:
            batch = next(ds_iter)
        except StopIteration:
            print(f'\nDataset exhausted at step {step}')
            break
        
        with jax.set_mesh(mesh):
            batch = jax.device_put(batch, data_sharding)
            params, state, opt_state, scalars = train_step(
                params, state, opt_state, batch
            )
        
        loss_val = float(scalars['loss'])
        loss_history.append(loss_val)
        
        # Log metrics
        with open(metrics_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'step', 'loss'])
            writer.writerow({
                'timestamp': datetime.now().isoformat(),
                'step': step + 1,
                'loss': loss_val
            })
        
        # Print progress
        if step == 0 or (step + 1) % 10 == 0:
            print(f'Step {step + 1}/{NUM_TRAIN_STEPS}: loss = {loss_val:.6f}')
        
        # Save checkpoints every 100 steps
        if (step + 1) % 100 == 0:
            save((params, state), step + 1)
    
    # Final checkpoint
    ckpt_path = save((params, state), step + 1)
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total steps: {step + 1}")
    print(f"Final loss: {loss_history[-1]:.6f}")
    print(f"Final checkpoint: {ckpt_path}")
    print(f"Metrics saved to: {metrics_csv}")

except KeyboardInterrupt:
    print(f"\n\nTraining interrupted at step {step}")
    ckpt_path = save((params, state), step + 1)
    print(f"Saved checkpoint: {ckpt_path}")
    sys.exit(0)

# RUN IN A NOTEBOOK SESSION
# # 8. Inference with Finetuned Model
# ## Load the finetuned checkpoint into a `DnaModel` for inference and compare predictions against ground truth.
# # Load default organism settings but overwrite with fine-tuned output metadata.
# default_settings_human = dna_model.default_organism_settings()[
#     dna_model.Organism.HOMO_SAPIENS
# ]
# settings_human_finetune = dataclasses.replace(
#     default_settings_human,
#     metadata=output_metadata[dna_model.Organism.HOMO_SAPIENS],
# )
# model = dna_model.create(
#     ckpt_path,
#     organism_settings={
#         dna_model.Organism.HOMO_SAPIENS: settings_human_finetune
#     },
# )

# # Select Test Interval
# interval = genome.Interval(
#     chromosome='chr21', start=46125238, end=46126738
# ).resize(SEQUENCE_LENGTH)

# # Generate Predictions
# preds = model.predict_interval(
#     interval,
#     requested_outputs=[dna_model.OutputType.RNA_SEQ],
#     ontology_terms=None,
# )

# # Load Ground Truth Tracks
# true_tracks = dataset_lib.MultiTrackExtractor(
#     output_metadata[ORGANISM], sequence_length=SEQUENCE_LENGTH
# ).extract(interval)

# # Visualize Predictions vs Ground Truth
# def compact_dict(**kwargs):
#   return {k: v for k, v in kwargs.items() if v is not None}


# def plot(*, interval, predictions, targets=None):
#   if targets is None:
#     colors = {'pred': 'black'}
#   else:
#     colors = {'pred': 'black', 'true': 'red'}
#   fig = plot_components.plot(
#       [
#           plot_components.OverlaidTracks(
#               tdata=compact_dict(
#                   pred=predictions.rna_seq,
#                   true=dataclasses.replace(
#                       predictions.rna_seq,
#                       values=targets['rna_seq'].astype(np.float32),
#                   )
#                   if targets is not None
#                   else None,
#               ),
#               colors=colors,
#           ),
#       ],
#       interval=interval.resize(int(2**11)),
#   )
#   return fig

# _ = plot(predictions=preds, interval=interval, targets=true_tracks)