# This command will launch the training on all available GPUs
torchrun --nproc_per_node=4 train_siglip2.py \
  --preprocessed_data_path='/home/jupyter/data/multi_20250526/preprocessed/' \
  --fast_features_path='/home/jupyter/data/multi_20250526/fast_features/' \
  --siglip_metadata_path='/home/jupyter/ml-search-rr/models/01_experiments/artifacts/_metadata_embed_selection_filtered.parquet' \
  --siglip_embeddings_path='/home/jupyter/ml-search-rr/models/01_experiments/artifacts/google_siglip2-base-patch16-384_embeddings_RAW.filtered.npy' \
  --wandb_name='siglip2-finetune-10M-items'





  python make_siglip2_filtered_fast_features.py \
  --preprocessed_dir "/home/jupyter/data/multi_20250526/preprocessed" \
  --fast_features_dir "/home/jupyter/data/multi_20250526/fast_features" \
  --siglip_meta_parquet "/home/jupyter/ml-search-rr/models/01_experiments/artifacts/_metadata_embed_selection_filtered.parquet" \
  --siglip_embeddings_npy "/home/jupyter/ml-search-rr/models/01_experiments/artifacts/google_siglip2-base-patch16-384_embeddings_RAW.filtered.npy" \
  --out_suffix "siglip2_filtered"


  python make_siglip2_filtered_dataset.py \
  --preprocessed_dir /home/jupyter/data/multi_20250526/preprocessed \
  --fast_features_dir /home/jupyter/data/multi_20250526/fast_features \
  --siglip_meta_parquet /home/jupyter/ml-search-rr/models/01_experiments/artifacts/_metadata_embed_selection_filtered.parquet \
  --siglip_embeddings_npy /home/jupyter/ml-search-rr/models/01_experiments/artifacts/google_siglip2-base-patch16-384_embeddings_RAW.filtered.npy \
  --out_suffix siglip2_filtered \
  --also_copy_sales_into_fast_features


python train_siglip2.py \
  --raw_data_path="/home/jupyter/data/multi_20250526/preprocessed" \
  --preprocessed_feature_path="/home/jupyter/data/multi_20250526/siglip2_run" \
  --wandb_name="siglip2-smoke-7k" \
  --max_query_length=64 \
  --overfit_batches=5 \
  --max_epochs=1 \
  --num_negatives=5000 \
  --learning_rate=1e-4 \
  --weight_decay=0.05




  torchrun --nproc_per_node=4 train_siglip2.py \
    --raw_data_path="/home/jupyter/data/multi_20250526/preprocessed" \
    --preprocessed_feature_path="/home/jupyter/data/multi_20250526/fast_features_siglip2_filtered" \
    --wandb_name="siglip2-oldregime-5k" \
    --max_query_length=64 \
    --accumulate_grad_batches=1 \
    --num_negatives=5000 \
    --learning_rate=1e-4 \
    --weight_decay=0.05 \
    --use_wandb=False

torchrun --nproc_per_node=4 train_siglip2.py \
  --raw_data_path="/home/jupyter/data/multi_20250526/preprocessed" \
  --preprocessed_feature_path="/home/jupyter/data/multi_20250526/fast_features_siglip2_filtered" \
  --wandb_name="siglip2-oldregime-3k" \
  --max_query_length=64 \
  --batch_size=4096 \
  --accumulate_grad_batches=2 \
  --num_negatives=5000 \
  --learning_rate=1e-4 \
  --weight_decay=0.05 \
  --use_wandb=False


torchrun --nproc_per_node=4 train_siglip2.py \
  --raw_data_path="/home/jupyter/data/multi_20250526/preprocessed/siglip2_filtered" \
  --preprocessed_feature_path="/home/jupyter/data/multi_20250526/fast_features_siglip2_filtered" \
  --wandb_name="siglip2-oldregime-3k" \
  --max_query_length=64 \
  --batch_size=4096 \
  --accumulate_grad_batches=2 \
  --num_negatives=5000 \
  --learning_rate=1e-4 \
  --weight_decay=0.05 \
  --use_wandb=False


  its training slower and loss doestn decrease. maybe something wrohn wioth hyperparatmers or its has issues vs old training. here is siglip:

# train_siglip2_like_clip.py
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import math
import os

import fire
import numpy as np
import pyarrow.feather as feather
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import wandb

from multi_retrieval_new.models.two_tower_model_refactored import TwoTower, ModelConfig
from multi_retrieval_new.data.data_module_refactored import RetrievalDataModule
from multi_retrieval_new.data.item_feature_store import UnifiedFeatureStore
from multi_retrieval_new.models.item_model_refactored_onnx import ItemModelWithImages
from multi_retrieval_new.models.query_siglip_model import QueryModelSigLIP, create_siglip_projection_head

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    # Paths (point to your *filtered* dirs)
    raw_data_path: str                         # where preprocessor.json lives (used for cat cardinalities if available)
    preprocessed_feature_path: str             # contains train/ & val/ feature dirs + sales feather files

    # Experiment tracking
    wandb_project: str = "retrieval-experiments"
    wandb_name: Optional[str] = None

    # Training hyperparameters
    seed: int = 42
    max_epochs: int = 200
    batch_size: int = 1024 * 6
    # learning_rate: float = 1e-3
    # weight_decay: float = 1e-3

    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    betas: tuple = (0.9, 0.98)
    eps: float = 1e-8
    # max_query_length: int = 64
    num_negatives: int = 5000   # in-batch only
    
    
    # Data loading
    num_workers: int = 16
    prefetch_factor: int = 4
    
    use_item_title: bool = False               # keep off for pure image+cats
    temperature_mode: str = "learnable"
    init_tau: float = 0.07
    extra_metrics: bool = False                 # enable R@1/5, MRR/MAP on val

    # SigLIP2 specifics
    query_model_name: str = "google/siglip2-base-patch16-384"
    image_embedding_dim: int = 768             # must match primary_photo_embedding.mmap
    final_embedding_dim: int = 256
    max_query_length: int = 64

    # Trainer / perf knobs
    overfit_batches: float = 0.0
    accumulate_grad_batches: int = 1
    compile_model: bool = False                # torch.compile for small speedups
    use_wandb: bool = False
    devices: int | str = "auto"  # or default to 4 if you prefer


# def _prefer_sales(preprocessed_feature_path: Path):
#     """Prefer enriched sales if present; fallback to raw."""
#     tr_enr = preprocessed_feature_path / "train_sales_enriched.feather"
#     va_enr = preprocessed_feature_path / "val_sales_enriched.feather"
#     tr = tr_enr if tr_enr.exists() else (preprocessed_feature_path / "train_sales.feather")
#     va = va_enr if va_enr.exists() else (preprocessed_feature_path / "val_sales.feather")
#     if not tr.exists() or not va.exists():
#         raise FileNotFoundError(f"Sales feathers not found under {preprocessed_feature_path}")
#     return tr, va


def _prefer_sales(root: Path):
    tr_enr = root / "train_sales_enriched.feather"
    va_enr = root / "val_sales_enriched.feather"
    tr = tr_enr if tr_enr.exists() else (root / "train_sales.feather")
    va = va_enr if va_enr.exists() else (root / "val_sales.feather")
    if not tr.exists() or not va.exists():
        raise FileNotFoundError(f"Sales feathers not found under {root}")
    return tr, va


def create_projection_head(input_dim=768, output_dim=256):
    """2-layer MLP projection head (GELU + LayerNorm)."""
    head = nn.Sequential(
        nn.Linear(input_dim, input_dim, bias=False),
        nn.GELU(),
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, output_dim, bias=False),
        nn.LayerNorm(output_dim),
    )
    nn.init.kaiming_uniform_(head[0].weight, a=math.sqrt(5))
    nn.init.kaiming_uniform_(head[3].weight, a=math.sqrt(5))
    return head


def train(config: TrainingConfig):
    # ===== Setup =====
    pl.seed_everything(config.seed, workers=True)

    # Fast matmul kernels (TF32) + precision hint
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    raw_data_path = Path(config.raw_data_path)
    preprocessed_feature_path = Path(config.preprocessed_feature_path)
    wandb_name = config.wandb_name or f"siglip2_run_{datetime.now().strftime('%Y%m%d_%H%M:%S')}"

    # wandb.init(project=config.wandb_project, name=wandb_name, config=asdict(config), mode=os.getenv("WANDB_MODE", "online"))

    # # ===== Data: sales =====
    # train_sales_path, val_sales_path = _prefer_sales(preprocessed_feature_path)
    # train_sales = feather.read_table(train_sales_path)
    # val_sales   = feather.read_table(val_sales_path)

    # prefer the preprocessed/siglip2_filtered folder for sales
    train_sales_path, val_sales_path = _prefer_sales(Path(config.raw_data_path))
    train_sales = feather.read_table(train_sales_path)
    val_sales   = feather.read_table(val_sales_path)


    # ===== Feature stores (already filtered; contain SigLIP2 image embeddings) =====
    log.info("Initializing fast feature stores from memory-mapped files...")
    train_item_store = UnifiedFeatureStore(preprocessed_feature_path / "train")
    val_item_store = UnifiedFeatureStore(preprocessed_feature_path / "val")

    # ==== Enrich sales with pos_iloc if missing ====
    def _ensure_pos_iloc(sales_pa, store):
        if "pos_iloc" in sales_pa.column_names:
            return sales_pa  # already enriched
        if "item_id" not in sales_pa.column_names:
            raise KeyError("Sales table must contain 'item_id' or 'pos_iloc'.")
        # Map item_id -> iloc using store._pos_map
        ids = sales_pa["item_id"].to_numpy()
        pos_ilocs = np.array([store._pos_map[int(i)] for i in ids], dtype=np.int64)
        import pyarrow as pa
        return sales_pa.append_column("pos_iloc", pa.array(pos_ilocs))

    train_sales = _ensure_pos_iloc(train_sales, train_item_store)
    val_sales   = _ensure_pos_iloc(val_sales,   val_item_store)

    # Sanity: ensure SigLIP2 dim matches config
    img = train_item_store.features.get("primary_photo_embedding")
    if img is None:
        raise KeyError("primary_photo_embedding not found in feature store (expected SigLIP2 vectors).")
    if img.ndim != 2 or int(img.shape[1]) != int(config.image_embedding_dim):
        raise AssertionError(f"primary_photo_embedding shape {img.shape} incompatible with image_embedding_dim={config.image_embedding_dim}")

    # ===== Query model (SigLIP2) =====
    query_text_head = create_projection_head(input_dim=config.image_embedding_dim, output_dim=config.final_embedding_dim)
    image_head      = create_projection_head(input_dim=config.image_embedding_dim, output_dim=config.final_embedding_dim)

    query_model = QueryModelSigLIP(
        model_name=config.query_model_name,
        projection_head=query_text_head,
        freeze_base=True,   # start frozen; you can add partial unfreeze later if needed
    )
    # Explicit tokenizer truncation
    query_model.tokenizer.model_max_length = config.max_query_length

    # # ===== DataModule =====
    # dm = RetrievalDataModule(
    #     batch_size=config.batch_size,
    #     num_workers=config.num_workers,
    #     prefetch_factor=config.prefetch_factor,
    #     train_queries=train_sales,
    #     val_queries=val_sales,
    #     train_items=train_item_store,
    #     val_items=val_item_store,
    #     tokenizer=query_model.tokenizer,         # IMPORTANT: use SigLIP2 tokenizer
    #     pin_memory=True,
    #     persistent_workers=config.num_workers > 0,
    #     if_use_item_text=config.use_item_title,  # typically False here
    # )

    dm = RetrievalDataModule(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        train_queries=train_sales,
        val_queries=val_sales,
        train_items=train_item_store,
        val_items=val_item_store,
        tokenizer=query_model.tokenizer,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
        if_use_item_text=config.use_item_title,
        num_negatives=config.num_negatives,  # ← important
    )
    dm.tokenizer.model_max_length = config.max_query_length


    # ===== Item tower (categoricals + image; text off by default) =====
    # Prefer cardinalities from Preprocessor, fallback to inferring from store
    try:
        from multi_retrieval_new.preprocessing.preprocessing import Preprocessor
        preprocessor = Preprocessor.load(raw_data_path / "preprocessor.json")
        cat_features = preprocessor.feature_cardinalities
    except Exception:
        exclude = {"primary_photo_embedding", "query_input_ids", "query_attention_mask",
                   "item_title_input_ids", "item_title_attention_mask"}
        id_cols = [n for n in train_item_store.features.keys() if n.endswith("_id") and n not in exclude]
        cat_features = {n: int(train_item_store.features[n].max()) + 1 for n in id_cols}
        log.warning("Preprocessor not found; inferred categorical cardinalities from feature store.")

    item_text_head = query_text_head if config.use_item_title else None
    item_model = ItemModelWithImages(
        cat_features=cat_features,
        embed_dim=config.final_embedding_dim,
        use_images=True,
        use_text=config.use_item_title,
        image_projection_head=image_head,
        shared_text_projection_head=item_text_head,
    )

    # ===== TwoTower model =====
    model_cfg = ModelConfig(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        temperature_mode=config.temperature_mode,
        init_tau=config.init_tau,
        extra_metrics=config.extra_metrics,      # enable extra retrieval metrics
        betas=config.betas,   # <—
        eps=config.eps,       # <—
    )
    if config.use_item_title and hasattr(model_cfg, "separate_weight_decay"):
        # no special changes needed; kept for parity with your old toggles
        pass

    model = TwoTower(query_model=query_model, item_model=item_model, config=model_cfg)

    # Optional compile (PyTorch 2.0+)
    if config.compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="max-autotune")
            log.info("Compiled model with torch.compile.")
        except Exception as e:
            log.warning(f"torch.compile failed: {e}")

    # ===== Logging & callbacks (improved) =====
    MONITOR_METRIC = "val/loss"
    MONITOR_MODE   = "min"

    # run_log_dir = Path("logs") / wandb_name
    run_log_dir = Path("logs") / wandb_name
    loggers = [TensorBoardLogger(save_dir=run_log_dir.parent, name=run_log_dir.name)]

    if config.use_wandb:
        # This does the init internally, safely across ranks
        os.environ.setdefault("WANDB_START_METHOD", "thread")  # avoids socket issues on fork
        loggers.append(WandbLogger(project=config.wandb_project, name=wandb_name, log_model="all"))

    # loggers = [
    #     TensorBoardLogger(save_dir=run_log_dir.parent, name=run_log_dir.name),
    #     WandbLogger(log_model="all"),
    # ]
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=run_log_dir / "checkpoints",
            monitor=MONITOR_METRIC,
            mode=MONITOR_MODE,
            filename=f"{{epoch}}-{{{MONITOR_METRIC}:.2f}}",
            save_top_k=1,
            save_last=True,                     # NEW: keep last
        ),
        EarlyStopping(
            monitor=MONITOR_METRIC,
            patience=5,
            verbose=True,
            mode=MONITOR_MODE,
        ),
    ]
    # precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"

    # ===== Trainer (DDP + bf16 if supported) =====
    trainer = pl.Trainer(
        accelerator="gpu",
        # devices="auto",
        strategy=DDPStrategy(find_unused_parameters=False),   # robust DDP
        max_epochs=config.max_epochs,
        precision="bf16-mixed" if use_bf16 else "16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=config.accumulate_grad_batches,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=50,
        check_val_every_n_epoch=1,   # ✅ run validation once per epoch
        # val_check_interval=1,                              # validate 4x/epoch (step-based)
        overfit_batches=config.overfit_batches,
        devices=config.devices,

    )

    # ===== Train =====
    log.info("Starting main training...")
    trainer.fit(model, dm)

    if trainer.global_rank == 0:
        log.info(f"Best model checkpoint at: {trainer.checkpoint_callback.best_model_path}")

    # wandb.finish()
    log.info("Training finished successfully.")


def main(**kwargs):
    cfg = TrainingConfig(**kwargs)
    train(cfg)


if __name__ == "__main__":
    fire.Fire(main)

    
    
# torchrun --nproc_per_node=4 train_siglip2_like_clip.py \
#   --raw_data_path="/home/jupyter/data/multi_20250526/preprocessed" \
#   --preprocessed_feature_path="/home/jupyter/data/multi_20250526/siglip2_run" \
#   --wandb_name="siglip2-likeclip-ddp" \
#   --max_query_length=64 \
#   --accumulate_grad_batches=1 \
#   --compile_model=False


# Smoke test first:

# python train_siglip2_like_clip.py \
#   --raw_data_path="/home/jupyter/data/multi_20250526/preprocessed" \
#   --preprocessed_feature_path="/home/jupyter/data/multi_20250526/siglip2_run" \
#   --wandb_name="siglip2-smoke" \
#   --max_query_length=64 \
#   --overfit_batches=5 \
#   --max_epochs=1


# torchrun --nproc_per_node=4 train_siglip2_like_clip.py \
#   --raw_data_path="/home/jupyter/data/multi_20250526/preprocessed" \
#   --preprocessed_feature_path="/home/jupyter/data/multi_20250526/siglip2_run" \
#   --wandb_name="siglip2-oldregime-7k" \
#   --max_query_length=64 \
#   --accumulate_grad_batches=1 \
#   --compile_model=False \
#   --num_negatives=7000 \
#   --learning_rate=1e-4 \
#   --weight_decay=0.05


# python train_siglip2_like_clip.py \
#   --raw_data_path="/home/jupyter/data/multi_20250526/preprocessed" \
#   --preprocessed_feature_path="/home/jupyter/data/multi_20250526/siglip2_run" \
#   --wandb_name="siglip2-smoke-7k" \
#   --max_query_length=64 \
#   --overfit_batches=5 \
#   --max_epochs=1 \
#   --num_negatives=7000 \
#   --learning_rate=1e-4 \
#   --weight_decay=0.05


here is older clip:


