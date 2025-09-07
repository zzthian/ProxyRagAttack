#!/usr/bin/env python3
"""
Transformer Query Predictor (Multi-dataset training)
--------------------------------------------------

This version supports **multiple datasets** and trains **one model** jointly
across them. Instead of merging the JSONL files manually, you can pass multiple
`--train` and `--valid` arguments, and the training loop will alternate batches
from each dataset.

Usage:
  python transformer_query_predictor.py \
      --train data/datasetA_train.jsonl data/datasetB_train.jsonl \
      --valid data/datasetA_valid.jsonl data/datasetB_valid.jsonl \
      --emb-dim 768 --d-model 512 --layers 4 --heads 8 --epochs 10
"""
from __future__ import annotations
import argparse

import math
import os
from dataclasses import dataclass
from typing import List
from utils import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import JsonlDataset, collate_batch


# ------------------------------
# Model
# ------------------------------


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class QueryPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_causal_mask: bool = False,
        segment_embed: bool = True,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.segment_embed = nn.Embedding(2, d_model) if segment_embed else None
        self.posenc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, input_dim))
        self.use_causal_mask = use_causal_mask

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, seg_ids: torch.Tensor):
        h = self.input_proj(x)
        if self.segment_embed is not None:
            h = h + self.segment_embed(seg_ids)
        h = self.posenc(h)
        src_key_padding_mask = ~attn_mask
        T = h.size(1)
        causal = None
        if self.use_causal_mask:
            causal = torch.triu(
                torch.ones(T, T, device=h.device) * float("-inf"), diagonal=1
            )
        h = self.encoder(h, mask=causal, src_key_padding_mask=src_key_padding_mask)
        last_hidden = h[
            torch.arange(h.size(0), device=h.device), attn_mask.sum(dim=1) - 1
        ]
        return self.head(last_hidden)


# ------------------------------
# Training / Evaluation
# ------------------------------


@dataclass
class Config:
    train: List[str]
    valid: List[str] | None
    epochs: int = 10
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    emb_dim: int = 768
    d_model: int = 512
    heads: int = 8
    layers: int = 4
    ff: int = 1024
    dropout: float = 0.1
    loss: str = "cosine"
    max_seq_len: int | None = 256
    use_causal_mask: bool = False
    segment_embed: bool = True
    grad_clip: float | None = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    outdir: str = "checkpoints"


def make_dataloader(paths: list[str], cfg: Config, shuffle: bool) -> DataLoader:
    ds = JsonlDataset(paths, cfg)  # pass paths directly
    # Check embedding dimension from encoder
    assert (
        ds.encoder.get_sentence_embedding_dimension() == cfg.emb_dim
    ), f"Found emb_dim={ds.encoder.get_sentence_embedding_dimension()}, expected {cfg.emb_dim}"
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
        num_workers=2,
    )


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_n = nn.functional.normalize(pred, dim=-1)
    tgt_n = nn.functional.normalize(target, dim=-1)
    cos = (pred_n * tgt_n).sum(dim=-1)
    return (1 - cos).mean()


def evaluate(model: nn.Module, loaders: List[DataLoader], device: str, loss_fn):
    model.eval()
    total_loss, total_cos, n = 0.0, 0.0, 0
    with torch.no_grad():
        for loader in loaders:
            for batch in loader:
                x = batch["input_embs"].to(device)
                attn = batch["attn_mask"].to(device)
                seg = batch["seg_ids"].to(device)
                y = batch["labels"].to(device)
                y_hat = model(x, attn, seg)
                loss = loss_fn(y_hat, y)
                y_hat_n = nn.functional.normalize(y_hat, dim=-1)
                y_n = nn.functional.normalize(y, dim=-1)
                cos = (y_hat_n * y_n).sum(dim=-1).mean().item()
                bsz = x.size(0)
                total_loss += loss.item() * bsz
                total_cos += cos * bsz
                n += bsz
    return total_loss / n, total_cos / n


def train_loop(cfg: Config):
    set_seed()
    os.makedirs(cfg.outdir, exist_ok=True)

    train_loaders = make_dataloader(cfg.train, cfg, shuffle=True)
    valid_loaders = make_dataloader(cfg.valid, cfg, shuffle=False)

    model = QueryPredictor(
        cfg.emb_dim,
        cfg.d_model,
        cfg.heads,
        cfg.layers,
        cfg.ff,
        cfg.dropout,
        cfg.use_causal_mask,
        cfg.segment_embed,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loss_fn = cosine_loss if cfg.loss == "cosine" else nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        # alternate batches from each loader
        iters = [iter(loader) for loader in train_loaders]
        while iters:
            for i, it in enumerate(list(iters)):
                try:
                    batch = next(it)
                except StopIteration:
                    iters.remove(it)
                    continue
                x = batch["input_embs"].to(cfg.device)
                attn = batch["attn_mask"].to(cfg.device)
                seg = batch["seg_ids"].to(cfg.device)
                y = batch["labels"].to(cfg.device)
                y_hat = model(x, attn, seg)
                loss = loss_fn(y_hat, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                bsz = x.size(0)
                running += loss.item() * bsz
                seen += bsz

        train_loss = running / seen
        if valid_loaders:
            val_loss, val_cos = evaluate(model, valid_loaders, cfg.device, loss_fn)
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_cos={val_cos:.4f}"
            )
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {"model_state": model.state_dict(), "cfg": vars(cfg)},
                    os.path.join(cfg.outdir, "best.pt"),
                )
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}")
            torch.save(
                {"model_state": model.state_dict(), "cfg": vars(cfg)},
                os.path.join(cfg.outdir, f"epoch{epoch}.pt"),
            )


# ------------------------------
# CLI
# ------------------------------


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Multi-dataset Transformer training")
    p.add_argument(
        "--train", nargs="+", required=True, help="Paths to training JSONL files"
    )
    p.add_argument(
        "--valid", nargs="*", default=None, help="Paths to validation JSONL files"
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--emb-dim", type=int, default=768)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--loss", choices=["cosine", "mse"], default="cosine")
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--use-causal-mask", action="store_true")
    p.add_argument("--no-segment-embed", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--outdir", default="checkpoints")
    args = p.parse_args()
    return Config(
        train=args.train,
        valid=args.valid if args.valid else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        emb_dim=args.emb_dim,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        ff=args.ff,
        dropout=args.dropout,
        loss=args.loss,
        max_seq_len=None if args.max_seq_len <= 0 else args.max_seq_len,
        use_causal_mask=args.use_causal_mask,
        segment_embed=not args.no_segment_embed,
        grad_clip=args.grad_clip,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train_loop(cfg)
