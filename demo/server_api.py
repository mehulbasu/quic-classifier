#!/usr/bin/env python3
# uvicorn demo.server_api:app --host 0.0.0.0 --port 8000 &> demo/server.log
"""FastAPI inference server for the HybridCNN QUIC classifier."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import sys

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_pytorch import HybridCNN

LOGGER = logging.getLogger("server_api")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [server_api] %(message)s")

MODEL_PATH = Path("artifacts/cnn_ddp/checkpoint_best.pt")
META_PATH = Path("datasets/cache_pytorch/train/meta.json")


def _load_meta(meta_path: Path) -> Dict[str, Any]:
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing training metadata: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_model(checkpoint_path: Path, meta: Dict[str, Any], device: torch.device) -> HybridCNN:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get("args")
    if saved_args is None:
        raise ValueError("Checkpoint is missing stored training args; please retrain with the chunked trainer.")
    model_args = SimpleNamespace(
        seq_hidden=saved_args["seq_hidden"],
        mlp_hidden=saved_args["mlp_hidden"],
        dropout=saved_args["dropout"],
        version_embed_dim=saved_args["version_embed_dim"],
    )
    num_versions = max(len(meta.get("version_values", [])), 1)
    model = HybridCNN(
        seq_len=int(meta["seq_len"]),
        tab_dim=int(meta["tab_dim"]),
        num_classes=int(meta["num_classes"]),
        num_versions=num_versions,
        args=model_args,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


class _ServerState:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.meta = _load_meta(META_PATH)
        self.label_names = list(self.meta.get("index_to_label", []))
        if not self.label_names:
            raise ValueError("index_to_label missing from metadata; cannot decode predictions")
        normalization = self.meta.get("normalization")
        if not normalization:
            raise ValueError("Normalization statistics missing from metadata")
        self.seq_mean = torch.tensor(normalization["seq_mean"], dtype=torch.float32, device=self.device).view(1, -1, 1)
        self.seq_std = torch.tensor(normalization["seq_std"], dtype=torch.float32, device=self.device).clamp_min_(1e-6).view(1, -1, 1)
        self.tab_mean = torch.tensor(normalization["tab_mean"], dtype=torch.float32, device=self.device).view(1, -1)
        self.tab_std = torch.tensor(normalization["tab_std"], dtype=torch.float32, device=self.device).clamp_min_(1e-6).view(1, -1)
        self.expected_tab_dim = self.tab_mean.shape[1]
        self.model = _load_model(MODEL_PATH, self.meta, self.device)

    def predict(self, payload: "PredictRequest") -> Dict[str, Any]:
        seq = torch.tensor(payload.sequences, dtype=torch.float32, device=self.device).unsqueeze(0)
        tab = torch.tensor(payload.tabular, dtype=torch.float32, device=self.device).unsqueeze(0)
        version = torch.tensor([payload.version_idx], dtype=torch.long, device=self.device)

        orig_tab_dim = tab.shape[1]
        if orig_tab_dim < self.expected_tab_dim:
            pad = torch.zeros(1, self.expected_tab_dim - orig_tab_dim, dtype=torch.float32, device=self.device)
            tab = torch.cat([tab, pad], dim=1)
            LOGGER.warning(
                "Tabular vector too short (%s < %s); padding with zeros",
                orig_tab_dim,
                self.expected_tab_dim,
            )
        elif orig_tab_dim > self.expected_tab_dim:
            LOGGER.warning(
                "Tabular vector too long (%s > %s); truncating",
                orig_tab_dim,
                self.expected_tab_dim,
            )
            tab = tab[:, : self.expected_tab_dim]

        seq = (seq - self.seq_mean) / self.seq_std
        tab = (tab - self.tab_mean) / self.tab_std

        with torch.no_grad():
            logits = self.model(seq, tab, version)
            probs = F.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

        index = pred_idx.item()
        if index >= len(self.label_names):
            raise RuntimeError(f"Predicted index {index} out of bounds for label list of size {len(self.label_names)}")
        return {
            "label": self.label_names[index],
            "confidence": float(confidence.item()),
        }


STATE = _ServerState()
app = FastAPI(title="HybridCNN QUIC Inference API", version="1.0")


class PredictRequest(BaseModel):
    sequences: List[List[float]] = Field(..., description="3x30 tensor represented as nested list")
    tabular: List[float] = Field(..., description="Tabular feature vector")
    version_idx: int

    @validator("sequences")
    def _validate_sequences(cls, value: List[List[float]]) -> List[List[float]]:
        if len(value) != 3:
            raise ValueError("'sequences' must have shape [3, 30]")
        if any(len(row) != 30 for row in value):
            raise ValueError("Each sequence channel must contain exactly 30 values")
        return value


class PredictResponse(BaseModel):
    label: str
    confidence: float


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    seq_channels = len(request.sequences)
    seq_len = len(request.sequences[0]) if request.sequences else 0
    LOGGER.info(
        "Received request seq_shape=[%s,%s] tab_dim=%s version=%s",
        seq_channels,
        seq_len,
        len(request.tabular),
        request.version_idx,
    )
    try:
        result = STATE.predict(request)
        return PredictResponse(**result)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        LOGGER.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _run_single_inference(sample_path: Path) -> None:
    with open(sample_path, "r", encoding="utf-8") as handle:
        sample_json = json.load(handle)
    payload = PredictRequest(**sample_json)
    result = STATE.predict(payload)
    print(json.dumps({"input": sample_path.name, **result}, indent=2))


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test HybridCNN inference on a sample payload")
    parser.add_argument("--sample", type=str, default="demo/golden-sample.json", help="Path to sample JSON payload")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    _run_single_inference(Path(args.sample))
