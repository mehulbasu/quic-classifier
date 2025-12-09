import torch
import json
import numpy as np
from pathlib import Path

# Find the last cached chunk
cache_files = sorted(list(Path("datasets/cache_pytorch/val/chunks").glob("*.pt")))
if not cache_files:
    raise FileNotFoundError("No .pt files found in datasets/cache_pytorch")

cache_file = cache_files[-1]
print(f"Loading from: {cache_file}")
data = torch.load(cache_file)

# Get index 0
# FIX: Map 'tabular' -> 'statics' for the API, and include indices
seq = data["sequences"][0].numpy()      # [3, 30]
tabular = data["tabular"][0].numpy()    # [Features]
sni = data["sni_idx"][0].item()         # Integer
ua = data["ua_idx"][0].item()           # Integer
ver = data["version_idx"][0].item()     # Integer
label = data["labels"][0].item()

# Create payload matching your model's likely inputs
payload = {
    "sequences": seq.tolist(),
    "tabular": tabular.tolist(),
    "sni_idx": sni,
    "ua_idx": ua,
    "version_idx": ver,
    "ground_truth_label": label
}

with open("demo/golden_sample.json", "w") as f:
    json.dump(payload, f, indent=2)

print(f"Success! Saved golden_sample.json. Label ID: {label}")

"""
I need to build the Server and Client for my QUIC Traffic Classification demo. The demo will involve two machines: the Cloudlab 
GPU server running the model inference, and my Mac laptop running the packet capture and feature extraction.
My dataset and model use a specific set of 5 input keys. Please ensure the code handles them exactly as described below.

### Part 1: The Inference Server (`demo/server_api.py`)
Write a FastAPI script for the GPU server.
1.  **Model Loading:**
    - Initialize the `HybridCNN` model (I will provide the class later).
    - Load weights from `artifacts/cnn_ddp/checkpoint_best.pt`.
    - Move to GPU and set it to evaluate. If necessary, create a new script similar to `scripts/eval_pytorch.py`
    to handle prediction logic.
2.  **API Endpoint (`POST /predict`):**
    - **Input Schema:** The JSON body must accept:
        - `sequences`: List[List[float]] (Shape: [3, 30])
        - `tabular`: List[float] (Shape: [64])
        - `sni_idx`: int
        - `ua_idx`: int
        - `version_idx`: int
    - **Forward Pass:**
        - Convert all inputs to PyTorch tensors.
        - Add batch dimension (unsqueeze 0).
        - Pass them to the model: `model(sequences, tabular, sni_idx, ua_idx, version_idx)`
        - **Note:** If the model only needs (sequences, tabular), ignore the indices.
    - **Output:** `{"label": String, "confidence": float}`.

### Part 2: The Client Agent (`demo/mac_agent.py`)
Write the Mac client using `scapy` and `streamlit`.
1.  **Feature Extraction:**
    - Sniff UDP port 443. Group by flow.
    - When a flow hits 30 packets, extract:
        - `sequences`: [Sizes (log1p), Directions (+/-1), IPT (log1p)]. Shape [3, 30].
        - `tabular`: Create a list of zeros of length 64 (placeholder for the demo).
        - `sni_idx`, `ua_idx`, `version_idx`: Send `0` for all of these (placeholders).
2.  **API Call:**
    - Send these 5 keys to `http://localhost:8000/predict`.
3.  **Dashboard:**
    - Display the prediction results in a Streamlit table.

**Important:** The field names in the JSON payload MUST match `sequences`, `tabular`, `sni_idx`, `ua_idx`, `version_idx`.
"""