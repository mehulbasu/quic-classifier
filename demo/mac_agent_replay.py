#!/usr/bin/env python3
"""Streamlit client that replays cached QUIC traces to the FastAPI server."""
# streamlit run demo/mac_agent_replay.py
from __future__ import annotations

import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
import streamlit as st

TRACE_PATH = Path(__file__).resolve().parent / "demo_traces.json"
SERVER_URL = "http://localhost:8000/predict"
FEATURE_KEYS = ("sequences", "tabular", "sni_idx", "ua_idx", "version_idx")
MAX_HISTORY = 200


def load_traces(trace_path: Path) -> List[Dict[str, object]]:
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")
    with open(trace_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise ValueError("Trace file must contain a non-empty list")
    return data


def init_state(traces: List[Dict[str, object]]) -> None:
    if "traces" not in st.session_state:
        shuffled = traces.copy()
        random.shuffle(shuffled)
        st.session_state.traces = shuffled
    if "trace_index" not in st.session_state:
        st.session_state.trace_index = 0
    if "history" not in st.session_state:
        st.session_state.history: List[Dict[str, object]] = []
    if "running" not in st.session_state:
        st.session_state.running = False


def replay_once(replay_delay: float) -> None:
    traces: List[Dict[str, object]] = st.session_state.traces
    if not traces:
        st.warning("Trace list is empty. Upload new traces and restart.")
        st.session_state.running = False
        return

    idx = st.session_state.trace_index % len(traces)
    record = traces[idx]
    payload = {key: record[key] for key in FEATURE_KEYS}
    true_label = record.get("ground_truth", "unknown")

    started = time.perf_counter()
    try:
        response = requests.post(SERVER_URL, json=payload, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
        st.session_state.running = False
        return
    latency_ms = (time.perf_counter() - started) * 1000.0

    result = response.json()
    predicted_label = result.get("label", "unknown")
    confidence = float(result.get("confidence", 0.0))

    row = {
        "Timestamp": datetime.now().strftime("%H:%M:%S"),
        "True Label": true_label,
        "Predicted Label": predicted_label,
        "Confidence": round(confidence, 4),
        "Latency (ms)": round(latency_ms, 1),
    }
    history = st.session_state.history
    history.append(row)
    if len(history) > MAX_HISTORY:
        del history[: len(history) - MAX_HISTORY]

    st.session_state.trace_index = (idx + 1) % len(traces)
    time.sleep(replay_delay)
    st.experimental_rerun()


def render_history_table() -> None:
    history = st.session_state.history
    if not history:
        st.info("No predictions yet. Click 'Start Replay' to begin.")
        return
    df = pd.DataFrame(history)

    def highlight_row(row: pd.Series) -> List[str]:
        match = row["True Label"] == row["Predicted Label"]
        color = "background-color: #d4edda" if match else "background-color: #f8d7da"
        return [color] * len(row)

    styled = df.style.apply(highlight_row, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="QUIC Trace Replay", layout="wide")
    try:
        traces = load_traces(TRACE_PATH)
    except (OSError, ValueError) as exc:
        st.error(str(exc))
        return

    init_state(traces)

    st.sidebar.header("Replay Controls")
    replay_delay = st.sidebar.slider("Replay Speed (seconds)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

    st.title("QUIC Trace Replay Client")
    st.caption("Replays cached CESNET traces through the FastAPI inference server")

    col_start, col_stop = st.columns(2)
    if col_start.button("Start Replay", type="primary"):
        st.session_state.running = True
    if col_stop.button("Stop", type="primary"):
        st.session_state.running = False

    st.divider()
    render_history_table()

    if st.session_state.running:
        replay_once(replay_delay)


if __name__ == "__main__":
    main()
