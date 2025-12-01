#!/usr/bin/env python3
"""Streamlit client that sniffs QUIC flows and calls the inference API."""
# sudo -E streamlit run demo/mac_agent.py
from __future__ import annotations

import math
import queue
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List

import requests
import streamlit as st
from scapy.all import AsyncSniffer, IP, UDP  # type: ignore

API_URL = "http://localhost:8000/predict"
FLOW_TARGET_LEN = 30
# The trained HybridCNN expects 49 tabular stats (per meta.json -> tab_dim).
TABULAR_DIM = 49
PACKET_LOG_LIMIT = 20


def _discover_local_ips() -> List[str]:
    try:
        hostname = socket.gethostname()
        _, _, addrs = socket.gethostbyname_ex(hostname)
        return sorted({"127.0.0.1", "::1", *addrs})
    except OSError:
        return ["127.0.0.1", "::1"]


LOCAL_IPS = set(_discover_local_ips())


@dataclass
class AgentState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    flow_stats: Dict["FlowKey", FlowState] = field(default_factory=dict)
    result_queue: queue.Queue = field(default_factory=queue.Queue)
    packet_counter: int = 0
    packet_logs: Deque[str] = field(default_factory=lambda: deque(maxlen=PACKET_LOG_LIMIT))
    sniffer: AsyncSniffer | None = None
    sniffer_error: str | None = None


@dataclass(frozen=True)
class FlowKey:
    src: str
    dst: str
    sport: int
    dport: int


@dataclass
class FlowState:
    sizes: List[float] = field(default_factory=list)
    dirs: List[float] = field(default_factory=list)
    ipts: List[float] = field(default_factory=list)
    last_ts: float = 0.0

    def reset(self) -> None:
        self.sizes.clear()
        self.dirs.clear()
        self.ipts.clear()
        self.last_ts = 0.0


FLOW_LOCK = threading.Lock()
FLOW_STATS: Dict[FlowKey, FlowState] = {}
RESULT_QUEUE: queue.Queue = queue.Queue()
def _get_agent_state() -> AgentState:
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = AgentState()
    return st.session_state.agent_state
PACKET_COUNTER = 0


def _direction(sign_ip: str) -> float:
    return 1.0 if sign_ip in LOCAL_IPS else -1.0


def _guess_quic_header(payload: bytes) -> str:
    if not payload:
        return "empty"
    byte = payload[0]
    if byte & 0x80:
        return "quic-long"
    if byte & 0x40:
        return "quic-short"
    return "unknown"


def _canonical_key(ip_src: str, ip_dst: str, sport: int, dport: int) -> FlowKey:
    forward = (ip_src, sport, ip_dst, dport)
    reverse = (ip_dst, dport, ip_src, sport)
    chosen = forward if forward <= reverse else reverse
    return FlowKey(*chosen)


def _record_packet_debug(state: AgentState, src: str, dst: str, sport: int, dport: int, payload: bytes) -> None:
    state.packet_counter += 1
    hint = _guess_quic_header(payload)
    head_snip = payload[:6].hex() if payload else ""
    entry = f"{time.strftime('%H:%M:%S')} {src}:{sport}->{dst}:{dport} len={len(payload)} hint={hint} head={head_snip}"
    state.packet_logs.append(entry)


def _process_packet(pkt, state: AgentState) -> None:  # pragma: no cover - executed in sniffer thread
    if IP not in pkt or UDP not in pkt:
        return
    if pkt[UDP].sport != 443 and pkt[UDP].dport != 443:
        return
    ip_layer = pkt[IP]
    udp_layer = pkt[UDP]
    payload = bytes(udp_layer.payload)
    size = float(len(payload))
    now = time.time()
    key = _canonical_key(ip_layer.src, ip_layer.dst, int(udp_layer.sport), int(udp_layer.dport))
    with state.lock:
        _record_packet_debug(state, ip_layer.src, ip_layer.dst, int(udp_layer.sport), int(udp_layer.dport), payload)
        flow = state.flow_stats.setdefault(key, FlowState())
        flow.sizes.append(math.log1p(max(size, 0.0)))
        flow.dirs.append(_direction(ip_layer.src))
        delta = 0.0 if not flow.last_ts else max(now - flow.last_ts, 0.0)
        flow.ipts.append(math.log1p(delta))
        flow.last_ts = now
        if len(flow.sizes) == FLOW_TARGET_LEN:
            _send_prediction(key, flow, state)
            flow.reset()


def _send_prediction(key: FlowKey, flow: FlowState, agent_state: AgentState) -> None:
    payload = {
        "sequences": [flow.sizes[:FLOW_TARGET_LEN], flow.dirs[:FLOW_TARGET_LEN], flow.ipts[:FLOW_TARGET_LEN]],
        "tabular": [0.0] * TABULAR_DIM,
        "sni_idx": 0,
        "ua_idx": 0,
        "version_idx": 0,
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()
        agent_state.result_queue.put(
            {
                "flow": f"{key.src}:{key.sport} -> {key.dst}:{key.dport}",
                "label": result.get("label", "unknown"),
                "confidence": f"{result.get('confidence', 0.0):.4f}",
                "timestamp": time.strftime("%H:%M:%S"),
            }
        )
    except requests.RequestException as exc:
        agent_state.result_queue.put(
            {
                "flow": f"{key.src}:{key.sport} -> {key.dst}:{key.dport}",
                "label": "API_ERROR",
                "confidence": str(exc),
                "timestamp": time.strftime("%H:%M:%S"),
            }
        )


def _ensure_sniffer(iface: str, state: AgentState) -> None:
    if state.sniffer and state.sniffer.running:
        return
    try:
        sniffer = AsyncSniffer(
            iface=iface,
            prn=lambda pkt: _process_packet(pkt, state),
            store=False,
            filter="udp port 443",
        )
        sniffer.start()
        state.sniffer = sniffer
        state.sniffer_error = None
    except Exception as exc:  # pragma: no cover - interactive failure path
        state.sniffer_error = str(exc)
        state.sniffer = None


def _stop_sniffer(state: AgentState) -> None:
    if state.sniffer and state.sniffer.running:
        state.sniffer.stop()
    state.sniffer = None


def _drain_results(state: AgentState) -> List[Dict[str, str]]:
    drained = []
    while not state.result_queue.empty():
        drained.append(state.result_queue.get())
    return drained


state = _get_agent_state()

st.set_page_config(page_title="QUIC Client", layout="wide")
st.title("QUIC Traffic Classification Client")
st.caption("Sniffs UDP/443 flows, builds features, and queries the GPU inference server.")

iface = st.text_input("Network interface", value="en0", help="macOS Wi-Fi is often en0. Use ifconfig to confirm.")
col_start, col_stop = st.columns(2)
with col_start:
    if st.button("Start Sniffing", disabled=bool(state.sniffer and state.sniffer.running)):
        _ensure_sniffer(iface, state)
with col_stop:
    if st.button("Stop Sniffing", disabled=not (state.sniffer and state.sniffer.running)):
        _stop_sniffer(state)

status = "running" if state.sniffer and state.sniffer.running else "stopped"
st.metric("Sniffer status", status)
if state.sniffer_error:
    st.error(f"Sniffer error: {state.sniffer_error}")

if "predictions" not in st.session_state:
    st.session_state.predictions = []

for entry in _drain_results(state):
    st.session_state.predictions.append(entry)
    st.session_state.predictions = st.session_state.predictions[-100:]

if st.session_state.predictions:
    st.subheader("Latest predictions (max 100)")
    st.dataframe(st.session_state.predictions[::-1], use_container_width=True)
else:
    st.info("Waiting for a flow to reach 30 packets on UDP/443…")

st.sidebar.header("Debug")
st.sidebar.json({"local_ips": sorted(LOCAL_IPS), "target": API_URL})

st.sidebar.header("Packet Debug")
st.sidebar.metric("UDP/443 packets seen", state.packet_counter)
if state.packet_logs:
    st.sidebar.write("Recent packets")
    st.sidebar.dataframe(
        [{"event": entry} for entry in reversed(list(state.packet_logs))],
        use_container_width=True,
    )
else:
    st.sidebar.write("No packet activity observed yet")

if state.sniffer and not state.sniffer.running:
    _stop_sniffer(state)
