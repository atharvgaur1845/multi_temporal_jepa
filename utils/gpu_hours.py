"""GPU-hour / time / memory accounting (infra — implemented for you).

Report these alongside every experiment so the temporal-vs-spatial comparison is honestly
'under equal compute' (Common Mistake #10) and so the full matrix's cost is transparent.
"""
from __future__ import annotations

import time

import torch


class GpuHourMeter:
    """Wall-clock timer + peak-memory tracker. Use one per run.

    Usage:
        meter = GpuHourMeter()
        meter.start()
        ... train ...
        stats = meter.stop()   # dict(gpu_hours, seconds, peak_mem_gb)
    """

    def __init__(self, device=None) -> None:
        # Track the device actually used (e.g. cuda:2); otherwise max_memory_allocated() defaults
        # to GPU 0 and reports 0.0 when you trained elsewhere.
        self._t0 = None
        self._device = device

    def start(self) -> None:
        if torch.cuda.is_available():
            # Ensure the target device's CUDA context exists before resetting its stats —
            # reset_peak_memory_stats() errors ("did you call init?") if start() runs before any
            # tensor has been placed on this device (e.g. in run_matrix, before build_model).
            if self._device is not None:
                # Defensive: a caller may still hand us a bare "cuda" (index None),
                # which set_device() rejects. Resolve it to the current index.
                d = torch.device(self._device)
                if d.type == "cuda" and d.index is None:
                    d = torch.device("cuda", torch.cuda.current_device())
                self._device = d
                torch.cuda.set_device(self._device)
                torch.zeros(1, device=self._device)  # force context init; negligible memory
            torch.cuda.reset_peak_memory_stats(self._device)
        self._t0 = time.perf_counter()

    def stop(self) -> dict:
        if self._t0 is None:
            raise RuntimeError("GpuHourMeter.stop() called before start()")
        seconds = time.perf_counter() - self._t0
        peak_mem_gb = (
            torch.cuda.max_memory_allocated(self._device) / 1e9 if torch.cuda.is_available() else 0.0
        )
        # One process on one GPU: gpu-hours == wall-clock hours. Scale by #GPUs if you go multi-GPU.
        return {"gpu_hours": seconds / 3600.0, "seconds": seconds, "peak_mem_gb": peak_mem_gb}
