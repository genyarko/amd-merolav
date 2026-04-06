"""Mock HIP/ROCm environment for CPU-only validation.

Patches torch.cuda and related modules so migrated code can be
executed and smoke-tested without a real AMD GPU.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from unittest.mock import MagicMock


def _build_mock_torch_cuda() -> types.ModuleType:
    """Build a mock torch.cuda module that pretends a HIP device is present."""
    mod = types.ModuleType("torch.cuda")
    mod.is_available = lambda: True
    mod.device_count = lambda: 1
    mod.current_device = lambda: 0
    mod.get_device_name = lambda device=0: "AMD Instinct MI300X (Mock)"
    mod.set_device = lambda device: None
    mod.synchronize = lambda device=None: None
    mod.empty_cache = lambda: None
    mod.memory_allocated = lambda device=None: 0
    mod.max_memory_allocated = lambda device=None: 0
    mod.memory_reserved = lambda device=None: 0
    mod.max_memory_reserved = lambda device=None: 0
    mod.reset_peak_memory_stats = lambda device=None: None
    mod.manual_seed = lambda seed: None
    mod.manual_seed_all = lambda seed: None

    # Mock Event
    class MockEvent:
        def __init__(self, enable_timing=True):
            pass
        def record(self, stream=None): pass
        def synchronize(self): pass
        def elapsed_time(self, other): return 0.0
        def wait(self, stream=None): pass

    mod.Event = MockEvent

    # Mock Stream
    class MockStream:
        def __init__(self, device=None, priority=0):
            pass
        def synchronize(self): pass
        def wait_event(self, event): pass
        def wait_stream(self, stream): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

    mod.Stream = MockStream

    # Mock amp
    amp = types.ModuleType("torch.cuda.amp")
    class MockAutocast:
        def __init__(self, enabled=True, dtype=None):
            self.enabled = enabled
        def __enter__(self): return self
        def __exit__(self, *args): pass
    amp.autocast = MockAutocast

    class MockGradScaler:
        def __init__(self, enabled=True):
            pass
        def scale(self, loss): return loss
        def step(self, optimizer): optimizer.step()
        def update(self): pass
        def get_scale(self): return 1.0
    amp.GradScaler = MockGradScaler

    mod.amp = amp

    return mod


def _build_mock_backends() -> types.ModuleType:
    """Build mock torch.backends.miopen module."""
    mod = types.ModuleType("torch.backends.miopen")
    mod.enabled = True
    mod.deterministic = False
    mod.version = lambda: "3.0.0"
    return mod


@contextmanager
def mock_hip_environment():
    """Context manager that patches sys.modules to simulate a ROCm environment.

    Usage:
        with mock_hip_environment():
            exec(compiled_code, namespace)
    """
    patches: dict[str, types.ModuleType | MagicMock] = {
        # Mock hip-python (replacement for pycuda)
        "hip": MagicMock(),
        "hip.hip": MagicMock(),
        # Mock miopen
        "miopen": MagicMock(),
        # Mock rocblas, rocfft, etc.
        "rocblas": MagicMock(),
        "rocfft": MagicMock(),
        "rocrand": MagicMock(),
        "rocsparse": MagicMock(),
        "migraphx": MagicMock(),
    }

    # Save originals and apply patches
    originals: dict[str, types.ModuleType | None] = {}
    for name, mock_mod in patches.items():
        originals[name] = sys.modules.get(name)
        sys.modules[name] = mock_mod

    # If torch is available, also patch torch.backends.miopen
    torch_patched = False
    original_miopen = None
    try:
        import torch
        if hasattr(torch, "backends"):
            original_miopen = getattr(torch.backends, "miopen", None)
            torch.backends.miopen = _build_mock_backends()
            torch_patched = True
    except ImportError:
        pass

    try:
        yield
    finally:
        # Restore originals
        for name, orig in originals.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig

        if torch_patched:
            try:
                import torch
                if original_miopen is None:
                    delattr(torch.backends, "miopen")
                else:
                    torch.backends.miopen = original_miopen
            except (ImportError, AttributeError):
                pass
