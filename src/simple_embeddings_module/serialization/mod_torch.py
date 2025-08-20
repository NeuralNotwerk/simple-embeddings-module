"""
PyTorch native serialization (pickle-based - for comparison only)
WARNING: Uses pickle - not secure!
"""

import time
import warnings
from pathlib import Path

import torch


def serialize_tensors_torch(tensors, output_path):
    """Serialize tensors using torch.save (pickle-based)"""
    warnings.warn("Using pickle-based serialization - NOT SECURE!", UserWarning)

    start_time = time.time()

    data = {"tensors": tensors, "count": len(tensors), "method": "torch_pickle"}

    torch.save(data, output_path)

    serialize_time = time.time() - start_time
    file_size = Path(output_path).stat().st_size

    return {
        "serialize_time": serialize_time,
        "file_size": file_size,
        "method": "torch_native",
    }


def deserialize_tensors_torch(input_path):
    """Deserialize tensors using torch.load (pickle-based)"""
    start_time = time.time()

    data = torch.load(input_path, map_location="cpu")
    tensors = data["tensors"]

    deserialize_time = time.time() - start_time

    return tensors, {"deserialize_time": deserialize_time, "method": "torch_native"}


def benchmark_torch(tensors, output_path):
    """Full benchmark of PyTorch native serialization"""
    print(f"⚠️  Benchmarking PyTorch native (PICKLE) with {len(tensors)} tensors...")

    # Serialize
    save_stats = serialize_tensors_torch(tensors, output_path)

    # Deserialize
    loaded_tensors, load_stats = deserialize_tensors_torch(output_path)

    # Verify integrity
    integrity_check = all(
        torch.allclose(original, loaded, rtol=1e-5)
        for original, loaded in zip(tensors, loaded_tensors)
    )

    return {
        "method": "torch_native (PICKLE)",
        "serialize_time": save_stats["serialize_time"],
        "deserialize_time": load_stats["deserialize_time"],
        "file_size": save_stats["file_size"],
        "integrity_check": integrity_check,
        "mb_per_second_write": (save_stats["file_size"] / 1024 / 1024)
        / save_stats["serialize_time"],
        "mb_per_second_read": (save_stats["file_size"] / 1024 / 1024)
        / load_stats["deserialize_time"],
    }
