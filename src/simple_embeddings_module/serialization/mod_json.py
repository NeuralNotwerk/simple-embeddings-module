"""
Standard json-based serialization module for PyTorch tensors
Uses tensor.tolist() for maximum security
"""
import json
import time
from pathlib import Path

import torch


def serialize_tensors_json(tensors, output_path):
    """Serialize tensors using standard json with tolist()"""
    start_time = time.time()
    tensor_data = []
    for i, tensor in enumerate(tensors):
        tensor_data.append(
            {
                "id": i,
                "data": tensor.cpu().tolist(),  # Convert to Python lists
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
            }
        )
    data = {"tensors": tensor_data, "count": len(tensors), "method": "json_tolist"}
    with open(output_path, "w") as f:
        json.dump(data, f)
    serialize_time = time.time() - start_time
    file_size = Path(output_path).stat().st_size
    return {"serialize_time": serialize_time, "file_size": file_size, "method": "json"}
def deserialize_tensors_json(input_path):
    """Deserialize tensors from json file"""
    start_time = time.time()
    with open(input_path, "r") as f:
        data = json.load(f)
    tensors = []
    for tensor_info in data["tensors"]:
        # Direct conversion from Python list to tensor
        tensor = torch.tensor(tensor_info["data"], dtype=torch.float32)
        tensors.append(tensor)
    deserialize_time = time.time() - start_time
    return tensors, {"deserialize_time": deserialize_time, "method": "json"}
def benchmark_json(tensors, output_path):
    """Full benchmark of json serialization"""
    print(f"Benchmarking standard json with {len(tensors)} tensors...")
    # Serialize
    save_stats = serialize_tensors_json(tensors, output_path)
    # Deserialize
    loaded_tensors, load_stats = deserialize_tensors_json(output_path)
    # Verify integrity
    integrity_check = all(
        torch.allclose(original, loaded, rtol=1e-5) for original, loaded in zip(tensors, loaded_tensors)
    )
    return {
        "method": "json",
        "serialize_time": save_stats["serialize_time"],
        "deserialize_time": load_stats["deserialize_time"],
        "file_size": save_stats["file_size"],
        "integrity_check": integrity_check,
        "mb_per_second_write": (save_stats["file_size"] / 1024 / 1024) / save_stats["serialize_time"],
        "mb_per_second_read": (save_stats["file_size"] / 1024 / 1024) / load_stats["deserialize_time"],
    }
