"""
orjson-based serialization module for PyTorch tensors
Uses OPT_SERIALIZE_NUMPY for efficient, secure serialization
"""
import time
from pathlib import Path
from typing import Any, Dict, Optional

import orjson
import torch

from ..sem_module_reg import ConfigParameter
from .mod_serialization_base import (
    SerializationProviderBase,
    SerializationProviderError,
)


class OrjsonSerializationProvider(SerializationProviderBase):
    """Secure PyTorch tensor serialization using orjson"""
    CONFIG_PARAMETERS = [
        ConfigParameter(
            key_name="indent",
            value_type="bool",
            config_description="Whether to indent JSON output for readability",
            required=False,
            value_opt_default=True,
        ),
        ConfigParameter(
            key_name="sort_keys",
            value_type="bool",
            config_description="Whether to sort JSON keys",
            required=False,
            value_opt_default=False,
        ),
    ]
    CAPABILITIES = {
        "format_name": "orjson",
        "is_secure": True,
        "supports_compression": False,  # Handled at storage layer
        "human_readable": True,
        "cross_platform": True,
        "max_tensor_size": -1,  # No specific limit
    }
    def __init__(self, **config):
        """Initialize orjson serialization provider"""
        super().__init__(**config)
        self.indent = config.get("indent", True)
        self.sort_keys = config.get("sort_keys", False)
        # Set orjson options
        self.orjson_options = orjson.OPT_SERIALIZE_NUMPY
        if self.indent:
            self.orjson_options |= orjson.OPT_INDENT_2
        if self.sort_keys:
            self.orjson_options |= orjson.OPT_SORT_KEYS
    def serialize_tensor(self, tensor: torch.Tensor) -> bytes:
        """Serialize PyTorch tensor to bytes"""
        self.validate_tensor(tensor)
        # Move to CPU and convert to numpy for serialization
        cpu_tensor = tensor.cpu() if tensor.device.type != "cpu" else tensor
        numpy_array = cpu_tensor.numpy()
        data = {
            "vectors": numpy_array,  # orjson will convert to lists
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "device": str(tensor.device),
            "serializer": "orjson",
        }
        return orjson.dumps(data, option=self.orjson_options)
    def deserialize_tensor(self, data: bytes, target_device: Optional[torch.device] = None) -> torch.Tensor:
        """Deserialize orjson bytes back to PyTorch tensor"""
        try:
            loaded = orjson.loads(data)
        except orjson.JSONDecodeError as e:
            raise SerializationProviderError("Invalid JSON data: %s" % e)
        # Validate structure
        if "vectors" not in loaded or "dtype" not in loaded:
            raise SerializationProviderError("Invalid tensor data structure")
        # Convert back to tensor (loaded['vectors'] is a plain Python list)
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "int64": torch.int64,
            "int32": torch.int32,
            "bool": torch.bool,
        }
        tensor = torch.tensor(loaded["vectors"], dtype=dtype_map.get(loaded["dtype"], torch.float32))
        # Move to target device if specified
        if target_device is not None:
            tensor = tensor.to(target_device)
        return tensor
    def validate_metadata(self, metadata: Dict[str, Any]) -> None:
        """Validate metadata dictionary with orjson-specific rules"""
        if not isinstance(metadata, dict):
            raise SerializationProviderError("Metadata must be a dictionary")
        # Use custom validation that allows ChunkMetadata objects
        self._validate_orjson_serializable(metadata)
    def _validate_orjson_serializable(self, obj, path: str = "root") -> None:
        """Validate that object can be serialized with orjson and custom default function"""
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return
        elif isinstance(obj, dict):
            for key, value in obj.items():
                if not isinstance(key, str):
                    raise SerializationProviderError("Dictionary key must be string at %s.%s" % (path, key))
                self._validate_orjson_serializable(value, "%s.%s" % (path, key))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._validate_orjson_serializable(item, "%s[%s]" % (path, i))
        elif hasattr(obj, "to_dict"):
            # Objects with to_dict method can be serialized with our custom default
            return
        elif hasattr(obj, "__dict__"):
            # Objects with __dict__ can be serialized with our custom default
            return
        else:
            # For other types, let orjson handle it with the default function
            return
    def serialize_metadata(self, metadata: Dict[str, Any]) -> bytes:
        """Serialize metadata dictionary to bytes"""
        self.validate_metadata(metadata)
        # Use custom default function to handle ChunkMetadata objects
        return orjson.dumps(metadata, option=self.orjson_options, default=_json_default)
    def deserialize_metadata(self, data: bytes) -> Dict[str, Any]:
        """Deserialize bytes back to metadata dictionary"""
        try:
            metadata = orjson.loads(data)
            # Restore ChunkMetadata objects
            return self._restore_objects(metadata)
        except orjson.JSONDecodeError as e:
            raise SerializationProviderError("Invalid JSON metadata: %s" % e)
    def _restore_objects(self, obj):
        """Recursively restore objects from serialized form"""
        if isinstance(obj, dict):
            if "_object_type" in obj and obj["_object_type"] == "ChunkMetadata":
                # Restore ChunkMetadata object
                try:
                    from ..chunking.mod_chunking_base import ChunkMetadata
                    obj_copy = obj.copy()
                    del obj_copy["_object_type"]
                    return ChunkMetadata.from_dict(obj_copy)
                except ImportError:
                    # If import fails, return as dict without the type marker
                    obj_copy = obj.copy()
                    if "_object_type" in obj_copy:
                        del obj_copy["_object_type"]
                    return obj_copy
            else:
                # Recursively process dictionary values
                return {key: self._restore_objects(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            # Recursively process list items
            return [self._restore_objects(item) for item in obj]
        else:
            return obj
    def _serialize_combined_data(self, data: Dict[str, Any]) -> bytes:
        """Serialize combined data structure"""
        return orjson.dumps(data, option=self.orjson_options, default=_json_default)
    def _deserialize_combined_data(self, data: bytes) -> Dict[str, Any]:
        """Deserialize combined data structure"""
        try:
            result = orjson.loads(data)
            return self._restore_objects(result)
        except orjson.JSONDecodeError as e:
            raise SerializationProviderError("Invalid JSON data: %s" % e)
def _json_default(obj):
    """Custom default function for orjson to handle non-serializable objects"""
    # Handle ChunkMetadata objects
    if hasattr(obj, "to_dict"):
        result = obj.to_dict()
        result["_object_type"] = obj.__class__.__name__
        return result
    # Handle other common non-serializable types
    if hasattr(obj, "__dict__"):
        result = obj.__dict__.copy()
        result["_object_type"] = obj.__class__.__name__
        return result
    # Fallback to string representation
    return str(obj)
# Legacy functions for backward compatibility
def serialize_tensors_orjson(tensors, output_path):
    """Legacy function - use OrjsonSerializationProvider instead"""
    provider = OrjsonSerializationProvider()
    start_time = time.time()
    # Convert tensors to numpy arrays for orjson
    tensor_data = []
    for i, tensor in enumerate(tensors):
        numpy_array = tensor.cpu().numpy()
        tensor_data.append(
            {
                "id": i,
                "data": numpy_array,  # orjson will convert this to lists
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
            }
        )
    data = {"tensors": tensor_data, "count": len(tensors), "method": "orjson_numpy"}
    # Serialize with numpy support
    with open(output_path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY))
    serialize_time = time.time() - start_time
    file_size = Path(output_path).stat().st_size
    return {
        "serialize_time": serialize_time,
        "file_size": file_size,
        "method": "orjson",
    }
def deserialize_tensors_orjson(input_path):
    """Legacy function - use OrjsonSerializationProvider instead"""
    start_time = time.time()
    with open(input_path, "rb") as f:
        data = orjson.loads(f.read())
    tensors = []
    for tensor_info in data["tensors"]:
        # orjson converts numpy arrays to plain Python lists
        tensor_data = tensor_info["data"]  # This is now a Python list
        # Direct conversion from list to tensor
        tensor = torch.tensor(tensor_data, dtype=torch.float32)
        tensors.append(tensor)
    deserialize_time = time.time() - start_time
    return tensors, {"deserialize_time": deserialize_time, "method": "orjson"}
def benchmark_orjson(tensors, output_path):
    """Legacy benchmark function"""
    print(f"Benchmarking orjson with {len(tensors)} tensors...")
    # Serialize
    save_stats = serialize_tensors_orjson(tensors, output_path)
    # Deserialize
    loaded_tensors, load_stats = deserialize_tensors_orjson(output_path)
    # Verify integrity
    integrity_check = all(
        torch.allclose(original, loaded, rtol=1e-5) for original, loaded in zip(tensors, loaded_tensors)
    )
    return {
        "method": "orjson",
        "serialize_time": save_stats["serialize_time"],
        "deserialize_time": load_stats["deserialize_time"],
        "file_size": save_stats["file_size"],
        "integrity_check": integrity_check,
        "mb_per_second_write": (save_stats["file_size"] / 1024 / 1024) / save_stats["serialize_time"],
        "mb_per_second_read": (save_stats["file_size"] / 1024 / 1024) / load_stats["deserialize_time"],
    }
