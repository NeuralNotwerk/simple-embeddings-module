"""
Simple Embeddings Module Registry System
Auto-discovers and registers modules in prescribed folders.
Provides isolated registries per SEMDatabase instance.
Handles configuration validation and capability negotiation.
"""
import inspect
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)
@dataclass
class ConfigParameter:
    """Configuration parameter definition for modules"""
    key_name: str
    value_type: str  # "str", "numeric", "list", "bool"
    config_description: str
    required: bool = False
    allows_none_null: bool = False
    value_opt_default: Any = None
    value_opt_regex: str = ""
    def __post_init__(self):
        """Validate parameter definition"""
        valid_types = ["str", "numeric", "list", "bool"]
        if self.value_type not in valid_types:
            raise ValueError("Invalid value_type: %s. Must be one of %s" % (self.value_type, valid_types))
        if not re.match(r"^[a-z][a-z0-9_]*$", self.key_name):
            raise ValueError("Invalid key_name: %s. Must be lowercase snake_case" % self.key_name)
@dataclass
class ModuleCapabilities:
    """Module capabilities and configuration requirements"""
    module_type: str  # "embedding", "storage", "serialization"
    module_name: str
    module_class: Type
    config_parameters: List[ConfigParameter] = field(default_factory=list)
    capabilities: Dict[str, Any] = field(default_factory=dict)
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize configuration for this module"""
        validated = {}
        for param in self.config_parameters:
            value = config.get(param.key_name)
            # Handle required parameters
            if param.required and (value is None or value == ""):
                raise ValueError("Required parameter '%s' is missing" % param.key_name)
            # Handle None/null values
            if value is None:
                if param.allows_none_null:
                    validated[param.key_name] = None
                    continue
                elif param.value_opt_default is not None:
                    validated[param.key_name] = param.value_opt_default
                    continue
                elif not param.required:
                    continue  # Skip optional unset parameters
                else:
                    raise ValueError("Parameter '%s' cannot be None" % param.key_name)
            # Type conversion and validation
            validated[param.key_name] = self._validate_parameter_value(param, value)
        return validated
    def _validate_parameter_value(self, param: ConfigParameter, value: Any) -> Any:
        """Validate and convert a single parameter value"""
        if param.value_type == "str":
            str_value = str(value)
            if param.value_opt_regex and not re.match(param.value_opt_regex, str_value):
                raise ValueError(
                    "Parameter '%s' value '%s' does not match regex: %s" % (param.key_name, str_value, param.value_opt_regex)
                )
            return str_value
        elif param.value_type == "numeric":
            try:
                if isinstance(value, (int, float)):
                    num_value = value
                else:
                    num_value = float(value) if "." in str(value) else int(value)
                if param.value_opt_regex:
                    str_num = str(num_value)
                    if not re.match(param.value_opt_regex, str_num):
                        raise ValueError(
                            "Parameter '%s' numeric value '%s' does not match regex: %s" % (param.key_name, num_value, param.value_opt_regex)
                        )
                return num_value
            except (ValueError, TypeError):
                raise ValueError("Parameter '%s' must be numeric, got: %s" % (param.key_name, value))
        elif param.value_type == "list":
            if not isinstance(value, list):
                raise ValueError("Parameter '%s' must be a list, got: %s" % (param.key_name, type(value)))
            if param.value_opt_regex:
                for item in value:
                    if not re.match(param.value_opt_regex, str(item)):
                        raise ValueError(
                            "Parameter '%s' list item '%s' does not match regex: %s" % (param.key_name, item, param.value_opt_regex)
                        )
            return value
        elif param.value_type == "bool":
            return self._parse_bool_value(param.key_name, value)
        else:
            raise ValueError("Unknown parameter type: %s" % param.value_type)
    def _parse_bool_value(self, key_name: str, value: Any) -> bool:
        """Parse boolean value from various string representations"""
        if isinstance(value, bool):
            return value
        str_value = str(value).lower()
        true_values = {"t", "true", "1", "on", "enabled", "y", "yes"}
        false_values = {"", "false", "0", "o", "disabled", "n", "no"}
        if str_value in true_values:
            return True
        elif str_value in false_values:
            return False
        else:
            raise ValueError(
                "Parameter '%s' boolean value '%s' not recognized. Use: %s" % (key_name, value, true_values | false_values)
            )
class ModuleRegistry:
    """Registry for managing modules within a SEMDatabase instance"""
    def __init__(self, instance_id: str):
        self.instance_id = instance_id
        self.registered_modules: Dict[str, Dict[str, ModuleCapabilities]] = {
            "embeddings": {},
            "storage": {},
            "serialization": {},
            "chunking": {},
        }
        self._active_modules: Dict[str, Any] = {}
    def register_module(self, capabilities: ModuleCapabilities) -> None:
        """Register a module with this registry"""
        module_type = capabilities.module_type
        module_name = capabilities.module_name
        if module_type not in self.registered_modules:
            raise ValueError("Unknown module type: %s" % module_type)
        self.registered_modules[module_type][module_name] = capabilities
        logger.debug("Registered %s module: %s for instance %s", module_type, module_name, self.instance_id)
    def unregister_module(self, module_type: str, module_name: str) -> None:
        """Unregister a module from this registry"""
        if module_type in self.registered_modules:
            self.registered_modules[module_type].pop(module_name, None)
            # Also remove from active modules if present
            active_key = f"{module_type}_{module_name}"
            self._active_modules.pop(active_key, None)
            logger.debug("Unregistered %s module: %s from instance %s", module_type, module_name, self.instance_id)
    def get_available_modules(self, module_type: str) -> List[str]:
        """Get list of available module names for a given type"""
        return list(self.registered_modules.get(module_type, {}).keys())
    def get_module_capabilities(self, module_type: str, module_name: str) -> Optional[ModuleCapabilities]:
        """Get capabilities for a specific module"""
        return self.registered_modules.get(module_type, {}).get(module_name)
    def instantiate_module(self, module_type: str, module_name: str, config: Dict[str, Any]) -> Any:
        """Instantiate a module with validated configuration"""
        capabilities = self.get_module_capabilities(module_type, module_name)
        if not capabilities:
            raise ValueError("Module not found: %s.%s" % (module_type, module_name))
        # Separate special parameters that bypass validation
        special_params = {}
        regular_config = {}
        for key, value in config.items():
            if key == "embedding_provider":  # Special case for chunking providers
                special_params[key] = value
            else:
                regular_config[key] = value
        # Validate regular configuration
        validated_config = capabilities.validate_config(regular_config)
        # Add back special parameters
        final_config = {**validated_config, **special_params}
        logger.debug("Instantiating %s.%s with final config keys: %s", module_type, module_name, list(final_config.keys()))
        # Instantiate module
        try:
            module_instance = capabilities.module_class(**final_config)
            active_key = f"{module_type}_{module_name}"
            self._active_modules[active_key] = module_instance
            logger.info("Instantiated %s module: %s for instance %s", module_type, module_name, self.instance_id)
            return module_instance
        except Exception as e:
            logger.error("Failed to instantiate %s.%s: %s", module_type, module_name, e)
            raise RuntimeError("Failed to instantiate %s.%s: %s" % (module_type, module_name, e))
    def get_active_module(self, module_type: str, module_name: str) -> Optional[Any]:
        """Get an active module instance"""
        active_key = f"{module_type}_{module_name}"
        return self._active_modules.get(active_key)
    def unload_all_modules_except(self, module_type: str, keep_module: str) -> None:
        """Unload all modules of a type except the specified one"""
        modules_to_remove = []
        for active_key, instance in self._active_modules.items():
            if active_key.startswith(f"{module_type}_") and not active_key.endswith(f"_{keep_module}"):
                modules_to_remove.append(active_key)
        for key in modules_to_remove:
            del self._active_modules[key]
            logger.debug("Unloaded module: %s from instance %s", key, self.instance_id)
class GlobalModuleDiscovery:
    """Global module discovery and registration system"""
    _discovered_modules: Dict[str, ModuleCapabilities] = {}
    _discovery_complete = False
    @classmethod
    def discover_modules(cls, base_path: Optional[Path] = None) -> None:
        """Discover all modules in the prescribed folders"""
        if cls._discovery_complete:
            return
        if base_path is None:
            # Default to the package directory
            base_path = Path(__file__).parent
        module_folders = {
            "embeddings": base_path / "embeddings",
            "storage": base_path / "storage",
            "serialization": base_path / "serialization",
            "chunking": base_path / "chunking",
        }
        for module_type, folder_path in module_folders.items():
            if folder_path.exists():
                cls._discover_modules_in_folder(module_type, folder_path)
        cls._discovery_complete = True
        logger.info("Module discovery complete. Found %d modules.", len(cls._discovered_modules))
    @classmethod
    def _discover_modules_in_folder(cls, module_type: str, folder_path: Path) -> None:
        """Discover modules in a specific folder"""
        logger.debug("Discovering modules in folder: %s (type: %s)", folder_path, module_type)
        for py_file in folder_path.glob("mod_*.py"):
            logger.debug("Found potential module file: %s", py_file.name)
            if py_file.name.startswith("mod_") and not py_file.name.endswith("_base.py"):
                # Exclude chunking implementation modules (not provider modules)
                if module_type == "chunking" and (
                    py_file.name.startswith("mod_chunking_ts") or py_file.name == "mod_chunking_ts_lang_lazy.py"
                ):
                    logger.debug("Skipping chunking implementation module: %s", py_file.name)
                    continue
                logger.debug("Attempting to load capabilities for: %s", py_file.name)
                try:
                    capabilities = cls._load_module_capabilities(module_type, py_file)
                    if capabilities:
                        module_key = f"{module_type}.{capabilities.module_name}"
                        cls._discovered_modules[module_key] = capabilities
                        logger.debug("Discovered module: %s", module_key)
                    else:
                        logger.debug("No capabilities returned for: %s", py_file.name)
                except Exception as e:
                    logger.warning("Failed to load module %s: %s", py_file, e)
    @classmethod
    def _load_module_capabilities(cls, module_type: str, py_file: Path) -> Optional[ModuleCapabilities]:
        """Load module capabilities from a Python file"""
        module_name = py_file.stem.replace("mod_", "")
        # Import the module using proper package path
        try:
            # Construct proper module path - folder names are already correct
            # Use relative import for installed package, absolute for development
            if __package__:
                # Installed package - use relative imports
                package_path = f"{__package__}.{module_type}.{py_file.stem}"
            else:
                # Development mode - use absolute path
                package_path = f"src.simple_embeddings_module.{module_type}.{py_file.stem}"
            # Import the module
            import importlib
            module = importlib.import_module(package_path)
        except ImportError as e:
            logger.warning("Failed to import %s: %s", package_path, e)
            return None
        except Exception as e:
            logger.warning("Error loading %s: %s", package_path, e)
            return None
        # Look for classes that inherit from the appropriate base class
        base_class_names = {
            "embeddings": "EmbeddingProviderBase",
            "storage": "StorageBackendBase",
            "serialization": "SerializationProviderBase",
            "chunking": "ChunkingProviderBase",
        }
        target_base = base_class_names.get(module_type)
        if not target_base:
            return None
        # Find the implementation class
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                obj.__module__ == module.__name__
                and hasattr(obj, "__bases__")
                and any(base.__name__ == target_base for base in obj.__mro__)
            ):
                # Get configuration parameters if defined
                config_params = getattr(obj, "CONFIG_PARAMETERS", [])
                capabilities = getattr(obj, "CAPABILITIES", {})
                return ModuleCapabilities(
                    module_type=module_type,
                    module_name=module_name,
                    module_class=obj,
                    config_parameters=config_params,
                    capabilities=capabilities,
                )
        return None
    @classmethod
    def get_discovered_modules(cls) -> Dict[str, ModuleCapabilities]:
        """Get all discovered modules"""
        if not cls._discovery_complete:
            cls.discover_modules()
        return cls._discovered_modules.copy()
    @classmethod
    def create_registry_for_instance(cls, instance_id: str) -> ModuleRegistry:
        """Create a new registry for a SEMDatabase instance"""
        if not cls._discovery_complete:
            cls.discover_modules()
        registry = ModuleRegistry(instance_id)
        # Register all discovered modules with this instance
        for module_key, capabilities in cls._discovered_modules.items():
            registry.register_module(capabilities)
        return registry
# Convenience functions for external use
def discover_all_modules(base_path: Optional[Path] = None) -> None:
    """Discover all available modules"""
    GlobalModuleDiscovery.discover_modules(base_path)
def create_module_registry(instance_id: str) -> ModuleRegistry:
    """Create a new module registry for a SEMDatabase instance"""
    return GlobalModuleDiscovery.create_registry_for_instance(instance_id)
def get_available_modules() -> Dict[str, List[str]]:
    """Get all available modules organized by type"""
    modules = GlobalModuleDiscovery.get_discovered_modules()
    result = {"embeddings": [], "storage": [], "serialization": [], "chunking": []}
    for module_key, capabilities in modules.items():
        module_type = capabilities.module_type
        module_name = capabilities.module_name
        if module_type in result:
            result[module_type].append(module_name)
    return result
