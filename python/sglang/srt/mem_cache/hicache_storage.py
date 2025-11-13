import hashlib
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)

# Try to import cuFile for GPU Direct Storage support
CUFILE_AVAILABLE = False
CUFILE_MODULE = None

try:
    import cufile
    CUFILE_AVAILABLE = True
    CUFILE_MODULE = cufile
    logger.debug("cuFile module found")
except ImportError:
    try:
        # Try PyTorch's GDS API (if available)
        from torch.cuda import gds
        CUFILE_AVAILABLE = True
        CUFILE_MODULE = gds
        logger.debug("PyTorch GDS API found")
    except ImportError:
        CUFILE_AVAILABLE = False
        logger.debug("cuFile not available, falling back to POSIX I/O")

def get_hash_str(token_ids: List[int], prior_hash: str = None) -> str:
    hasher = hashlib.sha256()

    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))

    for t in token_ids:
        if isinstance(t, tuple):
            # EAGLE bigram mode: hash both elements to uniquely identify the bigram
            for elem in t:
                hasher.update(elem.to_bytes(4, byteorder="little", signed=False))
        else:
            # Regular mode: single integer token
            hasher.update(t.to_bytes(4, byteorder="little", signed=False))

    return hasher.hexdigest()


@dataclass
class HiCacheStorageConfig:
    tp_rank: int
    tp_size: int
    is_mla_model: bool
    is_page_first_layout: bool
    model_name: Optional[str]
    extra_config: Optional[dict] = None


@dataclass
class HiCacheStorageExtraInfo:
    prefix_keys: Optional[List[str]] = (None,)
    extra_info: Optional[dict] = None


class HiCacheStorage(ABC):
    """
    HiCacheStorage is a class that provides a generic key-value interface for storing and retrieving KV cache.
    It abstracts the underlying storage mechanism, allowing different implementations to be used.
    """

    # todo, the page size of storage backend does not have to be the same as the same as host memory pool

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        self.mem_pool_host = mem_pool_host

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        pass

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        pass

    @abstractmethod
    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        """
        Retrieve the value associated with the given key.
        Returns None if the key does not exist.
        """
        pass

    # TODO: Deprecate
    @abstractmethod
    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        pass

    @abstractmethod
    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store the value associated with the given key.
        Returns True if the operation was successful, False otherwise.
        """
        pass

    # TODO: Deprecate
    @abstractmethod
    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store multiple key-value pairs.
        Returns True if all operations were successful, False otherwise.
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if the key exists in the storage.
        Returns True if the key exists, False otherwise.
        """
        pass

    # TODO: Use a finer-grained return type (e.g., List[bool])
    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        """
        Check if the keys exist in the storage.
        return the number of consecutive existing keys from the start.
        Can be overridden by subclasses for more efficient implementation.
        """
        for i in range(len(keys)):
            if not self.exists(keys[i]):
                return i
        return len(keys)

    def clear(self) -> None:
        pass

    def get_stats(self):
        return None


class HiCacheFile(HiCacheStorage):

    def __init__(
        self, storage_config: HiCacheStorageConfig, file_path: str = "/tmp/hicache"
    ):
        self.file_path = os.getenv("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR", file_path)

        tp_rank, tp_size, model_name, is_mla_model = (
            storage_config.tp_rank,
            storage_config.tp_size,
            storage_config.model_name,
            storage_config.is_mla_model,
        )
        model_name = "-".join(model_name.split("/")) if model_name else ""
        if is_mla_model:
            self.config_suffix = f"_{model_name}"
        else:
            self.config_suffix = f"_{model_name}_{tp_rank}_{tp_size}"

        if not os.path.exists(self.file_path) and tp_rank == 0:
            os.makedirs(self.file_path)
            logger.info(f"Created HiCacheFile storage directory at {self.file_path}")
        # Check if GDS should be enabled
        # Store model type for batch operations
        self.is_mla_backend = is_mla_model
        self.local_rank = tp_rank

        # Check if GDS should be enabled (from config or environment variable)
        enable_gds = False
        if storage_config.extra_config and "enable_gds" in storage_config.extra_config:
            enable_gds = bool(storage_config.extra_config["enable_gds"])
        else:
            enable_gds = os.getenv("SGLANG_HICACHE_FILE_BACKEND_ENABLE_GDS", "false").lower() == "true"

        # GDS requires CUDA and cuFile
        self.use_gds = enable_gds and CUFILE_AVAILABLE and torch.cuda.is_available()

        if self.use_gds:
            try:
                # Initialize cuFile if using cufile module
                if CUFILE_MODULE and hasattr(CUFILE_MODULE, 'init'):
                    CUFILE_MODULE.init()
                logger.info("GPU Direct Storage (GDS) enabled for HiCacheFile backend")
            except Exception as e:
                logger.warning(f"Failed to initialize cuFile, falling back to POSIX I/O: {e}")
                self.use_gds = False
        else:
            if enable_gds and not CUFILE_AVAILABLE:
                logger.warning("GDS requested but cuFile not available. Install cufile package to enable GDS.")
            elif enable_gds and not torch.cuda.is_available():
                logger.warning("GDS requested but CUDA not available. Falling back to POSIX I/O.")

        # Will be set in register_mem_pool_host
        self.kv_buffer_is_cuda = False
        self.use_gds_for_batch = False

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.config_suffix

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)
        
        # Check if kv_buffer supports GDS zero-copy
        # GDS supports both GPU memory and CPU pinned memory:
        # - GPU memory: Direct zero-copy from NVMe to GPU
        # - CPU pinned memory: Zero-copy from NVMe to CPU pinned memory, then DMA to GPU
        if hasattr(mem_pool_host, 'kv_buffer'):
            self.kv_buffer_is_cuda = mem_pool_host.kv_buffer.is_cuda
            self.kv_buffer_is_pinned = (
                hasattr(mem_pool_host, 'pin_memory') and mem_pool_host.pin_memory
            )
            
            # Enable GDS if buffer is on GPU or CPU pinned memory
            if self.use_gds:
                if self.kv_buffer_is_cuda:
                    # Direct GPU zero-copy: NVMe -> GPU (best performance)
                    self.use_gds_for_batch = True
                    logger.info("GDS enabled for GPU memory (direct zero-copy)")
                elif self.kv_buffer_is_pinned:
                    # CPU pinned memory zero-copy: NVMe -> CPU pinned -> GPU via DMA
                    # This avoids CPU copy overhead and uses DMA for CPU->GPU transfer
                    self.use_gds_for_batch = True
                    logger.info(
                        "GDS enabled for CPU pinned memory "
                        "(zero-copy from NVMe, DMA to GPU)"
                    )
                else:
                    # Regular CPU memory: GDS may not provide significant benefit
                    self.use_gds_for_batch = False
                    logger.warning(
                        "GDS enabled but kv_buffer is on regular CPU memory (not pinned). "
                        "Consider using pin_memory=True for better zero-copy performance. "
                        "Falling back to POSIX I/O for batch operations."
                    )
            else:
                self.use_gds_for_batch = False
        else:
            self.kv_buffer_is_cuda = False
            self.kv_buffer_is_pinned = False
            self.use_gds_for_batch = False

    def get(
        self,
        key: str,
        target_location: torch.Tensor,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        if not os.path.exists(tensor_path):
            logger.warning(f"Failed to fetch {key} from HiCacheFile storage.")
            return None

        try:
            expected = target_location.numel() * target_location.element_size()
            if self.use_gds and target_location.is_cuda:
                # Use cuFile for zero-copy GPU Direct Storage
                return self._gds_read(tensor_path, target_location, expected)
            else:
                # Fallback to POSIX I/O
                with open(tensor_path, "rb", buffering=0) as f:
                    buf = memoryview(target_location.view(torch.uint8).contiguous().numpy())
                    if f.readinto(buf) != expected:
                        raise IOError(f"Short read for {key}")
                return target_location
        except Exception as e:
            logger.error(f"Failed to read {key} from HiCacheFile storage: {e}")
        except FileNotFoundError:
            logger.warning(f"Failed to fetch {key} from HiCacheFile storage.")
            return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: List[torch.Tensor],
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        return [
            self.get(key, target_location)
            for key, target_location in zip(
                keys, target_locations or [None] * len(keys)
            )
        ]

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if self.exists(key):
            logger.debug(f"Key {key} already exists. Skipped.")
            return True

        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        try:
            if self.use_gds and value is not None and value.is_cuda:
                # Use cuFile for zero-copy GPU Direct Storage
                return self._gds_write(tensor_path, value)
            else:
                # Fallback to POSIX I/O
                value.contiguous().view(dtype=torch.uint8).numpy().tofile(tensor_path)
                return True
        except Exception as e:
            logger.error(f"Failed to save tensor {key}: {e}")
            return False

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        for key, value in zip(keys, values):
            if not self.set(key, value):
                return False
        return True

    def exists(self, key: str) -> bool:
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        return os.path.exists(tensor_path)

    def clear(self) -> bool:
        try:
            for filename in os.listdir(self.file_path):
                file_path = os.path.join(self.file_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("Cleared all entries in HiCacheFile storage.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear HiCacheFile storage: {e}")
            return False

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """Batch get using host_indices from memory pool with zero-copy GDS support."""
        if not hasattr(self, 'mem_pool_host') or self.mem_pool_host is None:
            logger.error("memory_pool_host not registered, cannot use batch_get_v1")
            return [False] * len(keys)
        
        page_num = len(host_indices) // self.mem_pool_host.page_size
        
        # Use zero-copy approach if GDS is enabled and buffer is on GPU
        if self.use_gds_for_batch:
            return self._batch_get_v1_gds(keys, host_indices, page_num)
        else:
            # Fallback to tensor-based approach (POSIX I/O)
            return self._batch_get_v1_posix(keys, host_indices, page_num)

    def _batch_get_v1_gds(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        page_num: int,
    ) -> List[bool]:
        """Zero-copy batch get using GDS with memory pointers or tensors."""
        # Check which GDS API is available
        use_cufile_api = CUFILE_MODULE and hasattr(CUFILE_MODULE, 'CuFile')
        use_pytorch_gds_api = CUFILE_MODULE and hasattr(CUFILE_MODULE, 'GdsFile')
        
        if use_cufile_api:
            # Use cuFile API with raw pointers (best performance)
            return self._batch_get_v1_gds_cufile(keys, host_indices, page_num)
        elif use_pytorch_gds_api:
            # Use PyTorch GdsFile API with tensors
            return self._batch_get_v1_gds_pytorch(keys, host_indices, page_num)
        else:
            logger.error("No GDS API available, falling back to POSIX")
            return self._batch_get_v1_posix(keys, host_indices, page_num)

    def _batch_get_v1_gds_cufile(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        page_num: int,
    ) -> List[bool]:
        """Zero-copy batch get using cuFile API with raw pointers."""
        # Get buffer metadata (pointers and sizes) - similar to mooncake
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(host_indices)
        
        # Prepare keys with suffix
        suffixed_keys = [self._get_suffixed_key(key) for key in keys]
        
        # For MHA: each key maps to 2 pointers (K and V)
        # For MLA: each key maps to 1 pointer
        if self.is_mla_backend:
            key_multiplier = 1
            gds_keys = suffixed_keys
        else:
            key_multiplier = 2
            gds_keys = []
            for key in suffixed_keys:
                gds_keys.append(f"{key}_k")
                gds_keys.append(f"{key}_v")
        
        # Ensure key count matches pointer count
        assert len(gds_keys) == len(ptr_list), \
            f"Key count ({len(gds_keys)}) != pointer count ({len(ptr_list)})"
        
        # Perform zero-copy reads using cuFile
        results = []
        for i, (key, ptr, size) in enumerate(zip(gds_keys, ptr_list, element_size_list)):
            tensor_path = os.path.join(self.file_path, f"{key}.bin")
            
            if not os.path.exists(tensor_path):
                results.append(False)
                continue
            
            try:
                f = CUFILE_MODULE.CuFile(tensor_path, "r")
                try:
                    bytes_read = f.pread(ptr, size, 0)
                    results.append(bytes_read == size)
                finally:
                    f.close()
            except Exception as e:
                logger.error(f"GDS read failed for {key}: {e}")
                results.append(False)
        
        # Post-process results (similar to mooncake's _batch_postprocess)
        if key_multiplier == 1:
            # MLA: results are 1:1 with keys
            return results
        else:
            # MHA: combine K and V results
            kv_pairs = zip(results[::2], results[1::2])
            return [k_res and v_res for k_res, v_res in kv_pairs]

    def _batch_get_v1_gds_pytorch(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        page_num: int,
    ) -> List[bool]:
        """Zero-copy batch get using PyTorch GdsFile API with tensors."""
        # For PyTorch GdsFile API, we need tensor objects, not raw pointers
        # Get target locations as tensors
        target_locations = []
        for i in range(page_num):
            page_tensor = self.mem_pool_host.get_data_page(
                host_indices[i * self.mem_pool_host.page_size], flat=False
            )
            target_locations.append(page_tensor)
        
        # Prepare keys with suffix
        suffixed_keys = [self._get_suffixed_key(key) for key in keys]
        
        # For MHA: each key maps to 2 tensors (K and V)
        # For MLA: each key maps to 1 tensor
        if self.is_mla_backend:
            gds_keys = suffixed_keys
            tensor_list = target_locations
        else:
            # MHA: split each page tensor into K and V
            gds_keys = []
            tensor_list = []
            for key, page_tensor in zip(suffixed_keys, target_locations):
                # page_tensor shape: (2, ...) for K and V
                gds_keys.append(f"{key}_k")
                gds_keys.append(f"{key}_v")
                tensor_list.append(page_tensor[0])  # K
                tensor_list.append(page_tensor[1])  # V
        
        # Perform zero-copy reads using PyTorch GdsFile
        results = []
        for key, target_tensor in zip(gds_keys, tensor_list):
            tensor_path = os.path.join(self.file_path, f"{key}.bin")
            
            if not os.path.exists(tensor_path):
                results.append(False)
                continue
            
            try:
                # Ensure tensor is contiguous
                if not target_tensor.is_contiguous():
                    target_tensor = target_tensor.contiguous()
                
                # Use PyTorch GdsFile API
                import os
                flags = os.O_RDONLY
                gds_file = CUFILE_MODULE.GdsFile(tensor_path, flags)
                try:
                    gds_file.register_handle()
                    gds_file.load_storage(target_tensor.untyped_storage(), offset=0)
                    results.append(True)
                finally:
                    gds_file.deregister_handle()
            except Exception as e:
                logger.error(f"GDS read failed for {key}: {e}")
                results.append(False)
        
        # Post-process results
        if self.is_mla_backend:
            return results
        else:
            # MHA: combine K and V results
            kv_pairs = zip(results[::2], results[1::2])
            return [k_res and v_res for k_res, v_res in kv_pairs]

    def _batch_get_v1_posix(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        page_num: int,
    ) -> List[bool]:
        """Fallback batch get using POSIX I/O with tensors."""
        target_locations = [
            self.mem_pool_host.get_data_page(
                host_indices[i * self.mem_pool_host.page_size], flat=False
            )
            for i in range(page_num)
        ]
        
        results = self.batch_get(keys, target_locations)
        return [r is not None for r in results]

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """Batch set using host_indices from memory pool with zero-copy GDS support."""
        if not hasattr(self, 'mem_pool_host') or self.mem_pool_host is None:
            logger.error("memory_pool_host not registered, cannot use batch_set_v1")
            return [False] * len(keys)
        
        page_num = len(host_indices) // self.mem_pool_host.page_size
        
        # Use zero-copy approach if GDS is enabled and buffer is on GPU
        if self.use_gds_for_batch:
            return self._batch_set_v1_gds(keys, host_indices, page_num)
        else:
            # Fallback to tensor-based approach (POSIX I/O)
            return self._batch_set_v1_posix(keys, host_indices, page_num)


    def _batch_set_v1_gds(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        page_num: int,
    ) -> List[bool]:
        """Zero-copy batch set using GDS with memory pointers or tensors."""
        # Check which GDS API is available
        use_cufile_api = CUFILE_MODULE and hasattr(CUFILE_MODULE, 'CuFile')
        use_pytorch_gds_api = CUFILE_MODULE and hasattr(CUFILE_MODULE, 'GdsFile')
        
        if use_cufile_api:
            # Use cuFile API with raw pointers (best performance)
            return self._batch_set_v1_gds_cufile(keys, host_indices, page_num)
        elif use_pytorch_gds_api:
            # Use PyTorch GdsFile API with tensors
            return self._batch_set_v1_gds_pytorch(keys, host_indices, page_num)
        else:
            logger.error("No GDS API available, falling back to POSIX")
            return self._batch_set_v1_posix(keys, host_indices, page_num)

    def _batch_set_v1_gds_cufile(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        page_num: int,
    ) -> List[bool]:
        """Zero-copy batch set using cuFile API with raw pointers."""
        # Get buffer metadata (pointers and sizes) - similar to mooncake
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(host_indices)
        
        # Prepare keys with suffix
        suffixed_keys = [self._get_suffixed_key(key) for key in keys]
        
        # For MHA: each key maps to 2 pointers (K and V)
        # For MLA: each key maps to 1 pointer
        if self.is_mla_backend:
            key_multiplier = 1
            gds_keys = suffixed_keys
        else:
            key_multiplier = 2
            gds_keys = []
            for key in suffixed_keys:
                gds_keys.append(f"{key}_k")
                gds_keys.append(f"{key}_v")
        
        # Check which keys already exist (skip them)
        exist_results = []
        for key in gds_keys:
            tensor_path = os.path.join(self.file_path, f"{key}.bin")
            exist_results.append(os.path.exists(tensor_path))
        
        # Perform zero-copy writes using cuFile for non-existing keys
        write_results = []
        for i, (key, ptr, size, exists) in enumerate(zip(gds_keys, ptr_list, element_size_list, exist_results)):
            if exists:
                write_results.append(True)  # Already exists, skip
                continue
            
            tensor_path = os.path.join(self.file_path, f"{key}.bin")
            
            try:
                # Create file if it doesn't exist
                if not os.path.exists(tensor_path):
                    with open(tensor_path, "wb") as f:
                        pass
                
                # Use cuFile to write directly from memory pointer
                f = CUFILE_MODULE.CuFile(tensor_path, "w")
                try:
                    bytes_written = f.pwrite(ptr, size, 0)
                    write_results.append(bytes_written == size)
                finally:
                    f.close()
            except Exception as e:
                logger.error(f"GDS write failed for {key}: {e}")
                write_results.append(False)
        
        # Post-process results (similar to mooncake's _batch_postprocess)
        if key_multiplier == 1:
            # MLA: results are 1:1 with keys
            return write_results
        else:
            # MHA: combine K and V results
            kv_pairs = zip(write_results[::2], write_results[1::2])
            return [k_res and v_res for k_res, v_res in kv_pairs]

    def _batch_set_v1_gds_pytorch(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        page_num: int,
    ) -> List[bool]:
        """Zero-copy batch set using PyTorch GdsFile API with tensors."""
        # For PyTorch GdsFile API, we need tensor objects, not raw pointers
        # Get source values as tensors
        values = []
        for i in range(page_num):
            page_tensor = self.mem_pool_host.get_data_page(
                host_indices[i * self.mem_pool_host.page_size], flat=False
            )
            values.append(page_tensor)
        
        # Prepare keys with suffix
        suffixed_keys = [self._get_suffixed_key(key) for key in keys]
        
        # For MHA: each key maps to 2 tensors (K and V)
        # For MLA: each key maps to 1 tensor
        if self.is_mla_backend:
            gds_keys = suffixed_keys
            tensor_list = values
        else:
            # MHA: split each page tensor into K and V
            gds_keys = []
            tensor_list = []
            for key, page_tensor in zip(suffixed_keys, values):
                # page_tensor shape: (2, ...) for K and V
                gds_keys.append(f"{key}_k")
                gds_keys.append(f"{key}_v")
                tensor_list.append(page_tensor[0])  # K
                tensor_list.append(page_tensor[1])  # V
        
        # Check which keys already exist (skip them)
        exist_results = []
        for key in gds_keys:
            tensor_path = os.path.join(self.file_path, f"{key}.bin")
            exist_results.append(os.path.exists(tensor_path))
        
        # Perform zero-copy writes using PyTorch GdsFile for non-existing keys
        write_results = []
        for key, value_tensor, exists in zip(gds_keys, tensor_list, exist_results):
            if exists:
                write_results.append(True)  # Already exists, skip
                continue
            
            tensor_path = os.path.join(self.file_path, f"{key}.bin")
            
            try:
                # Ensure tensor is contiguous
                if not value_tensor.is_contiguous():
                    value_tensor = value_tensor.contiguous()
                
                # Create file if it doesn't exist
                if not os.path.exists(tensor_path):
                    with open(tensor_path, "wb") as f:
                        pass
                
                # Use PyTorch GdsFile API
                import os
                flags = os.O_WRONLY | os.O_CREAT
                gds_file = CUFILE_MODULE.GdsFile(tensor_path, flags)
                try:
                    gds_file.register_handle()
                    gds_file.save_storage(value_tensor.untyped_storage(), offset=0)
                    write_results.append(True)
                finally:
                    gds_file.deregister_handle()
            except Exception as e:
                logger.error(f"GDS write failed for {key}: {e}")
                write_results.append(False)
        
        # Post-process results
        if self.is_mla_backend:
            return write_results
        else:
            # MHA: combine K and V results
            kv_pairs = zip(write_results[::2], write_results[1::2])
            return [k_res and v_res for k_res, v_res in kv_pairs]

    def _batch_set_v1_posix(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        page_num: int,
    ) -> List[bool]:
        """Fallback batch set using POSIX I/O with tensors."""
        values = [
            self.mem_pool_host.get_data_page(
                host_indices[i * self.mem_pool_host.page_size], flat=False
            )
            for i in range(page_num)
        ]
        
        results = self.batch_set(keys, values)
        if isinstance(results, bool):
            return [results] * len(keys)
        return results

    def _gds_read(self, file_path: str, target_location: torch.Tensor, expected_size: int) -> torch.Tensor:
        """Read from file using GPU Direct Storage (cuFile) for zero-copy."""
        try:
            # Ensure tensor is contiguous
            if not target_location.is_contiguous():
                target_location = target_location.contiguous()

            gpu_ptr = target_location.data_ptr()
            gpu_size = target_location.numel() * target_location.element_size()

            # Try cufile module first
            if CUFILE_MODULE and hasattr(CUFILE_MODULE, 'CuFile'):
                # Use cufile.CuFile API
                f = CUFILE_MODULE.CuFile(file_path, "r")
                try:
                    bytes_read = f.pread(gpu_ptr, gpu_size, 0)
                    if bytes_read != expected_size:
                        raise IOError(f"Short read: expected {expected_size}, got {bytes_read}")
                    return target_location
                finally:
                    f.close()
            elif CUFILE_MODULE and hasattr(CUFILE_MODULE, 'GdsFile'):
                # Use PyTorch's GdsFile API
                import os
                flags = os.O_RDONLY
                gds_file = CUFILE_MODULE.GdsFile(file_path, flags)
                try:
                    gds_file.register_handle()
                    gds_file.load_storage(target_location.untyped_storage(), offset=0)
                    return target_location
                finally:
                    gds_file.deregister_handle()
            else:
                raise RuntimeError("Unsupported cuFile API")

        except Exception as e:
            logger.error(f"GDS read failed for {file_path}, falling back to POSIX: {e}")
            # Fallback to POSIX I/O
            with open(file_path, "rb", buffering=0) as f:
                buf = memoryview(target_location.view(torch.uint8).contiguous().numpy())
                if f.readinto(buf) != expected_size:
                    raise IOError(f"Short read for {file_path}")
            return target_location

    def _gds_write(self, file_path: str, value: torch.Tensor) -> bool:
        """Write to file using GPU Direct Storage (cuFile) for zero-copy."""
        try:
            # Ensure tensor is contiguous
            if not value.is_contiguous():
                value = value.contiguous()

            # Create file if it doesn't exist
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    pass  # Create empty file

            gpu_ptr = value.data_ptr()
            gpu_size = value.numel() * value.element_size()

            # Try cufile module first
            if CUFILE_MODULE and hasattr(CUFILE_MODULE, 'CuFile'):
                # Use cufile.CuFile API
                f = CUFILE_MODULE.CuFile(file_path, "w")
                try:
                    bytes_written = f.pwrite(gpu_ptr, gpu_size, 0)
                    if bytes_written != gpu_size:
                        raise IOError(f"Short write: expected {gpu_size}, got {bytes_written}")
                    return True
                finally:
                    f.close()
            elif CUFILE_MODULE and hasattr(CUFILE_MODULE, 'GdsFile'):
                # Use PyTorch's GdsFile API
                import os
                flags = os.O_WRONLY | os.O_CREAT
                gds_file = CUFILE_MODULE.GdsFile(file_path, flags)
                try:
                    gds_file.register_handle()
                    gds_file.save_storage(value.untyped_storage(), offset=0)
                    return True
                finally:
                    gds_file.deregister_handle()
            else:
                raise RuntimeError("Unsupported cuFile API")

        except Exception as e:
            logger.error(f"GDS write failed for {file_path}, falling back to POSIX: {e}")
            # Fallback to POSIX I/O
            value.contiguous().view(dtype=torch.uint8).numpy().tofile(file_path)
            return True
