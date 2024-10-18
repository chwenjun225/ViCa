import ctypes
from typing import Optional, List, Union

import numpy as np
import tensorrt as trt
from cuda import cuda, cudart

def check_cuda_err(err):
    if isinstance(err, cuda.CUresult): 
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("CUDA Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknow error type: {}".format(err))

def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res 

class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array."""
    def __init__(self, size: int, dtype: Optional[np.dtype] = None):
        dtype = dtype or np.dtype(np.uint8)
        nbytes = size * dtype.itemsize 
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))

        self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size, ))
        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes 

    @property
    def host(self) -> np.ndarray:
        return self._host
    
    @host.setter 
    def host(self, data: Union[np.ndarray, bytes]):
        if isinstance(data, np.ndarray):
            if data.size > self.host.size:
                pass

def allocate_buffers():
    pass 

def free_buffers():
    pass

def memcpy_host_to_device():
    pass

def memcpy_device_to_host():
    pass

def _do_inference_base():
    pass


def do_inference():
    pass
