import os


_base_path = os.path.abspath(os.path.dirname(__file__))
_include_path = _base_path + '/include'
_lib_path = _base_path + '/lib'
_common_kernel_path = _lib_path + '/libnlcpy_ve_kernel_common.so'
_fast_math_kernel_path = _lib_path + '/libnlcpy_ve_kernel_fast_math.so'
_no_fast_math_kernel_path = _lib_path + '/libnlcpy_ve_kernel_no_fast_math.so'
_profiling_kernel_path = _lib_path + '/libnlcpy_profiling.so'
