import os

_base_path = os.path.abspath(os.path.dirname(__file__))
_include_path = os.path.join(_base_path, 'include')


class LibPath(object):

    def __init__(self, ve_arch):
        if ve_arch == 1:
            import nlcpy_ve1_kernel
            self._lib_dir = nlcpy_ve1_kernel.get_kernel_path()
        elif ve_arch == 3:
            import nlcpy_ve3_kernel
            self._lib_dir = nlcpy_ve3_kernel.get_kernel_path()
        else:
            raise RuntimeError(f"Unknown VE architecture version: {ve_arch}")

        self._common_kernel_path = os.path.join(
            self._lib_dir, 'libnlcpy_ve_kernel_common.so')
        self._fast_math_kernel_path = os.path.join(
            self._lib_dir, 'libnlcpy_ve_kernel_fast_math.so')
        self._no_fast_math_kernel_path = os.path.join(
            self._lib_dir, 'libnlcpy_ve_kernel_no_fast_math.so')
        self._profiling_kernel_path = os.path.join(
            self._lib_dir, 'libnlcpy_profiling.so')
