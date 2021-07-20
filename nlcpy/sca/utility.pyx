#
# * The source code in this file is developed independently by NEC Corporation.
#
# # NLCPy License #
#
#     Copyright (c) 2020-2021 NEC Corporation
#     All rights reserved.
#
#     Redistribution and use in source and binary forms, with or without
#     modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#     * Neither NEC Corporation nor the names of its contributors may be
#       used to endorse or promote products derived from this software
#       without specific prior written permission.
#
#     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#     ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#     WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#     DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#     FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#     (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#     LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#     ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#     (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#     SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# distutils: language = c++
import nlcpy
import numpy

from nlcpy import veo
from nlcpy.request import request
from nlcpy.core.core cimport ndarray
from nlcpy.core cimport internal
from nlcpy.core cimport core

from nlcpy.sca.kernel cimport kernel
from nlcpy.sca cimport handle
from nlcpy.sca.descriptor cimport descriptor
from nlcpy.sca.description cimport description

from libcpp.vector cimport vector
from libc.stdint cimport *


cpdef create_descriptor(arrays):
    """Returns one or more stencil descriptors.

    Creates stencil descriptor from ndarrays.

    Parameters
    ----------
    arrays : ndarray or sequence of ndarray
        Ndarrays to be used as input or output of stencil calculations.

    Returns
    -------
    out : sequence  of nlcpy.sca.descriptor.descriptor
        The stencil descriptor.

    Note
    ----
    For usage of the stencil descriptor, see :ref:`Basic Usage <label_sca_basic>` and
    :ref:`Advanced Usage <label_sca_advanced>`.

    Examples
    --------
    >>> import nlcpy as vp
    >>> xin = vp.arange(10, dtype='f4')
    >>> xout = vp.zeros_like(xin)
    >>> dxin, dxout = vp.sca.create_descriptor((xin, xout))
    """

    if type(arrays) is ndarray:
        return descriptor(arrays)

    descriptors = []
    if type(arrays) in (list, tuple):
        for _i in arrays:
            if type(_i) is not ndarray:
                raise TypeError('input elements must be ndarray object.')
            descriptors.append(descriptor(_i))
    else:
        raise TypeError(
            'input type must be `list of nlcpy.ndarray` or '
            '`tuple of nlcpy.ndarray` or `nlcpy.ndarray`.')

    # ndim check
    cdef int64_t ndim1 = arrays[0].ndim
    cdef int64_t ndim2
    for i in range(1, len(arrays)):
        ndim2 = arrays[i].ndim
        if ndim1 != ndim2:
            raise ValueError('input arrays must have same dimensions.')

    return descriptors

cpdef empty_description():
    """Returns an empty stencil description.

    Returns
    -------
    out : nlcpy.sca.description.description
        An empty stencil description.

    Note
    ----
    For usage of the empty stencil description, see
    :ref:`Advanced Usage <label_sca_advanced>`.
    """
    return description()

cpdef create_kernel(description desc_i, description desc_o=None):
    """Creates a SCA kernel.

    Creates a SCA kernel from a stencil description that denotes stencil shapes.
    If the keyword argument desc_o is omitted, the SCA kernel's output array is
    automatically created when the SCA kernel is executed.

    Parameters
    ----------
    desc_i : nlcpy.sca.description.description
        The stenil description that is associated with the input ndarray.
    desc_o : nlcpy.sca.description.description, optional
        The stenil description that is associated with the output ndarray. If not given
        or None, the kernel creates a new output ndarray when the kernel execution is
        done.

    Returns
    -------
    kernel : nlcpy.sca.kernel.kernel
        The SCA kernel.

    See Also
    --------
    nlcpy.sca.kernel.kernel.execute : Executes the created SCA kernel and returns the
        result of stencil computations.
    nlcpy.sca.destroy_kernel : Destroy a SCA kernel.
        It is recommended that the kernel be properly destroyed by the function after
        you finish using the kernel.

    Examples
    --------
    >>> import nlcpy as vp
    >>> xin = vp.arange(10, dtype='f4')
    >>> xout = vp.zeros_like(xin)
    >>> dxin, dxout = vp.sca.create_descriptor((xin, xout))
    >>> desc_i = dxin[-1] + dxin[0] + dxin[1]
    >>> desc_o = dxout[0]
    >>> kern = vp.sca.create_kernel(desc_i, desc_o=desc_o)
    >>> kern.execute()
    array([ 0.,  3.,  6.,  9., 12., 15., 18., 21., 24.,  0.], dtype=float32)
    """
    if desc_o is None:
        out_tmp = create_optimized_array(desc_i.shape, dtype=desc_i.dtype)
        out_location = [0 for _ in range(out_tmp.ndim)]
        desc_o = create_descriptor(out_tmp)[out_location]

    if desc_i.ndim != desc_o.ndim:
        raise ValueError('input array and output array must have same dimension.')
    if desc_i.dtype != desc_o.dtype:
        raise TypeError('input array and output array must have same dtype.')
    if len(desc_o.elems) > 1:
        raise ValueError('too many elements for output description.')

    hnd = handle.sca_handle(desc_o.dtype)
    hnd.set_elements(desc_i, desc_o)
    kern = hnd.create_kernel()
    hnd.reset_stencil_elements()
    return kern

cpdef destroy_kernel(kernel kern):
    """Destroy a SCA kernel.

    Parameters
    ----------
    kern : nlcpy.sca.kernel.kernel
        The SCA kernel to destroy.

    Examples
    --------
    >>> import nlcpy as vp
    >>> xin = vp.arange(10, dtype='f4')
    >>> dxin = vp.sca.create_descriptor(xin)
    >>> kern = vp.sca.create_kernel(dxin[0], desc_o=None)
    >>> vp.sca.destroy_kernel(kern)
    >>> kern.execute()   # doctest: +SKIP
    ...
    RuntimeError: this kernel has already been destroyed.
    """
    kern._destroy()

cpdef ndarray convert_optimized_array(a, dtype=None):
    """Converts existing ndarrays into optimized ndarrays, whose strides are adjusted to
    improve perfomance, filled with zeros.

    Parameters
    ----------
    a : ndarray
        The ndarray to be optimized.
    dtype : str or dtype, optional
        The type of the output array. If *dtype* is not given or ``None``, infer the data
        type from input arguments.

    Returns
    -------
    optimized_array : ndarray
        The optimized ndarray.

    Note
    ----
    This function returns a copy of the input ndarray *a*, not a view. So,
    ``id(optimized_array)`` is different from that of ``id(a)``.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = vp.random.rand(1000, 1000)
    >>> x_opt = vp.sca.convert_optimized_array(x, dtype='f8')
    >>> x_opt.strides
    (8008, 8)
    >>> (x == x_opt).all()
    array(True)
    """
    a = nlcpy.asarray(a)
    cdef int64_t ndim = a.ndim
    if ndim == 0 or ndim > 4:
        raise ValueError('input array has invalid dimension: '
                         'got `{}`, expected `1 <= ndim <= 4`.'
                         .format(ndim))

    cdef int64_t nx
    cdef int64_t ny
    cdef int64_t nz
    cdef int64_t nw
    nx = a._shape[ndim-1]
    if ndim > 1:
        ny = a._shape[ndim-2]
    else:
        ny = 1
    if ndim > 2:
        nz = a._shape[ndim-3]
    else:
        nz = 1
    if ndim > 3:
        nw = a._shape[ndim-4]
    else:
        nw = 1

    mx = numpy.empty(1, dtype='i8')
    my = numpy.empty(1, dtype='i8')
    mz = numpy.empty(1, dtype='i8')

    if dtype is None:
        if a.dtype in (nlcpy.dtype('f4'), nlcpy.dtype('f8')):
            dtype = a.dtype
        else:
            dtype = 'float64'
    dt = numpy.dtype(dtype)
    if dt == numpy.dtype('float32'):
        func_name = 'nlcpy_sca_utility_optimize_leading_s'
    elif dt == numpy.dtype('float64'):
        func_name = 'nlcpy_sca_utility_optimize_leading_d'
    else:
        raise TypeError('dtype is only acceptable `float32` or `float64`')
    fpe_flags = request._get_fpe_flag()
    args = (
        <int64_t>nx,
        <int64_t>ny,
        <int64_t>nz,
        veo.OnStack(mx, inout=veo.INTENT_OUT),
        veo.OnStack(my, inout=veo.INTENT_OUT),
        veo.OnStack(mz, inout=veo.INTENT_OUT),
        veo.OnStack(fpe_flags, inout=veo.INTENT_OUT),
    )
    request._push_and_flush_request(
        func_name,
        args,
        sync=True
    )

    cdef vector[Py_ssize_t] out_shape
    out_shape.resize(ndim, 0)
    out_slice = []

    if ndim == 1:
        out_shape[0] = nx
        out_slice.append(slice(None))
    elif ndim == 2:
        out_shape[1] = int(mx)
        out_shape[0] = ny
        out_slice.append(slice(None))
        out_slice.append(slice(0, nx))
    elif ndim == 3:
        out_shape[2] = int(mx)
        out_shape[1] = int(my)
        out_shape[0] = nz
        out_slice.append(slice(None))
        out_slice.append(slice(0, ny))
        out_slice.append(slice(0, nx))
    elif ndim == 4:
        out_shape[3] = int(mx)
        out_shape[2] = int(my)
        out_shape[1] = int(mz)
        out_shape[0] = nw
        out_slice.append(slice(None))
        out_slice.append(slice(0, nz))
        out_slice.append(slice(0, ny))
        out_slice.append(slice(0, nx))

    base = nlcpy.zeros(out_shape, dtype=dt)
    out = base[out_slice]
    out[...] = a[...]
    return out


cpdef ndarray create_optimized_array(shape, dtype='float64'):
    """Creates an optimized ndarray, whose strides are adjusted to improve perfomance,
    filled with zeros.

    Parameters
    ----------
    shape : int or tuple of ints
        The shape to be used by stencil calculations.
    dtype : str or dtype, optional
        The type of the output array. If *dtype* is not given, the dtype of
        *optimized_array* is set to ``float64``.

    Returns
    -------
    optimized_array : ndarray
        The optimized ndarray.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.sca.create_optimized_array(4, dtype='f4')
    array([0., 0., 0., 0.], dtype=float32)

    >>> shape = (1000, 1000)
    >>> x_opt = vp.sca.create_optimized_array(shape)
    >>> x_opt.strides
    (8008, 8)
    """
    cdef tuple _shape = internal.get_size(shape)
    del shape

    cdef int64_t ndim = len(_shape)
    if ndim == 0 or ndim > 4:
        raise ValueError('input shape has invalid length: '
                         'got `{}`, expected `1 <= len(shape) <= 4`.'
                         .format(ndim))
    if ndim == 1 and _shape[0] == 0:
        raise ValueError('input shape has invalid value: '
                         'got `{}`, expected `shape > 0`.'
                         .format(_shape[0]))

    cdef int64_t nx
    cdef int64_t ny
    cdef int64_t nz
    cdef int64_t nw
    nx = _shape[ndim-1]
    if ndim > 1:
        ny = _shape[ndim-2]
    else:
        ny = 1
    if ndim > 2:
        nz = _shape[ndim-3]
    else:
        nz = 1
    if ndim > 3:
        nw = _shape[ndim-4]
    else:
        nw = 1

    mx = numpy.empty(1, dtype='i8')
    my = numpy.empty(1, dtype='i8')
    mz = numpy.empty(1, dtype='i8')

    dt = numpy.dtype(dtype)
    if dt == numpy.dtype('float32'):
        func_name = 'nlcpy_sca_utility_optimize_leading_s'
    elif dt == numpy.dtype('float64'):
        func_name = 'nlcpy_sca_utility_optimize_leading_d'
    else:
        raise TypeError('dtype is only acceptable `float32` or `float64`')
    fpe_flags = request._get_fpe_flag()
    args = (
        <int64_t>nx,
        <int64_t>ny,
        <int64_t>nz,
        veo.OnStack(mx, inout=veo.INTENT_OUT),
        veo.OnStack(my, inout=veo.INTENT_OUT),
        veo.OnStack(mz, inout=veo.INTENT_OUT),
        veo.OnStack(fpe_flags, inout=veo.INTENT_OUT),
    )

    request._push_and_flush_request(
        func_name,
        args,
        sync=True
    )

    cdef vector[Py_ssize_t] out_shape
    out_shape.resize(ndim, 0)
    out_slice = []

    if ndim == 1:
        out_shape[0] = nx
        out_slice.append(slice(None))
    elif ndim == 2:
        out_shape[1] = int(mx)
        out_shape[0] = ny
        out_slice.append(slice(None))
        out_slice.append(slice(0, nx))
    elif ndim == 3:
        out_shape[2] = int(mx)
        out_shape[1] = int(my)
        out_shape[0] = nz
        out_slice.append(slice(None))
        out_slice.append(slice(0, ny))
        out_slice.append(slice(0, nx))
    elif ndim == 4:
        out_shape[3] = int(mx)
        out_shape[2] = int(my)
        out_shape[1] = int(mz)
        out_shape[0] = nw
        out_slice.append(slice(None))
        out_slice.append(slice(0, nz))
        out_slice.append(slice(0, ny))
        out_slice.append(slice(0, nx))

    base = nlcpy.zeros(out_shape, dtype=dt)
    out = base[out_slice]
    return out


cpdef batch_run(kernels, int iteration):
    if type(kernels) not in (list, tuple):
        raise TypeError('kernels must be `list` or `tuple`')
    code_adrs = numpy.empty(len(kernels), dtype='u8')

    for i, k in enumerate(kernels):
        if type(k) is not kernel:
            raise TypeError('kernels element must be `kernel`')
        code_adrs[i] = k.code_adr

    n_code = len(kernels)
    args = (
        veo.OnStack(code_adrs, inout=veo.INTENT_IN),
        <int64_t>iteration,
        <int64_t>n_code
    )

    request._push_and_flush_request(
        'nlcpy_sca_batch_run',
        args
    )

    return kernels[(iteration - 1) % n_code].out
