#
# * The source code in this file is based on the soure code of CuPy.
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
# # CuPy License #
#
#     Copyright (c) 2015 Preferred Infrastructure, Inc.
#     Copyright (c) 2015 Preferred Networks, Inc.
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy
#     of this software and associated documentation files (the "Software"), to deal
#     in the Software without restriction, including without limitation the rights
#     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#     copies of the Software, and to permit persons to whom the Software is
#     furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included in
#     all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,

# distutils: language = c++

import sys
import threading
import contextlib
import numpy

from nlcpy.kernel_register import ve_kernel_register
from nlcpy.core.core cimport ndarray
from nlcpy.core.core cimport array
from nlcpy.core cimport manipulation
from nlcpy.veo._veo cimport VeoProc
from nlcpy.veo._veo cimport OnStack
from nlcpy.veo._veo cimport INTENT_OUT
from nlcpy.request cimport request
from nlcpy.veosinfo import node_info
from nlcpy.veosinfo import check_node_status
from nlcpy.veosinfo import mem_info
from nlcpy.mempool cimport mempool
from nlcpy import _environment
from cpython.object cimport *
from libcpp.vector cimport vector
from nlcpy._path import LibPath
from nlcpy.veo._veo import get_ve_arch
cimport cython


cdef VENodePool _venode_pool = None
cdef object _thread_local = threading.local()
cdef bool _is_multive = True


cdef class _TLS:
    """ Thread Local Stack for VE devices.

    """
    cdef readonly list _devices

    def __init__(self):
        self._devices = [0]

    @staticmethod
    cdef _TLS get():
        try:
            stack = _thread_local._device_stack
        except AttributeError:
            stack = _TLS()
            _thread_local._device_stack = stack
        return <_TLS>stack

    cdef void push(self, int veid) except *:
        self._devices.append(veid)

    cdef int pop(self) except *:
        self._devices.pop()
        return <int>self._devices[-1]

    cdef void change_top(self, int veid) except *:
        self._devices[-1] = veid


cdef class _CurrentVENode:

    cdef readonly VENode _venode

    def __init__(self):
        self._venode = _get_venode_pool()[0]

    @staticmethod
    cdef _CurrentVENode get():
        try:
            current_venode = _thread_local._current_venode
        except AttributeError:
            current_venode = _CurrentVENode()
            _thread_local._current_venode = current_venode
        return <_CurrentVENode>current_venode

    cdef VENode getve(self):
        return <VENode>self._venode

    cdef void setve(self, VENode venode):
        self._venode = venode


cpdef _create_venode_pool():
    global _venode_pool, _is_multive
    _venode_pool = VENodePool()
    assert _venode_pool.venode_list[0] == _CurrentVENode.get().getve()
    if len(_venode_pool.venode_list) == 1:
        _is_multive = False

cdef inline _get_venode_pool():
    global _venode_pool
    if _venode_pool is None:
        _create_venode_pool()
    return <VENodePool>_venode_pool

cpdef VENode _get_venode(int veid=0):
    return <VENode>_get_venode_pool()[veid]

cpdef VENode _find_venode_from_proc_handle(uint64_t key_proc):
    vnp = _get_venode_pool()
    for _venode in vnp.venode_list:
        if _venode.connected:
            if key_proc == _venode.proc._proc_handle:
                return <VENode>_venode
    return None


@cython.no_gc
cdef class VENode:
    """ Class that represents VE node.

    Note
    ----
        The instance of this class can be retrieved from :func:`nlcpy.venode.VE`

    Parameters
    ----------
    pid : int
        Physical id.
    lid : int
        Logical id.
    serial_id : int
        Serial id of VE.
    phys : bool
        Whether pass into the argument of veo_proc_create() for physical id or not.
    connect: bool
        Whether call veo_proc_create() when calling constructor or not.

    """

    def __init__(self, int pid, int lid, int serial_id, bool phys, bool connect=True):
        self.pid = pid
        self.lid = lid
        self.serial_id = serial_id
        self._phys = phys
        self._connected = False
        eve_arch = _environment._get_ve_nlcpy_ve_arch()
        if eve_arch:
            self.arch = eve_arch
        else:
            if phys:
                self.arch = get_ve_arch(self.pid)
            else:
                self.arch = get_ve_arch(self.lid)
        self.libpath = LibPath(self.arch)
        self._is_fast_math = _environment._is_fast_math()
        if connect:
            self.connect()

    def connect(self):
        """ Establish connection to VE.
        """

        _environment._set_ve_ld_preload(self.arch)
        _environment._set_ve_ld_library_path(self.libpath._lib_dir)

        if self._connected:
            return
        if self._phys:
            self.proc = VeoProc(self.pid)
        else:
            self.proc = VeoProc(self.lid)
        self.ctx = self.proc.open_context()
        self.lib, self.lib_prof = ve_kernel_register._register_ve_kernel(
            self.proc, self.libpath, self._is_fast_math)
        # alloc_mempool must be called later than kernel register
        self.pool = mempool.MemPool(self)
        self.request_manager = request.RequestManager(self, request.VeoReqs())
        self._connected = True
        # must do after "self._connected = True" to avoid deadlock
        self._initialize_asl_library()
        self._set_constant()

    def disconnect(self):
        """ Disconnect to VE (NotImplemented yet).
        """
        raise NotImplementedError

    def _initialize_asl_library(self):
        fpe = request._get_fpe_flag(self)
        self.request_manager._push_and_flush_request(
            'nlcpy_asl_initialize',
            (OnStack(fpe, inout=INTENT_OUT),),
            sync=True
        )

    def _set_constant(self):
        self.request_manager._push_and_flush_request(
            'nlcpy_set_constant',
            (),
            sync=True
        )

    def __int__(self):
        return self.serial_id

    def __repr__(self):
        return '<VE node logical_id=%d, physical_id=%d, arch=%d>' % \
               (self.lid, self.pid, self.arch)

    def __enter__(self):
        if not self.connected:
            self.connect()
        cdef VENode venode = _CurrentVENode.get().getve()
        cdef int current_id = venode.serial_id
        cdef int next_id = self.serial_id
        _TLS.get().push(next_id)
        if next_id != current_id:
            _CurrentVENode.get().setve(self)
        return self

    def __exit__(self, *args):
        self.request_manager._flush(sync=False)
        # restore VE node from stacked object
        _CurrentVENode.get().setve(_get_venode(_TLS.get().pop()))

    def use(self):
        """ Sets this VE to current active VE.

        Note that combination usage of this method and `with` context manager is
        discouraged.

        Examples
        --------
        >>> import nlcpy
        >>> ve0 = nlcpy.venode.VE(0)
        >>> ve1 = nlcpy.venode.VE(1)
        >>> _ = ve1.use()
        >>> # Current VE is 1.
        >>> with ve0:
        ...     pass
        ...     # Current VE is 0.
        ...     # When exiting the context manager, the current VE is set a top of
        ...     # stacked VE ids (In this case, the top is VE 0).
        >>> # Current VE is 0.
        """

        if not self.connected:
            self.connect()
        _CurrentVENode.get().setve(self)
        return self

    def apply(self):
        """ Sets this VE to current active VE and change a top of stacked VE ids
        to this id.

        Note that combination usage of this method and `with` context manager shows
        different behaviours from :meth:`use`.

        Examples
        --------
        >>> import nlcpy
        >>> ve0 = nlcpy.venode.VE(0)
        >>> ve1 = nlcpy.venode.VE(1)
        >>> _ = ve1.apply()
        >>> # Current VE is 1.
        >>> with ve0:
        ...     pass
        ...     # Current VE is 0.
        ...     # When exiting the context manager, the current VE is set a top of
        ...     # stacked VE ids (In this case, the top is VE 1).
        >>> # Current VE is 1.
        """

        if not self.connected:
            self.connect()
        _CurrentVENode.get().setve(self)
        _TLS.get().change_top(self.serial_id)
        return self

    @property
    def connected(self):
        return self._connected

    @property
    def id(self):
        return self.serial_id

    @property
    def status(self):
        """ Gets the VE status.

        Examples
        --------
        >>> import nlcpy
        >>> from pprint import pprint
        >>> pprint(nlcpy.venode.VE(0).status)  # doctest: +SKIP
        {'arch': 3,
         'fast_math': False,
         'lid': 0,
         'main_mem_used': 5389680640,
         'main_total_memsize': 103079215104,
         'mempool_capacity': 1073741824,
         'mempool_remainder': 1073741824,
         'mempool_used': 0,
         'ncore': 16,
         'offload_timing': 'lazy',
         'pid': 0,
         'running_request_on_VE': 0,
         'stacked_request_on_VH': 0}
        """

        pool_status = self.pool.get_status()
        ve_meminfo = self.meminfo
        return {
            'lid': self.lid,
            'pid': self.pid,
            'arch': self.arch,
            'ncore': self.ncore,
            'fast_math': self._is_fast_math,
            'main_total_memsize': ve_meminfo['kb_main_total'] * 1024,
            'main_mem_used': ve_meminfo['kb_main_used'] * 1024,
            'mempool_capacity': pool_status['pool_capacity'],
            'mempool_used': pool_status['pool_used'],
            'mempool_remainder': pool_status['pool_remainder'],
            'offload_timing': self.request_manager.timing,
            'stacked_request_on_VH': self.request_manager.nreq,
            'running_request_on_VE': self.request_manager.veo_reqs.cnt
        }

    @property
    def ncore(self):
        vni = node_info()
        return vni['cores'][vni['nodeid'].index(self.pid)]

    @property
    def meminfo(self):
        return mem_info(self.pid)

    def _destroy_handle(self):
        try:
            self.request_manager._push_and_flush_request(
                'random_destroy_handle',
                (),
                callback=None,
                sync=True
            )
            self.request_manager._push_and_flush_request(
                'nlcpy_fft_destroy_handle',
                (),
                callback=None,
                sync=True
            )
            self.request_manager._push_and_flush_request(
                'nlcpy_asl_finalize',
                (),
                callback=None,
                sync=True
            )
        except Exception as e:
            pass

    def synchronize(self):
        """ Synchronizes to this VE.

        See Also
        --------
        nlcpy.request.flush : Flushes stacked requests on VH to specified VE.
        """

        if self.connected:
            self.request_manager._flush(sync=True)

    def __richcmp__(VENode self, object other, int op):
        if op == Py_EQ:  # __eq__()
            return isinstance(other, VENode) and self.serial_id == other.serial_id
        if op == Py_NE:  # __ne__()
            return not (isinstance(other, VENode) and self.serial_id == other.serial_id)
        if not isinstance(other, VENode):
            return NotImplemented
        if op == Py_LT:  # __lt__()
            return self.serial_id < other.serial_id
        if op == Py_LE:  # __le__()
            return self.serial_id <= other.serial_id
        if op == Py_GT:  # __gt__()
            return self.serial_id > other.serial_id
        if op == Py_GE:  # __ge__()
            return self.serial_id >= other.serial_id
        return NotImplemented


@cython.no_gc
cdef class VENodePool:

    def __init__(self):
        vni = node_info()
        if _environment._is_mpi():
            if not _environment._is_mpi_initialized():
                try:
                    _mpi4pyve = sys.modules['mpi4pyve']
                except KeyError:
                    raise RuntimeError('NLCPy must be imported after MPI initialization')
                if int(_mpi4pyve.__version__.split('.')[0]) < 1:
                    raise RuntimeError('requires mpi4pyve>=1.0.0')
                else:
                    raise RuntimeError('NLCPy must be imported after MPI initialization')
            # get VE node list
            if _environment._is_venodelist():
                nodelist = _environment._get_venodelist_ids()
                if _environment._is_ve_nlcpy_nodelist():
                    ve_nlcpy_nodelist = _environment._get_ve_nlcpy_nodelist_ids()
                    if max(ve_nlcpy_nodelist) >= len(nodelist):
                        raise ValueError('VE_NLCPY_NODELIST {} is out of range'
                                         .format(max(ve_nlcpy_nodelist)))
                    if min(ve_nlcpy_nodelist) < 0:
                        raise ValueError('VE_NLCPY_NODELIST {} is out of range'
                                         .format(min(ve_nlcpy_nodelist)))
                    nodelist = [nodelist[nid] for nid in ve_nlcpy_nodelist]
            elif _environment._is_ve_nlcpy_nodelist():
                nodelist = _environment._get_ve_nlcpy_nodelist_ids()
            else:
                nodelist = [min(vni['nodeid'])]
            # check node
            for i, nl in enumerate(nodelist):
                if nl not in vni['nodeid']:
                    raise ValueError('VE node id {} is not valid'
                                     .format(nl))
                if check_node_status(nl):
                    raise RuntimeError('VE node {} is offline'.format(nl))
                for _nl in nodelist[i + 1:]:
                    if nl == _nl:
                        raise ValueError('duplicate VE node id {}'.format(nl))
            # process and thread assignment
            local_rank = _environment._get_nmpi_local_rank()
            mpi_local_size = _environment._get_mpi_local_size()
            node_count = {nid: 0 for nid in nodelist}
            logical_node_ids = []
            for i in range(mpi_local_size):
                logical_id = int(len(nodelist) * i / mpi_local_size)
                logical_node_ids.append(logical_id)
                node_count[nodelist[logical_id]] += 1
            myid = logical_node_ids[local_rank]
            ncores = []
            for nid, nc in node_count.items():
                vni_idx = vni['nodeid'].index(nid)
                if nc > vni['cores'][vni_idx]:
                    raise ValueError('The number of process on VE exceeds '
                                     'the number of VE core')
                for i in range(nc):
                    ncores.append((vni['cores'][vni_idx] + i) // nc)
            _environment._set_ve_omp_num_threads(ncores[local_rank])
            if _environment._is_venodelist():
                if _environment._is_ve_nlcpy_nodelist():
                    # _VENODELIST and VE_NLCPY_NODELIST are set,
                    # VE_NLCPY_NODELIST behaves as a logical node number.
                    mypid = nodelist[myid]
                    mylid = ve_nlcpy_nodelist[myid]
                else:
                    # Only _VENODELIST is set.
                    mypid = nodelist[myid]
                    mylid = myid
                is_phys = False
            else:
                # Only VE_NLCPY_NODELIST is set, or neither VE_NLCPY_NODELIST nor
                # _VENODELIST are set.
                mypid = nodelist[myid]
                mylid = local_rank
                is_phys = True
            self.venode_list = [VENode(mypid, mylid, 0, is_phys)]
        elif _environment._is_venodelist():
            # If _VENODELIST is set, veo_proc_create() accepts
            # logical node number of _VENODELIST.
            # e.g) _VENODELIST="1 2":
            #      veo_proc_create(0) -> create VE process on VE Node #1
            #      veo_proc_create(1) -> create VE process on VE Node #2
            nodelist = _environment._get_venodelist_ids()
            for i, nl in enumerate(nodelist):
                if nl not in vni['nodeid']:
                    raise ValueError('_VENODELIST {} is not valid'
                                     .format(nl))
                if check_node_status(nl):
                    raise RuntimeError('VE node {} is offline'.format(nl))
                for _nl in nodelist[i + 1:]:
                    if nl == _nl:
                        raise ValueError('duplicate VE node id {}'.format(nl))
            _environment._set_ve_omp_num_threads()
            if _environment._is_ve_nlcpy_nodelist():
                # If _VENODELIST and VE_NLCPY_NODELIST are set,
                # VE_NLCPY_NODELIST behaves as a logical node number.
                ve_nlcpy_nodelist = _environment._get_ve_nlcpy_nodelist_ids()
                if max(ve_nlcpy_nodelist) >= len(nodelist):
                    raise ValueError('VE_NLCPY_NODELIST {} is out of range'
                                     .format(max(ve_nlcpy_nodelist)))
                if min(ve_nlcpy_nodelist) < 0:
                    raise ValueError('VE_NLCPY_NODELIST {} is out of range'
                                     .format(min(ve_nlcpy_nodelist)))
                for i, nl in enumerate(ve_nlcpy_nodelist):
                    for _nl in ve_nlcpy_nodelist[i + 1:]:
                        if nl == _nl:
                            raise ValueError(
                                'duplicate VE node id {}'.format(nodelist[nl]))
                self.venode_list = [
                    VENode(int(nodelist[lid]), int(lid), int(serial_id), False)
                    for serial_id, lid in enumerate(ve_nlcpy_nodelist)]
            else:
                self.venode_list = [VENode(int(pid), int(lid), int(lid), False)
                                    for lid, pid in enumerate(nodelist)]
        elif _environment._is_ve_nlcpy_nodelist():  # interactive multi VEs
            # In this case, veo_proc_create() accepts physical node number.
            nodelist = _environment._get_ve_nlcpy_nodelist_ids()
            for i, nl in enumerate(nodelist):
                if nl not in vni['nodeid']:
                    raise ValueError('VE_NLCPY_NODELIST {} is not valid'
                                     .format(nl))
                if check_node_status(nl):
                    raise RuntimeError('VE node {} is offline'.format(nl))
                for _nl in nodelist[i + 1:]:
                    if nl == _nl:
                        raise ValueError('duplicate VE node id {}'.format(nl))
            _environment._set_ve_omp_num_threads()
            self.venode_list = [VENode(int(pid), int(lid), int(lid), True)
                                for lid, pid in enumerate(nodelist)]
        elif _environment. _is_ve_node_number():  # interactive single VE
            # If VE_NODE_NUMBER is set, only create a single veo_proc.
            ve_node_number = _environment._get_ve_node_number()
            if ve_node_number not in vni['nodeid']:
                raise ValueError('VE_NODE_NUMBER {} is not valid'
                                 .format(ve_node_number))
            if check_node_status(ve_node_number):
                raise RuntimeError('VE node {} is offline'.format(ve_node_number))
            _environment._set_ve_omp_num_threads()
            self.venode_list = [VENode(ve_node_number, -1, 0, False), ]
        else:
            # VE node numbers are got from veosinfo(ve_node_info()) and create VENode
            # on their number.
            # Only create a veo_proc on logical node number 1.
            available = []
            for nodeid, status in zip(vni['nodeid'], vni['status']):
                if status == 0:
                    available.append(nodeid)
            if len(available) == 0:
                raise RuntimeError('cannot find available VE node')
            available.sort()
            _environment._set_ve_omp_num_threads()
            self.venode_list = [
                VENode(int(pid), int(lid), int(lid), True) if lid == 0
                else VENode(int(pid), int(lid), int(lid), True, connect=False)
                for lid, pid in enumerate(available)]

    def __repr__(self):
        return '<VE node pool {}>'.format(self.venode_list)

    def __getitem__(self, int node_id):
        if not 0 <= node_id < len(self.venode_list):
            raise ValueError('VE node id `{}` is out of range'.format(node_id))
        cdef VENode venode = self.venode_list[node_id]
        return <VENode>venode


_cache_vh = None


cdef _is_cache_reuse(ndarray x_ve, x_vh):
    if x_vh is None:
        return False
    if not isinstance(x_vh, numpy.ndarray):
        raise TypeError
    return (x_ve.shape == x_vh.shape and
            x_ve.dtype == x_vh.dtype and
            x_ve.nbytes == x_vh.nbytes and
            x_ve._c_contiguous == x_vh.flags.c_contiguous and
            x_ve._f_contiguous == x_vh.flags.f_contiguous)


cpdef transfer_array(ndarray src, VENode target_ve, ndarray dst=None):
    """ Transfers N-dimension array to a specified target VE.

    Parameters
    ----------
    src : ndarray
        Array on VE to be transferred.
    target_ve: VENode
        Target VE.
    dst : ndarray, optional
        Array on target VE that receives data from `src`.
        Note that this array must be on `target_ve`.
        If omit this argument, the new array is created on
        target VE.
        If specify this argument, the return array of this
        function is the same to `dst`.

    Returns
    -------
    out : ndarray
        Array on target VE that is transferred from `src`.

    Note
    ----
        This function cannot yet transfer data directly between VEs,
        in other words, the data transfer between VEs has to go through VH.
        If you need to transfer large data, we recommend using
        `mpi4py-ve <https://github.com/SX-Aurora/mpi4py-ve>`_ .

    Examples
    --------
    >>> import nlcpy
    >>> with nlcpy.venode.VE(0):
    ...     x_ve0 = nlcpy.arange(10)
    >>> x_ve1 = nlcpy.venode.transfer_array(x_ve0, nlcpy.venode.VE(1))
    >>> x_ve1
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> x_ve1.venode  # doctest: +SKIP
    <VE node logical_id=1, physical_id=1>
    """

    if src.venode == target_ve:
        prev_ve = VE()
        try:
            target_ve.use()
            if dst is not None:
                if dst.venode != target_ve:
                    raise ValueError('dst({}) does not exist on {}'
                                     .format(dst.venode, target_ve))
                dst[...] = src
                ret = dst
            else:
                ret = src.copy()
        finally:
            prev_ve.use()
        return ret

    # VE -> VH
    global _cache_vh
    if _is_cache_reuse(src, _cache_vh):  # try to reuse VH cache memory
        src.get(order='A', out=_cache_vh)
        host_arr = _cache_vh
    else:
        host_arr = src.get()
        _cache_vh = host_arr

    # VH -> VE
    if dst is None:
        prev_ve = VE()
        try:
            target_ve.use()
            dst = array(host_arr)
        finally:
            prev_ve.use()
    else:
        if dst.venode != target_ve:
            raise ValueError('dst({}) does not exist on {}'
                             .format(dst.venode, target_ve))
        if (host_arr.nbytes == dst.nbytes and
                host_arr.flags.c_contiguous == dst._c_contiguous and
                host_arr.flags.f_contiguous == dst._f_contiguous):
            # write memory without memory allocation
            dst._venode.request_manager._flush(sync=False)
            wreq = dst._venode.ctx.async_write_mem(dst.ve_adr, host_arr.data, dst.nbytes)
            dst._venode.request_manager.veo_reqs._push_req(wreq, callback='nothing')
            # dst._venode.proc.write_mem(dst.ve_adr, host_arr.data, dst.nbytes)
        else:
            # copy array with memory allocation
            prev_ve = VE()
            try:
                target_ve.use()
                manipulation._copyto(
                    dst, array(host_arr), casting='same_kind', where=True)
            finally:
                prev_ve.use()

    return dst


def get_num_available_venodes():
    """ Gets a number of available VE nodes.

    Examples
    --------
    >>> import nlcpy
    >>> nlcpy.venode.get_num_available_venodes()  # doctest: +SKIP
    4
    """

    return len(_get_venode_pool().venode_list)


def synchronize_all_ve():
    """ Synchronizes to all VE.

    See Also
    --------
    nlcpy.venode.VENode.synchronize : Synchronizes to the VE.
    nlcpy.request.flush : Flushes stacked requests on VH to specified VE.
    """

    for ve in _get_venode_pool().venode_list:
        ve.synchronize()


cpdef VENode VE(int veid=-1):
    """ Gets an instance of :class:`nlcpy.venode.VENode`.

    Parameters
    ----------
    veid : int
        VE device id.
        The `veid` must be satisfied with a following condition.

        0 <= `veid` < :func:`get_num_available_venodes`

        if set to ``-1`` (default), the current active VE node is returned.

    Examples
    --------
    >>> import nlcpy
    >>> with nlcpy.venode.VE(0) as ve0:
    ...     # Operations run on VE#0
    ...     nlcpy.array([1, 2, 3]).venode  # doctest: +SKIP
    <VE node logical_id=0, physical_id=0>
    >>> with nlcpy.venode.VE(1) as ve1:
    ...     # Operations run on VE#1
    ...     nlcpy.array([1, 2, 3]).venode  # doctest: +SKIP
    <VE node logical_id=1, physical_id=1>
    """

    if veid == -1:
        return <VENode>_CurrentVENode.get().getve()
    else:
        return <VENode>_get_venode(veid)
