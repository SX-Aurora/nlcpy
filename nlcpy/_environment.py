#
# * The source code in this file is developed independently by NEC Corporation.
#
# # NLCPy License #
#
#     Copyright (c) 2020 NEC Corporation
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

import os
import re
from nlcpy import _path


def _set_ve_omp_num_threads(n=None):
    if n is None:
        ve_num_threads = os.environ.get('VE_OMP_NUM_THREADS', None)
    else:
        ve_num_threads = os.environ.get('VE_OMP_NUM_THREADS', str(n))
    if ve_num_threads is not None:
        os.environ['VE_OMP_NUM_THREADS'] = ve_num_threads


def _set_ve_ld_library_path():
    _ve_ld_library_path = os.environ.get('VE_LD_LIBRARY_PATH', '')
    os.environ['VE_LD_LIBRARY_PATH'] = _path._lib_path + ':' + _ve_ld_library_path


# XXX: This is a temporary workaround
# find NCC directory and set VE_LD_PRELOAD
def _set_ve_ld_preload():
    NCC_COMP_VER = '3.0.7'
    base = '/opt/nec/ve/ncc'
    files = os.listdir(base)
    dirs = [os.path.join(base, f)
            for f in files
            if os.path.isdir(os.path.join(base, f))]
    dirs.append(os.path.join(base, NCC_COMP_VER))
    dirs.sort(
        key=lambda s: [int(u) for u in os.path.basename(s).split('.')], reverse=True)
    NCC_PATH = dirs[0]
    if os.path.basename(NCC_PATH) != NCC_COMP_VER:
        _ve_ld_preload = os.environ.get('VE_LD_PRELOAD', '')
        os.environ['VE_LD_PRELOAD'] = NCC_PATH + '/lib/libncc.so.2:' + _ve_ld_preload


def _is_fast_math():
    fast_math = os.environ.get('VE_NLCPY_FAST_MATH', 'no')
    if fast_math in ('yes', 'YES'):
        return True
    else:
        return False


def _get_pool_size():
    _pool_size = os.environ.get('VE_NLCPY_MEMPOOL_SIZE', None)
    if _pool_size:
        m = re.fullmatch(r'^\s*([0-9]+)\s*([BKMGbkmg]?)\s*$', _pool_size)
        if m is None:
            raise ValueError('VE_NLCPY_MEMPOOL_SIZE invalid.')
        g = m.groups()
        if g[1] == '':  # as K
            order = 1024
        elif g[1] in 'bB':
            order = 1
        elif g[1] in 'kK':
            order = 1024
        elif g[1] in 'mM':
            order = 1024 ** 2
        elif g[1] in 'gG':
            order = 1024 ** 3
        else:
            raise ValueError('VE_NLCPY_MEMPOOL_SIZE invalid.')
        _pool_size = int(g[0]) * order
    return _pool_size


def _is_mpi():
    mpirank = os.environ.get('MPIRANK', None)
    if mpirank is None:
        return False
    if mpirank.isdecimal():
        return True
    else:
        return False


def _is_mpi_initialized():
    return int(os.environ.get('_MPI4PYVE_MPI_INITIALIZED', 0))


def _get_mpi_local_size():
    mpi_local_size = os.environ.get('_MPI4PYVE_MPI_LOCAL_SIZE', None)
    if mpi_local_size is None:
        mpi_local_size = os.environ.get('MPISIZE', None)
    if mpi_local_size is None:
        raise ValueError
    return int(mpi_local_size)


def _is_venodelist():
    return os.getenv('_VENODELIST')


def _is_ve_nlcpy_nodelist():
    return os.getenv('VE_NLCPY_NODELIST')


def _is_ve_node_number():
    return os.getenv('VE_NODE_NUMBER')


def _get_ve_node_number():
    return int(os.getenv('VE_NODE_NUMBER'))


def _get_nmpi_local_rank():
    return int(os.environ.get('NMPI_LOCAL_RANK', 0))


def _get_venodelist_ids():
    return list(map(lambda x: int(x), os.environ['_VENODELIST'].split(' ')))


def _get_ve_nlcpy_nodelist_ids():
    return list(map(lambda x: int(x), os.environ['VE_NLCPY_NODELIST'].split(',')))


def _is_numpy_wrap_enabled():
    return not os.environ.get('VE_NLCPY_ENABLE_NUMPY_WRAP') in ('no', 'NO')
