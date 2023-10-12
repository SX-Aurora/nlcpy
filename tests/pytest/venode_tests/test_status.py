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

import unittest
import nlcpy
import os
from nlcpy import _environment
from nlcpy import venode


pool_size = _environment._get_pool_size()
if pool_size is None:
    pool_size = nlcpy.mempool._get_default_pool_size()


def eval(lid, pid, serial_id):
    ve = venode.VE(serial_id).use()
    ve.synchronize()
    nlcpy.request.set_offload_timing_lazy(ve)

    status = ve.status
    if isinstance(lid, list):
        if len(lid) == 2:
            assert (status['lid'] == lid[0]) or (status['lid'] == lid[1])
        else:
            raise IndexError('requires len(lid) <= 2')
    else:
        assert status['lid'] == lid
    assert status['pid'] == pid
    ve_arch = os.environ.get('VE_NLCPY_VE_ARCH', None)
    if ve_arch:
        assert status['arch'] == int(ve_arch)
    assert status['mempool_capacity'] == pool_size
    prev_used = status['mempool_used']
    assert status['mempool_remainder'] == pool_size - prev_used
    assert status['offload_timing'] == 'lazy'
    assert status['stacked_request_on_VH'] == 0
    assert status['running_request_on_VE'] == 0

    x = nlcpy.empty(10, dtype='f8')
    status = ve.status
    assert (status['mempool_used'] == prev_used + 80
            or status['mempool_used'] == prev_used)
    assert (status['mempool_remainder'] == pool_size - 80 - prev_used
            or status['mempool_remainder'] == pool_size - prev_used)

    y = x + 1
    status = ve.status
    assert status['stacked_request_on_VH'] == 1
    assert status['running_request_on_VE'] == 0

    nlcpy.request.flush(ve, sync=False)
    status = ve.status
    assert status['stacked_request_on_VH'] == 0
    assert status['running_request_on_VE'] == 2

    nlcpy.request.flush(ve, sync=True)
    status = ve.status
    assert status['stacked_request_on_VH'] == 0
    assert status['running_request_on_VE'] == 0

    nlcpy.request.set_offload_timing_onthefly(ve)
    _ = y + 1
    status = ve.status
    assert status['offload_timing'] == 'on-the-fly'
    assert status['stacked_request_on_VH'] == 0
    assert status['running_request_on_VE'] == 0

    nlcpy.request.set_offload_timing_lazy(ve)
    _ = nlcpy.arange(10)


class TestStatus(unittest.TestCase):

    def setUp(self):
        self._prev_ve = venode.VE(0)

    def tearDown(self):
        self._prev_ve.apply()

    def test_status_no_environment(self):
        if _environment._is_ve_node_number():
            self.skipTest('This is no environment test')
        if _environment._is_ve_nlcpy_nodelist():
            self.skipTest('This is no environment test')
        if _environment._is_venodelist():
            self.skipTest('This is no environment test')
        if _environment._is_mpi():
            self.skipTest('This is no environment test')
        if _environment._is_mpi():
            self.skipTest('This is no environment test')

        for lid in range(venode.get_num_available_venodes()):
            eval(lid, lid, lid)

        nlcpy.venode.synchronize_all_ve()
        for lid in range(venode.get_num_available_venodes()):
            status = venode.VE(lid).status
            assert status['stacked_request_on_VH'] == 0
            assert status['running_request_on_VE'] == 0

    def test_status_ve_node_number(self):
        if _environment._is_ve_node_number():
            ve_node_number = _environment._get_ve_node_number()
        else:
            self.skipTest('This is VE_NODE_NUMBER test')
        if (_environment._is_venodelist() or
                _environment._is_ve_nlcpy_nodelist()):
            self.skipTest('This is VE_NODE_NUMBER only test')
        if _environment._is_mpi():
            self.skipTest('This is VE_NODE_NUMBER test')

        eval([-1, 0], ve_node_number, 0)

    def test_status_ve_nlcpy_nodelist(self):
        if _environment._is_ve_nlcpy_nodelist():
            ve_nlcpy_nodelist = _environment._get_ve_nlcpy_nodelist_ids()
        else:
            self.skipTest('This is VE_NLCPY_NODELIST test')
        if _environment._is_venodelist():
            self.skipTest('This is VE_NLCPY_NODELIST test')
        if _environment._is_mpi():
            self.skipTest('This is no environment test')

        for lid, pid in enumerate(ve_nlcpy_nodelist):
            eval(lid, pid, lid)

        nlcpy.venode.synchronize_all_ve()
        for lid, pid in enumerate(ve_nlcpy_nodelist):
            status = venode.VE(lid).status
            assert status['stacked_request_on_VH'] == 0
            assert status['running_request_on_VE'] == 0

    def test_status_venodelist(self):
        if _environment._is_venodelist():
            venodelist = _environment._get_venodelist_ids()
        else:
            self.skipTest('This is _VENODELIST test')
        if _environment._is_ve_nlcpy_nodelist():
            self.skipTest('This is _VENODELIST test')
        if _environment._is_mpi():
            self.skipTest('This is no environment test')

        for lid, pid in enumerate(venodelist):
            eval(lid, pid, lid)

        nlcpy.venode.synchronize_all_ve()
        for lid, pid in enumerate(venodelist):
            status = venode.VE(lid).status
            assert status['stacked_request_on_VH'] == 0
            assert status['running_request_on_VE'] == 0

    def test_status_ve_nlcpy_nodelist_and_venodelist(self):
        if _environment._is_ve_nlcpy_nodelist():
            ve_nlcpy_nodelist = _environment._get_ve_nlcpy_nodelist_ids()
        else:
            self.skipTest('This is VE_NLCPY_NODELIST and _VENODELIST test')
        if _environment._is_venodelist():
            venodelist = _environment._get_venodelist_ids()
        else:
            self.skipTest('This is VE_NLCPY_NODELIST and _VENODELIST test')
        if _environment._is_mpi():
            self.skipTest('This is no environment test')

        for serial_id, lid in enumerate(ve_nlcpy_nodelist):
            eval(lid, venodelist[lid], serial_id)

        nlcpy.venode.synchronize_all_ve()
        for serial_id, lid in enumerate(ve_nlcpy_nodelist):
            status = venode.VE(serial_id).status
            assert status['stacked_request_on_VH'] == 0
            assert status['running_request_on_VE'] == 0

    def test_status_mpi(self):
        if not _environment._is_mpi():
            self.skipTest('This is MPI test')
        ve = nlcpy.venode.VE()
        eval(ve.lid, ve.pid, ve.serial_id)

        nlcpy.venode.synchronize_all_ve()
        status = venode.VE().status
        assert status['stacked_request_on_VH'] == 0
        assert status['running_request_on_VE'] == 0
