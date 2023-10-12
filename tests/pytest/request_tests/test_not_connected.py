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
import pytest
import numpy
from nlcpy import testing
from nlcpy import request
from nlcpy import venode


@testing.multi_ve(2)
class TestNotConnected(unittest.TestCase):

    def test_get_fpe_flag(self):
        ve1 = venode.VE(1)
        if ve1.connected:
            pytest.skip('VE#1 already connected')
        fpe = request.request._get_fpe_flag(venode=ve1)
        assert isinstance(fpe, numpy.ndarray)

    def test_set_max_request(self):
        ve1 = venode.VE(1)
        if ve1.connected:
            pytest.skip('VE#1 already connected')
        prev_num = venode.VE(0).request_manager.max_req
        request.set_max_request(1, venode=ve1)
        assert venode.VE(0).request_manager.max_req == prev_num
        assert venode.VE(1).request_manager.max_req == 1
        request.set_max_request(prev_num, venode=ve1)

    def test_set_offload_timing_onthefly(self):
        ve1 = venode.VE(1)
        if ve1.connected:
            pytest.skip('VE#1 already connected')
        request.set_offload_timing_onthefly(venode=ve1)
        tim = request.get_offload_timing(venode=ve1)
        assert tim == 'on-the-fly'
        request.set_offload_timing_lazy(venode=ve1)
        tim = request.get_offload_timing(venode=ve1)
        assert tim == 'lazy'

    def test_set_offload_timing_lazy(self):
        ve1 = venode.VE(1)
        if ve1.connected:
            pytest.skip('VE#1 already connected')
        request.set_offload_timing_lazy(venode=ve1)
        tim = request.get_offload_timing(venode=ve1)
        assert tim == 'lazy'

    def test_get_offload_timing(self):
        ve1 = venode.VE(1)
        if ve1.connected:
            pytest.skip('VE#1 already connected')
        tim = request.get_offload_timing(venode=ve1)
        assert tim == 'lazy'

    def test_flush(self):
        ve1 = venode.VE(1)
        request.flush(ve1)
