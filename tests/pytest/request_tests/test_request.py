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
import nlcpy
from nlcpy import testing
from nlcpy import request
from nlcpy import venode
from nlcpy import veo


class TestRequest(unittest.TestCase):

    def test_get_request_manager(self):
        reqm = request.request._get_request_manager()
        assert reqm is venode.VE().request_manager

    def test_get_veo_requests(self):
        vreq = request.request._get_veo_requests()
        assert vreq is venode.VE().request_manager.veo_reqs

    def test_callback_not_callable(self):
        a = nlcpy.array([[3, 1], [1, 2]], order='F')
        b = nlcpy.array([9, 8], order='F')

        info = numpy.empty(1, dtype='l')
        fpe = request.request._get_fpe_flag()
        args = (
            a,
            b,
            veo.OnStack(info, inout=veo.INTENT_OUT),
            veo.OnStack(fpe, inout=veo.INTENT_OUT),
        )

        with pytest.raises(RuntimeError):
            request.request._push_and_flush_request(
                'nlcpy_solve',
                args,
                callback=1
            )

    def test_set_max_request(self):
        prev_num = venode.VE().request_manager.max_req
        request.set_max_request(1)
        assert venode.VE().request_manager.max_req == 1
        request.set_max_request(prev_num)

    def test_above_max_request(self):
        max_req = venode.VE().request_manager.max_req
        x = nlcpy.zeros(10, dtype='i8')
        y = x.copy()
        request.flush()
        for _ in range(max_req + 1):
            x += 1
        testing.assert_array_equal(x, y + max_req + 1)

    def test_invalid_arg(self):
        with pytest.raises(RuntimeError):
            request.request._push_request(
                "nlcpy_arange",
                "creation_op",
                (None, None, None))

    def test_unkown_func_name(self):
        with pytest.raises(RuntimeError):
            request.request._push_request(
                "Foobar",
                "creation_op",
                (None, None, None))

    def test_unkown_func_type(self):
        with pytest.raises(RuntimeError):
            request.request._push_request(
                "nlcpy_arange",
                "Foobar",
                (None, None, None))

    def test_get_offload_timing(self):
        request.set_offload_timing_onthefly()
        tim = request.get_offload_timing()
        assert tim == 'on-the-fly'
        request.set_offload_timing_lazy()
        tim = request.get_offload_timing()
        assert tim == 'lazy'

    def test_above_n_request(self):
        args = [nlcpy.empty(10) for _ in range(100)]
        with pytest.raises(RuntimeError):
            request.request._push_request(
                "nlcpy_arange",
                "creation_op",
                args)

    def test_str_repr(self):
        ve = venode.VE()
        assert ve.request_manager.__str__() is not None
        assert ve.request_manager.__repr__() is not None
        assert ve.request_manager.veo_reqs.__repr__() is not None
