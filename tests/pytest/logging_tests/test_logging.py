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
import numpy
from nlcpy import logging
from nlcpy import request
from nlcpy import testing
from nlcpy import venode
from tempfile import NamedTemporaryFile


loggers = (
    logging.REQUEST,
    logging.VEO,
    logging.MEMPOOL,
    logging.NPWRAP,
    logging.FFT,
)


class TestLogging(unittest.TestCase):

    def tearDown(self):
        for _logger in loggers:
            logging.reset_handler(_logger)
        request.set_offload_timing_lazy()

    def test_stream_handler(self):
        for _logger in loggers:
            logging.set_stream_handler(_logger)

        rin = nlcpy.random.rand(10, 10) * .1
        _ = nlcpy.fft.rfft2(rin)

        request.set_offload_timing_onthefly()
        mempool_size = venode.VE().status['mempool_capacity']
        nlcpy.ones(mempool_size // 8 + 8, dtype='f8')

        src = nlcpy.arange(10, dtype='f8')
        dst = numpy.empty(src.shape, dtype=src.dtype)
        ve = venode.VE()
        ve.synchronize()
        req = ve.ctx.async_read_mem(dst.data, src.ve_adr, src.nbytes)
        ret = req.wait_result()
        assert ret == 0
        testing.assert_array_equal(src, dst)

        addr = ve.proc.alloc_mem(8)
        ve.proc.free_mem(addr)

    def test_file_handler(self):
        with NamedTemporaryFile() as tmpfile:
            for _logger in loggers:
                logging.set_file_handler(_logger, tmpfile.name)

            rin = nlcpy.random.rand(10, 10) * .1
            _ = nlcpy.fft.rfft2(rin)

            request.set_offload_timing_onthefly()
            nlcpy.ones(10)

    @testing.multi_ve(2)
    def test_logging_veo_proc_init(self):
        for _logger in loggers:
            logging.set_stream_handler(_logger)
        venode.VE(1).connect()
