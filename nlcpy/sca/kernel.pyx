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

from nlcpy import veo
from nlcpy.core cimport core
from nlcpy.core.core cimport ndarray
from nlcpy.sca cimport utility
from nlcpy.sca.description cimport description
from nlcpy.request cimport request

from libc.stdint cimport *

import numpy
cimport numpy as cnp


cdef class kernel:
    """Class for using SCA.

    Refer to :func:`nlcpy.sca.create_kernel`, :func:`nlcpy.sca.destroy_kernel` for more
    details.

    See Also
    --------
    nlcpy.sca.create_kernel : Creates a SCA kernel.
    nlcpy.sca.destroy_kernel : Destroy a SCA kernel.
    """

    def __init__(self, uint64_t code_adr, description desc_i, description desc_o):
        self.code_adr = code_adr
        self.out = desc_o.elems[0].array
        self.desc_i = desc_i
        self.desc_o = desc_o
        self.destroyed = False

    def execute(self):
        """Executes the created SCA kernel and returns the result of stencil computations.

        For usage of this function, see :ref:`Basic Usage <label_sca_basic>` and
        :ref:`Advanced Usage <label_sca_advanced>`.

        See Also
        --------
        nlcpy.sca.create_kernel : Creates a SCA kernel.
        nlcpy.sca.destroy_kernel : Destroy a SCA kernel.

        """
        if self.destroyed:
            raise RuntimeError('this kernel has already been destroyed.')

        request._push_request(
            "nlcpy_sca_code_execute",
            "sca_op",
            (self.code_adr,),
        )

        return self.out

    def _destroy(self):
        if self.destroyed:
            raise RuntimeError('this kernel has already been destroyed.')
        fpe_flags = request._get_fpe_flag()
        args = (
            self.code_adr,
            veo.OnStack(fpe_flags, inout=veo.INTENT_OUT),
        )
        request._push_and_flush_request(
            'nlcpy_sca_code_destroy',
            args,
        )
        self.code_adr = 0
        self.out = None
        self.desc_i = None
        self.desc_o = None
        self.destroyed = True

    def __dealloc__(self):
        # exclude True and None
        if self.destroyed is not False:
            return
        self._destroy()
