#
# * The source code in this file is based on the soure code of CuPy.
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

from nlcpy import veo
from nlcpy import jit
from nlcpy import venode


class CustomVEKernel:
    """Custom VE kernel class.

    This class provides simple wrapper functionality related to execution
    of a VE function.

    Parameters
    ----------
    func : nlcpy.veo.VeoFunction
        VE function.

    Note
    ----
    The instance of this class can be retrieved from
    :meth:`nlcpy.jit.CustomVELibrary.get_function`.
    """

    def __init__(self, func, ve_lib):
        if type(func) is not veo.VeoFunction:
            raise TypeError('func must be given VeoFunction.')
        if type(ve_lib) is not jit.CustomVELibrary:
            raise TypeError('ve_lib must be given CustomVELibrary.')
        self._func = func
        self._ve_lib = ve_lib

    def __call__(self, *args, callback=None, sync=False):
        """Invokes the VE function.

        Parameters
        ----------
        *args : variable length arguments
            Arguments of the VE function.
        callback : function
            Callback function that will be executed after the completion of
            the VE function call.
            For details, please refer to the
            :ref:`Callback Setting <label_callback>`.
            Defaults to ``None`` that means do nothing.
        sync : bool
            Whether synchronize function call or not.
            If set to ``True``, this function will return the return value of the
            VE function. The data type of it depends on ``ret_type`` of the
            :meth:`nlcpy.jit.CustomVELibrary.get_function`.
            If set to ``False``, it will return ``None``.
            Defaults to ``False``.

        """

        if not self._ve_lib._is_valid():
            raise RuntimeError('the library is not active.')
        if self._ve_lib._venode != venode.VE():
            raise ValueError('this kernel exists on {}, '
                             'but the current device is set {}'
                             .format(self._ve_lib._venode, venode.VE()))
        res = self._ve_lib._venode.request_manager._push_and_flush_request_core(
            self._func,
            args,
            callback=callback,
            sync=sync,
        )
        return res

    def __repr__(self):  # pragma: no cover
        return '<CustomVEKernel({})>'.format(
            ''.join(['func={}'.format(self._func)])
        )
