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

_cblas_kernel_list = {
    "wrapper_cblas_sgemm": {
        "ret": "uint64_t",
        "args":
            [
                "int64_t",
                "int64_t",
                "int64_t",
                "int64_t",
                "int64_t",
                "int64_t",
                b"void *",
                "int64_t",
                b"void *",
                "int64_t",
                b"void *",
                "int64_t",
                b"int32_t *",
            ],
    },

    "wrapper_cblas_dgemm": {
        "ret": "uint64_t",
        "args":
            [
                "int64_t",
                "int64_t",
                "int64_t",
                "int64_t",
                "int64_t",
                "int64_t",
                b"void *",
                "int64_t",
                b"void *",
                "int64_t",
                b"void *",
                "int64_t",
                b"int32_t *",
            ],
    },

    "wrapper_cblas_cgemm": {
        "ret": "uint64_t",
        "args":
            [
                "int64_t",
                "int64_t",
                "int64_t",
                "int64_t",
                "int64_t",
                "int64_t",
                b"void *",
                "int64_t",
                b"void *",
                "int64_t",
                b"void *",
                "int64_t",
                b"int32_t *",
            ],
    },

    "wrapper_cblas_zgemm": {
        "ret": "uint64_t",
        "args":
            [
                "int64_t",
                "int64_t",
                "int64_t",
                "int64_t",
                "int64_t",
                "int64_t",
                b"void *",
                "int64_t",
                b"void *",
                "int64_t",
                b"void *",
                "int64_t",
                b"int32_t *",
            ],
    },

    "wrapper_cblas_sdot": {
        "ret": "uint64_t",
        "args":
            [
                b"void *",
                b"void *",
                b"void *",
                b"int32_t *",
            ],
    },

    "wrapper_cblas_ddot": {
        "ret": "uint64_t",
        "args":
            [
                b"void *",
                b"void *",
                b"void *",
                b"int32_t *",
            ],
    },

    "wrapper_cblas_cdotu_sub": {
        "ret": "uint64_t",
        "args":
            [
                b"void *",
                b"void *",
                b"void *",
                b"int32_t *",
            ],
    },

    "wrapper_cblas_zdotu_sub": {
        "ret": "uint64_t",
        "args":
            [
                b"void *",
                b"void *",
                b"void *",
                b"int32_t *",
            ],
    },

}
