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

_random_kernel_list = {
    "asl_library_initialize": {
        "ret": "uint64_t",
        "args": ["void"],
    },
    "asl_library_finalize": {
        "ret": "uint64_t",
        "args": ["void"],
    },
    "asl_random_create": {
        "ret": "uint64_t",
        "args": [b"int64_t *", "int"],
    },
    "asl_random_distribute_normal": {
        "ret": "uint64_t",
        "args": ["int64_t", "double", "double"],
    },
    "asl_random_distribute_uniform": {
        "ret": "uint64_t",
        "args": ["int64_t"],
    },
    "asl_random_generate_d": {
        "ret": "uint64_t",
        "args": ["int64_t", "int64_t", "uint64_t"],
    },
    "asl_random_destroy": {
        "ret": "uint64_t",
        "args": ["int64_t"],
    },
    "nlcpy_random_generate_uniform_f64": {
        "ret": "uint64_t",
        "args": [b"void *", b"int32_t *"],
    },
    "nlcpy_random_generate_integers": {
        "ret": "uint64_t",
        "args": [b"void *", b"void *", "int64_t", "uint64_t", b"int32_t *"],
    },
    "nlcpy_random_generate_unsigned_integers": {
        "ret": "uint64_t",
        "args": [b"void *", b"void *", "uint64_t", "uint64_t", b"int32_t *"],
    },
    "nlcpy_random_generate_normal_f64": {
        "ret": "uint64_t",
        "args": [b"void *", "double", "double", b"int32_t *"],
    },
    "nlcpy_random_generate_gamma_f64": {
        "ret": "uint64_t",
        "args": [b"void *", "double", "double", b"int32_t *"],
    },
    "nlcpy_random_generate_poisson_f64": {
        "ret": "uint64_t",
        "args": [b"void *", "double", b"int32_t *"],
    },
    "nlcpy_random_generate_binomial_f64": {
        "ret": "uint64_t",
        "args": [b"void *", "int64_t", "double", b"int32_t *"],
    },
    "nlcpy_random_generate_cauchy_f64": {
        "ret": "uint64_t",
        "args": [b"void *", "double", "double", b"int32_t *"],
    },
    "nlcpy_random_generate_exponential_f64": {
        "ret": "uint64_t",
        "args": [b"void *", "double", b"int32_t *"],
    },
    "nlcpy_random_generate_geometric_f64": {
        "ret": "uint64_t",
        "args": [b"void *", "double", b"int32_t *"],
    },
    "nlcpy_random_generate_gumbel_f64": {
        "ret": "uint64_t",
        "args": [b"void *", "double", "double", b"int32_t *"],
    },
    "nlcpy_random_generate_logistic_f64": {
        "ret": "uint64_t",
        "args": [b"void *", "double", "double", b"int32_t *"],
    },
    "nlcpy_random_generate_lognormal_f64": {
        "ret": "uint64_t",
        "args": [b"void *", "double", "double", b"int32_t *"],
    },
    "nlcpy_random_generate_lognormal_box_muller_f64": {
        "ret": "uint64_t",
        "args": [b"void *", "double", "double", b"int32_t *"],
    },
    "nlcpy_random_generate_normal_box_muller_f64": {
        "ret": "uint64_t",
        "args": [b"void *", "double", "double", b"int32_t *"],
    },
    "nlcpy_random_generate_weibull_f64": {
        "ret": "uint64_t",
        "args": [b"void *", "double", "double", b"int32_t *"],
    },
    "nlcpy_random_shuffle": {
        "ret": "uint64_t",
        "args": [b"void *", b"void *", b"void *", b"int32_t *"],
    },
    "nlcpy_random_set_seed": {
        "ret": "uint64_t",
        "args": [b"void *", b"int32_t *"],
    },
    "random_destroy_handle": {
        "ret": "uint64_t",
        "args": ["void"],
    },
    "nlcpy_random_get_state_size": {
        "ret": "int64_t",
        "args": ["void"],
    },
    "nlcpy_random_save_state": {
        "ret": "uint64_t",
        "args": [b"void *", b"int32_t *"],
    },
    "nlcpy_random_restore_state": {
        "ret": "uint64_t",
        "args": [b"void *", b"int32_t *"],
    },
}
