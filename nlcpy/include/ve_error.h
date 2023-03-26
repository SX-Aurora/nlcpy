/*
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
*/
#ifndef VE_ERROR_H_INCLUDED
#define VE_ERROR_H_INCLUDED

#define NLCPY_ERROR_OK         0b00000000000
#define NLCPY_ERROR_DTYPE      0b00000000001
#define NLCPY_ERROR_NDIM       0b00000000010
#define NLCPY_ERROR_MEMORY     0b00000000100
#define NLCPY_ERROR_FUNCNUM    0b00000001000
#define NLCPY_ERROR_FUNCTYPE   0b00000010000
#define NLCPY_ERROR_INDEX      0b00000100000
#define NLCPY_ERROR_ASL        0b00001000000
#define NLCPY_ERROR_SCA        0b00010000000
#define NLCPY_ERROR_INTERNAL   0b10000000000

#endif /* VE_ERROR_H_INCLUDED */
