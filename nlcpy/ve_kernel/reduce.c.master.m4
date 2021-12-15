/*
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
*/
@#include <stdio.h>
@#include <stdint.h>
@#include <stdbool.h>
@#include <stdlib.h>
@#include <limits.h>
@#include <alloca.h>
@#include <assert.h>
@#include <float.h>
@#include <math.h>
@#include <complex.h>

@#include "nlcpy.h"

include(macros.m4)dnl
#define_switch (x->dtype)

define(<--@macro_copyto@-->,<--@
static uint64_t reduce_copyto_$1(const ve_array *x, ve_array *y) {
  $2 *py = ($2 *)y->ve_adr;
  switch (x->dtype) {
    case ve_i32:{
      int32_t *px = (int32_t*)x->ve_adr;
      for (int64_t i = 0; i < y->size; i++) {
        py[i] = @CAST_OPERATOR@(px[0],@DTAG1@,$1);
      }
      break;
    }
    case ve_i64:{
      int64_t *px = (int64_t*)x->ve_adr;
      for (int64_t i = 0; i < y->size; i++) {
        py[i] = @CAST_OPERATOR@(px[0],@DTAG1@,$1);
      }
      break;
    }
    case ve_u32:{
      uint32_t *px = (uint32_t*)x->ve_adr;
      for (int64_t i = 0; i < y->size; i++) {
        py[i] = @CAST_OPERATOR@(px[0],@DTAG1@,$1);
      }
      break;
    }
    case ve_u64:{
      uint64_t *px = (uint64_t*)x->ve_adr;
      for (int64_t i = 0; i < y->size; i++) {
        py[i] = @CAST_OPERATOR@(px[0],@DTAG1@,$1);
      }
      break;
    }
    case ve_f32:{
      float *px = (float*)x->ve_adr;
      for (int64_t i = 0; i < y->size; i++) {
        py[i] = @CAST_OPERATOR@(px[0],@DTAG1@,$1);
      }
      break;
    }
    case ve_f64:{
      double *px = (double*)x->ve_adr;
      for (int64_t i = 0; i < y->size; i++) {
        py[i] = @CAST_OPERATOR@(px[0],@DTAG1@,$1);
      }
      break;
    }
    case ve_c64:{
      float _Complex *px = (float _Complex*)x->ve_adr;
      for (int64_t i = 0; i < y->size; i++) {
        py[i] = @CAST_OPERATOR@(px[0],@DTAG1@,$1);
      }
      break;
    }
    case ve_c128:{
      double _Complex *px = (double _Complex*)x->ve_adr;
      for (int64_t i = 0; i < y->size; i++) {
        py[i] = @CAST_OPERATOR@(px[0],@DTAG1@,$1);
      }
      break;
    }
    case ve_bool:{
      int32_t *px = (int32_t*)x->ve_adr;
      for (int64_t i = 0; i < y->size; i++) {
        py[i] = @CAST_OPERATOR@(px[0],@DTAG1@,$1);
      }
      break;
    }
     default: return (uint64_t)NLCPY_ERROR_DTYPE;
  }
  return (uint64_t)NLCPY_ERROR_OK;
}
@-->)dnl

macro_copyto(i32,int32_t)dnl
macro_copyto(i64,int64_t)dnl
macro_copyto(u32,uint32_t)dnl
macro_copyto(u64,uint64_t)dnl
macro_copyto(f32,float)dnl
macro_copyto(f64,double)dnl
macro_copyto(c64,float _Complex)dnl
macro_copyto(c128,double _Complex)dnl
macro_copyto(bool,int32_t)dnl

/****************************
 *
 *       REDUCE OPERATOR
 *
 * **************************/

uint64_t FILENAME(ve_arguments *args, int32_t *psw)
{
    uint64_t err = NLCPY_ERROR_OK;
    ve_array *x = &(args->reduce.x);
    ve_array *y = &(args->reduce.y);
    ve_array *w = &(args->reduce.w);
    int32_t axis = args->reduce.axis;
    int32_t init_flag = args->reduce.init_flag;
    ve_array *initial = &(args->reduce.initial);
    int32_t where_flag = args->reduce.where_flag;
    ve_array *where = &(args->reduce.where);
    ve_array *out = &(args->reduce.out);
    int32_t do_copyto = args->reduce.do_copyto;

    if (axis >= 0) {
#begin_switch
        if (y->dtype==w->dtype) {
            switch (y->dtype) {
#if defined(DTAG_i32)
            case ve_i32:  err = FILENAME_@DTAG1@_i32 (x, y, axis, init_flag, initial, where_flag, where, psw); break;
#endif
#if defined(DTAG_i64)
            case ve_i64:  err = FILENAME_@DTAG1@_i64 (x, y, axis, init_flag, initial, where_flag, where, psw); break;
#endif
#if defined(DTAG_u32)
            case ve_u32:  err = FILENAME_@DTAG1@_u32 (x, y, axis, init_flag, initial, where_flag, where, psw); break;
#endif
#if defined(DTAG_u64)
            case ve_u64:  err = FILENAME_@DTAG1@_u64 (x, y, axis, init_flag, initial, where_flag, where, psw); break;
#endif
#if defined(DTAG_f32)
            case ve_f32:  err = FILENAME_@DTAG1@_f32 (x, y, axis, init_flag, initial, where_flag, where, psw); break;
#endif
#if defined(DTAG_f64)
            case ve_f64:  err = FILENAME_@DTAG1@_f64 (x, y, axis, init_flag, initial, where_flag, where, psw); break;
#endif
#if defined(DTAG_c64)
            case ve_c64:  err = FILENAME_@DTAG1@_c64 (x, y, axis, init_flag, initial, where_flag, where, psw); break;
#endif
#if defined(DTAG_c128)
            case ve_c128: err = FILENAME_@DTAG1@_c128(x, y, axis, init_flag, initial, where_flag, where, psw); break;
#endif
#if defined(DTAG_bool)
            case ve_bool: err = FILENAME_@DTAG1@_bool(x, y, axis, init_flag, initial, where_flag, where, psw); break;
#endif
            default: err = NLCPY_ERROR_DTYPE;
            }

        } else {
            switch (w->dtype) {
#if defined(DTAG_i32)
            case ve_i32:  err = FILENAME_@DTAG1@_i32 (x, w, axis, init_flag, initial, where_flag, where, psw); break;
#endif
#if defined(DTAG_i64)
            case ve_i64:  err = FILENAME_@DTAG1@_i64 (x, w, axis, init_flag, initial, where_flag, where, psw); break;
#endif
#if defined(DTAG_u32)
            case ve_u32:  err = FILENAME_@DTAG1@_u32 (x, w, axis, init_flag, initial, where_flag, where, psw); break;
#endif
#if defined(DTAG_u64)
            case ve_u64:  err = FILENAME_@DTAG1@_u64 (x, w, axis, init_flag, initial, where_flag, where, psw); break;
#endif
#if defined(DTAG_f32)
            case ve_f32:  err = FILENAME_@DTAG1@_f32 (x, w, axis, init_flag, initial, where_flag, where, psw); break;
#endif
#if defined(DTAG_f64)
            case ve_f64:  err = FILENAME_@DTAG1@_f64 (x, w, axis, init_flag, initial, where_flag, where, psw); break;
#endif
#if defined(DTAG_c64)
            case ve_c64:  err = FILENAME_@DTAG1@_c64 (x, w, axis, init_flag, initial, where_flag, where, psw); break;
#endif
#if defined(DTAG_c128)
            case ve_c128: err = FILENAME_@DTAG1@_c128(x, w, axis, init_flag, initial, where_flag, where, psw); break;
#endif
#if defined(DTAG_bool)
            case ve_bool: err = FILENAME_@DTAG1@_bool(x, w, axis, init_flag, initial, where_flag, where, psw); break;
#endif
            default: err = NLCPY_ERROR_DTYPE;
            }

@#ifdef _OPENMP
@#pragma omp barrier
@#endif /* _OPENMP */

            int32_t pswc;
            where_flag = 0;
            switch (y->dtype) {
/* #if defined(DTAG_OUT_i32) */
            case ve_i32:  err |= nlcpy_cast_i32 (w, y, where_flag, where, &pswc); break;
/* #endif */
/* #if defined(DTAG_OUT_i64) */
            case ve_i64:  err |= nlcpy_cast_i64 (w, y, where_flag, where, &pswc); break;
/* #endif */
/* #if defined(DTAG_OUT_u32) */
            case ve_u32:  err |= nlcpy_cast_u32 (w, y, where_flag, where, &pswc); break;
/* #endif */
/* #if defined(DTAG_OUT_u64) */
            case ve_u64:  err |= nlcpy_cast_u64 (w, y, where_flag, where, &pswc); break;
/* #endif */
/* #if defined(DTAG_OUT_f32) */
            case ve_f32:  err |= nlcpy_cast_f32 (w, y, where_flag, where, &pswc); break;
/* #endif */
/* #if defined(DTAG_OUT_f64) */
            case ve_f64:  err |= nlcpy_cast_f64 (w, y, where_flag, where, &pswc); break;
/* #endif */
/* #if defined(DTAG_OUT_c64) */
            case ve_c64:  err |= nlcpy_cast_c64 (w, y, where_flag, where, &pswc); break;
/* #endif */
/* #if defined(DTAG_OUT_c128) */
            case ve_c128: err |= nlcpy_cast_c128(w, y, where_flag, where, &pswc); break;
/* #endif */
/* #if defined(DTAG_OUT_bool) */
            case ve_bool: err |= nlcpy_cast_bool(w, y, where_flag, where, &pswc); break;
/* #endif */
            default: err = NLCPY_ERROR_DTYPE;
            }
            *psw |= pswc;
        }
#end_switch
    }

@#ifdef _OPENMP
@#pragma omp barrier
@#endif /* _OPENMP */
    if (do_copyto) {
        int32_t pswc;
        if (y->size != out->size) {
            switch(out->dtype) {
                case ve_i32:  err |= reduce_copyto_i32 (y, out); break;
                case ve_i64:  err |= reduce_copyto_i64 (y, out); break;
                case ve_u32:  err |= reduce_copyto_u32 (y, out); break;
                case ve_u64:  err |= reduce_copyto_u64 (y, out); break;
                case ve_f32:  err |= reduce_copyto_f32 (y, out); break;
                case ve_f64:  err |= reduce_copyto_f64 (y, out); break;
                case ve_c64:  err |= reduce_copyto_c64 (y, out); break;
                case ve_c128:  err |= reduce_copyto_c128 (y, out); break;
                case ve_bool:  err |= reduce_copyto_bool (y, out); break;
                default: err = NLCPY_ERROR_DTYPE;
            }
        } else {
            switch(out->dtype) {
                case ve_i32:  err |= nlcpy_cast_i32 (y, out, 0, where, &pswc); break;
                case ve_i64:  err |= nlcpy_cast_i64 (y, out, 0, where, &pswc); break;
                case ve_u32:  err |= nlcpy_cast_u32 (y, out, 0, where, &pswc); break;
                case ve_u64:  err |= nlcpy_cast_u64 (y, out, 0, where, &pswc); break;
                case ve_f32:  err |= nlcpy_cast_f32 (y, out, 0, where, &pswc); break;
                case ve_f64:  err |= nlcpy_cast_f64 (y, out, 0, where, &pswc); break;
                case ve_c64:  err |= nlcpy_cast_c64 (y, out, 0, where, &pswc); break;
                case ve_c128:  err |= nlcpy_cast_c128 (y, out, 0, where, &pswc); break;
                case ve_bool:  err |= nlcpy_cast_bool (y, out, 0, where, &pswc); break;
                default: err = NLCPY_ERROR_DTYPE;
            }
        }
        *psw |= pswc;
    }

    return (uint64_t)err;
}
