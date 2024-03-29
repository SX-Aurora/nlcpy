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
@#define C_CONTIGUOUS  0
@#define F_CONTIGUOUS  1
@#define KEEP          2
@#define OTHER         3

include(macros.m4)dnl
#define_switch (a->dtype @ b->dtype)

static uint64_t get_max_shape_dim(const ve_array *a){
  uint64_t max_shape = 0;
  uint64_t max_shape_dim;
  for(uint64_t i = 0; i < a->ndim; ++i){
    if(a->shape[i] > max_shape){
      max_shape = a->shape[i];
      max_shape_dim = i;
    }
  }
  return max_shape_dim;
}

static void get_first_coord(uint64_t *coord, const uint64_t N, const uint64_t *shape,
                            uint64_t pos, const uint64_t loop_dim){
  for(uint64_t i = 0; i < N; ++i){
    if(i == loop_dim){
      coord[i] = 0;
    } else {
       coord[i] = pos % shape[i];
       pos /= shape[i];
    }
  }
}

static void get_next_coord(uint64_t *coord, const uint64_t N, const uint64_t *shape,
                           const uint64_t loop_dim){
  for(uint64_t i = 0; i < N; ++i){
    if(i == loop_dim)  continue; // keep 0 on loop_dim
    if(++coord[i] == shape[i]){
      coord[i] = 0;
    } else{
      break;
    }
  }
}

static uint64_t dot(const uint64_t N, const uint64_t *x, const uint64_t *y){
  uint64_t sum = 0;
  for(uint64_t i = 0; i < N; ++i){
    sum += x[i] * y[i];
  }
  return sum;
}

/****************************
 *
 *       OUTER OPERATOR
 *
 * **************************/

define(<--@macro_outer_operator@-->,<--@
uint64_t FILENAME_$1(ve_array *out, const ve_array *a, const ve_array *b, const int32_t where_vectorize)
{
  $2 *pout = ($2 *)out->ve_adr;

#begin_switch
  const @TYPE1@ *pa  = (@TYPE1@ *) a->ve_adr;
  const @TYPE2@ *pb  = (@TYPE2@ *) b->ve_adr;

@#ifdef _OPENMP
  const int nt = omp_get_num_threads();
  const int it = omp_get_thread_num();
@#else
  const int nt = 1;
  const int it = 0;
@#endif /* _OPENMP */
  if (where_vectorize == 0){ // vectorize for input 0 (a)
    const uint64_t str_i = b->size * it / nt;
    const uint64_t end_i = b->size * (it+1) / nt;
    for(uint64_t i = str_i; i < end_i; ++i){
      const @TYPE2@ b_val = pb[i];
#pragma _NEC ivdep
      for(uint64_t j = 0; j < a->size; ++j){
        @BINARY_OPERATOR@(pa[j],b_val,pout[i*a->size + j],$1)
      }
    }
  } else { // vectorize for input 1 (b)
    const uint64_t str_i = a->size * it / nt;
    const uint64_t end_i = a->size * (it+1) / nt;
    for(uint64_t i = str_i; i < end_i; ++i){
      const @TYPE1@ a_val = pa[i];
#pragma _NEC ivdep
      for(uint64_t j = 0; j < b->size; ++j){
        @BINARY_OPERATOR@(a_val,pb[j],pout[i*b->size + j],$1)
      }
    }
  }
#end_switch
  return (uint64_t)NLCPY_ERROR_OK;
}

@-->)dnl

define(<--@macro_broadcast@-->,<--@
static uint64_t broadcast_$1(const ve_array *src, ve_array *out, const ve_array *bcast_dim,
                             uint32_t flag_where, const ve_array *where){
  $2 *pout = ($2 *)out->ve_adr;
  $2 *psrc = ($2 *)src->ve_adr;
  const Bint *pbcast_dim = (Bint *)bcast_dim->ve_adr;
  const uint64_t loop_dim = get_max_shape_dim(out);
  const uint64_t out_stride_loop = out->strides[loop_dim];
  const uint64_t nouter_loop = out->size / out->shape[loop_dim];
  uint64_t src_steps[out->ndim];
  const Bint *pwhere = (Bint *)where->ve_adr;

  uint64_t accumelate_size = 1;
  // src is created to be c_contiguous in python
  for (int k = (int)out->ndim -1; 0 <= k; --k){
    if(pbcast_dim[k]){
      src_steps[k] = 0;
    } else {
      src_steps[k] = accumelate_size;
      accumelate_size *= out->shape[k];
    }
  }

  const uint64_t src_step_loop = src_steps[loop_dim];
@#ifdef _OPENMP
  const int nt = omp_get_num_threads();
  const int it = omp_get_thread_num();
@#else
  const int nt = 1;
  const int it = 0;
@#endif /* _OPENMP */
  const uint64_t str_i = nouter_loop * it / nt;
  const uint64_t end_i = nouter_loop * (it+1) / nt;
  uint64_t coord[out->ndim];
  for(uint64_t i = str_i; i < end_i; ++i){
    // get coord when inner loop starts
    if(i == str_i){
      get_first_coord(coord, out->ndim, out->shape, str_i, loop_dim);
    } else {
      get_next_coord(coord, out->ndim, out->shape, loop_dim);
    }

    const uint64_t src_spos = dot(out->ndim, src_steps, coord);
    const uint64_t out_sadr = (uint64_t)pout + dot(out->ndim, out->strides, coord);
    if(!flag_where){
#pragma _NEC ivdep
      for(uint64_t j = 0; j < out->shape[loop_dim]; ++j){
        *($2 *)(out_sadr + out_stride_loop * j) = psrc[src_spos + src_step_loop * j];
      }
    } else {
      const uint64_t where_step = where->strides[loop_dim]/where->itemsize;
      const uint64_t where_spos = dot(where->ndim, where->strides, coord)/where->itemsize;
#pragma _NEC ivdep
      for(uint64_t j = 0; j < out->shape[loop_dim]; ++j){
        if(pwhere[where_spos + j * where_step]){
          *($2 *)(out_sadr + out_stride_loop * j) = psrc[src_spos + src_step_loop * j];
        }
      }
    }
  }

  return (uint64_t)NLCPY_ERROR_OK;
}
@-->)dnl

#if defined(DTAG_i32)
macro_outer_operator(i32,int32_t)dnl
#endif
#if defined(DTAG_i64)
macro_outer_operator(i64,int64_t)dnl
#endif
#if defined(DTAG_u32)
macro_outer_operator(u32,uint32_t)dnl
#endif
#if defined(DTAG_u64)
macro_outer_operator(u64,uint64_t)dnl
#endif
#if defined(DTAG_f32)
macro_outer_operator(f32,float)dnl
#endif
#if defined(DTAG_f64)
macro_outer_operator(f64,double)dnl
#endif
#if defined(DTAG_c64)
macro_outer_operator(c64,float _Complex)dnl
#endif
#if defined(DTAG_c128)
macro_outer_operator(c128,double _Complex)dnl
#endif
#if defined(DTAG_bool)
macro_outer_operator(bool,int32_t)dnl
#endif
macro_broadcast(i32,int32_t)dnl
macro_broadcast(i64,int64_t)dnl
macro_broadcast(u32,uint32_t)dnl
macro_broadcast(u64,uint64_t)dnl
macro_broadcast(f32,float)dnl
macro_broadcast(f64,double)dnl
macro_broadcast(c64,float _Complex)dnl
macro_broadcast(c128,double _Complex)dnl
macro_broadcast(bool,int32_t)dnl


uint64_t FILENAME(ve_arguments *args, int32_t *psw)
{
  uint64_t err = NLCPY_ERROR_OK;
  ve_array *out = &(args->outer.out);
  ve_array *a = &(args->outer.x);
  ve_array *b = &(args->outer.y);
  int32_t where_vectorize = args->outer.where_vectorize;
  int32_t flag_where = args->outer.flag_where;
  ve_array *where = &(args->outer.where);
  int32_t flag_bcast = args->outer.flag_bcast;
  ve_array *bcast_dim = &(args->outer.bcast_dim);
  ve_array *workspace = &(args->outer.workspace);
  ve_array * bcast_src = &(args->outer.bcast_src);

  switch (workspace->dtype) {
#if defined(DTAG_i32)
  case ve_i32: err = FILENAME_i32(workspace, a, b, where_vectorize); break;
#endif
#if defined(DTAG_i64)
  case ve_i64: err = FILENAME_i64(workspace, a, b, where_vectorize); break;
#endif
#if defined(DTAG_u32)
  case ve_u32: err = FILENAME_u32(workspace, a, b, where_vectorize); break;
#endif
#if defined(DTAG_u64)
  case ve_u64: err = FILENAME_u64(workspace, a, b, where_vectorize); break;
#endif
#if defined(DTAG_f32)
  case ve_f32: err = FILENAME_f32(workspace, a, b, where_vectorize); break;
#endif
#if defined(DTAG_f64)
  case ve_f64: err = FILENAME_f64(workspace, a, b, where_vectorize); break;
#endif
#if defined(DTAG_c64)
  case ve_c64: err = FILENAME_c64(workspace, a, b, where_vectorize); break;
#endif
#if defined(DTAG_c128)
  case ve_c128: err = FILENAME_c128(workspace, a, b, where_vectorize); break;
#endif
#if defined(DTAG_bool)
  case ve_bool: err = FILENAME_bool(workspace, a, b, where_vectorize); break;
#endif
  default: err = NLCPY_ERROR_DTYPE;
  }

  ve_array *temp_pout;
  int32_t temp_where_flag;
  if (flag_bcast){
    temp_pout = bcast_src;
    temp_where_flag = 0; // ignore where
  } else {
    temp_pout = out;
    temp_where_flag = flag_where;
  }

@#ifdef _OPENMP
@#pragma omp barrier
@#endif /* _OPENMP */
  int32_t pswc;
  switch(out->dtype){
#if defined(DTAG_OUT_i32)
  case ve_i32: err |= nlcpy_cast_i32(workspace, temp_pout, temp_where_flag, where, &pswc); break;
#endif
#if defined(DTAG_OUT_i64)
  case ve_i64: err |= nlcpy_cast_i64(workspace, temp_pout, temp_where_flag, where, &pswc); break;
#endif
#if defined(DTAG_OUT_u32)
  case ve_u32: err |= nlcpy_cast_u32(workspace, temp_pout, temp_where_flag, where, &pswc); break;
#endif
#if defined(DTAG_OUT_u64)
  case ve_u64: err |= nlcpy_cast_u64(workspace, temp_pout, temp_where_flag, where, &pswc); break;
#endif
#if defined(DTAG_OUT_f32)
  case ve_f32: err |= nlcpy_cast_f32(workspace, temp_pout, temp_where_flag, where, &pswc); break;
#endif
#if defined(DTAG_OUT_f64)
  case ve_f64: err |= nlcpy_cast_f64(workspace, temp_pout, temp_where_flag, where, &pswc); break;
#endif
#if defined(DTAG_OUT_c64)
  case ve_c64: err |= nlcpy_cast_c64(workspace, temp_pout, temp_where_flag, where, &pswc); break;
#endif
#if defined(DTAG_OUT_c128)
  case ve_c128: err |= nlcpy_cast_c128(workspace, temp_pout, temp_where_flag, where, &pswc); break;
#endif
#if defined(DTAG_OUT_bool)
  case ve_bool: err |= nlcpy_cast_bool(workspace, temp_pout, temp_where_flag, where, &pswc); break;
#endif
  default: err = NLCPY_ERROR_DTYPE;
  }
  *psw |= pswc;

  // broadcast
  if(flag_bcast){
@#ifdef _OPENMP
@#pragma omp barrier
@#endif /* _OPENMP */
    switch(out->dtype){
#if defined(DTAG_OUT_i32)
    case ve_i32: err |= broadcast_i32(bcast_src, out, bcast_dim, flag_where, where); break;
#endif
#if defined(DTAG_OUT_i64)
    case ve_i64: err |= broadcast_i64(bcast_src, out, bcast_dim, flag_where, where); break;
#endif
#if defined(DTAG_OUT_u32)
    case ve_u32: err |= broadcast_u32(bcast_src, out, bcast_dim, flag_where, where); break;
#endif
#if defined(DTAG_OUT_u64)
    case ve_u64: err |= broadcast_u64(bcast_src, out, bcast_dim, flag_where, where); break;
#endif
#if defined(DTAG_OUT_f32)
    case ve_f32: err |= broadcast_f32(bcast_src, out, bcast_dim, flag_where, where); break;
#endif
#if defined(DTAG_OUT_f64)
    case ve_f64: err |= broadcast_f64(bcast_src, out, bcast_dim, flag_where, where); break;
#endif
#if defined(DTAG_OUT_c64)
    case ve_c64: err |= broadcast_c64(bcast_src, out, bcast_dim, flag_where, where); break;
#endif
#if defined(DTAG_OUT_c128)
    case ve_c128: err |= broadcast_c128(bcast_src, out, bcast_dim, flag_where, where); break;
#endif
#if defined(DTAG_OUT_bool)
    case ve_bool: err |= broadcast_bool(bcast_src, out, bcast_dim, flag_where, where); break;
#endif
    default: err = NLCPY_ERROR_DTYPE;
    }
  }

  return (uint64_t)err;
}
