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
#ifndef VE_REQUEST_H_INCLUDED
#define VE_REQUEST_H_INCLUDED

#include "ve_array.h"

typedef struct binary_op_arguments_tag {
    ve_array x;
    ve_array y;
    ve_array z;
    ve_array w;
    int32_t  where_flag;
    ve_array where;
} binary_op_arguments;

typedef struct unary_op_arguments_tag {
    ve_array x;
    ve_array z;
    ve_array w;
    int32_t  where_flag;
    ve_array where;
} unary_op_arguments;

typedef struct single_op_arguments_tag {
    ve_array x;
} single_op_arguments;

typedef struct take_arguments_tag {
    ve_array src;
    ve_array idx;
    ve_array out;
    uint64_t ldim;
    uint64_t cdim;
    uint64_t rdim;
    uint64_t index_range;
} take_arguments;

typedef struct eye_arguments_tag {
    ve_array out;
    uint64_t n;
    uint64_t m;
    uint64_t k;
} eye_arguments;

typedef struct linspace_arguments_tag {
    ve_array out;
    ve_array start;
    ve_array stop;
    ve_array delta;
    ve_array step;
    uint64_t endpoint;
    ve_array denormal;
} linspace_arguments;

typedef struct tri_arguments_tag {
    ve_array out;
    int64_t k;
} tri_arguments;

typedef struct nonzero_arguments_tag {
    ve_array a;
    ve_array ret;
    uint64_t axis;
} nonzero_arguments;

typedef struct prepare_indexing_arguments_tag {
    ve_array s;
    ve_array reduced_idx;
    uint64_t a_shape_i;
    uint64_t stride;
} prepare_indexing_arguments;

typedef struct scatter_arguments_tag {
    ve_array a_src;
    ve_array a_idx;
    ve_array a_val;
    uint64_t cdim;
    uint64_t rdim;
    uint64_t adim;
} scatter_arguments;

typedef struct where_arguments_tag {
    ve_array out;
    ve_array condition;
    ve_array x;
    ve_array y;
} where_arguments;


typedef struct argfunc_arguments_tag {
    ve_array x;
    ve_array y;
    ve_array z;
    ve_array initial;
    uint64_t corder;
    uint64_t axis;
} argfunc_arguments;

typedef struct reduce_arguments_tag {
    ve_array x;
    ve_array y;
    ve_array w;
    uint64_t axis;
    uint64_t init_flag;
    ve_array initial;
    uint64_t where_flag;
    ve_array where;
    ve_array out;
    uint64_t do_copyto;
} reduce_arguments;

typedef struct accumulate_arguments_tag {
    ve_array x;
    ve_array y;
    ve_array w;
    uint64_t axis;
} accumulate_arguments;

typedef struct outer_arguments_tag {
    ve_array out;
    ve_array x;
    ve_array y;
    uint64_t where_vectorize;
    uint64_t flag_where;
    ve_array where;
    uint64_t flag_bcast;
    ve_array bcast_dim;
    ve_array workspace;
    ve_array bcast_src;
} outer_arguments;

typedef struct tile_arguments_tag {
    ve_array a;
    ve_array b;
} tile_arguments;

typedef struct delete_arguments_tag {
    ve_array input;
    ve_array del_obj;
    uint64_t axis;
    ve_array idx;
    ve_array output;
    ve_array obj_count;
} delete_arguments;

typedef struct insert_arguments_tag {
    ve_array a;
    ve_array obj;
    ve_array values;
    ve_array out;
    int64_t axis;
    ve_array work;
} insert_arguments;

typedef struct repeat_arguments_tag {
    ve_array a;
    ve_array rep;
    int64_t axis;
    ve_array out;
    ve_array aind;
    ve_array info;
} repeat_arguments;

typedef struct roll_arguments_tag {
    ve_array a;
    ve_array shift;
    ve_array axis;
    ve_array work;
    ve_array result;
} roll_arguments;

typedef struct diff_arguments_tag {
    ve_array a;
    uint64_t n;
    uint64_t axis;
    ve_array b;
    ve_array w;
} diff_arguments;

typedef struct copy_op_arguments_tag {
    ve_array x;
    ve_array y;
} copy_op_arguments;

typedef struct copy_masked_op_arguments_tag {
    ve_array x;
    ve_array y;
    ve_array where;
} copy_masked_op_arguments;

typedef struct sca_op_arguments_tag {
    ve_array code;
} sca_op_arguments;

typedef struct sort_multi_op_arguments_tag {
    ve_array x;
    ve_array y;
    ve_array w;
    int32_t stable;
} sort_multi_op_arguments;

typedef struct shuffle_op_arguments_tag {
    ve_array x;
    ve_array idx;
    ve_array work;
    int32_t axis;
} shuffle_op_arguments;

typedef struct clip_op_arguments_tag {
    ve_array a;
    ve_array out;
    ve_array work;
    ve_array amin;
    ve_array amax;
    ve_array where;
} clip_op_arguments;

typedef struct fill_diagonal_op_arguments_tag {
    ve_array a;
    ve_array val;
    int64_t wrap;
} fill_diagonal_op_arguments;

typedef struct block_op_arguments_tag {
    ve_array arrays;
    ve_array out;
    ve_array offsets;
} block_op_arguments;

typedef struct domain_mask_arguments_tag {
    ve_array a;
    ve_array b;
    ve_array arr;
    ve_array out;
} domain_mask_arguments;

typedef struct simple_fnorm_arguments_tag {
    ve_array x;
    ve_array y;
} simple_fnorm_arguments;

typedef struct fnorm_arguments_tag {
    ve_array x;
    ve_array y;
    ve_array work1;
    ve_array work2;
    int64_t axis1;
    int64_t axis2;
} fnorm_arguments;

typedef union ve_arguments_tag{
    binary_op_arguments binary;
    unary_op_arguments unary;
    single_op_arguments single;
    take_arguments take;
    eye_arguments eye;
    linspace_arguments linspace;
    tri_arguments tri;
    nonzero_arguments nonzero;
    prepare_indexing_arguments prepare_indexing;
    scatter_arguments scatter;
    where_arguments where;
    argfunc_arguments argfunc;
    reduce_arguments reduce;
    accumulate_arguments accumulate;
    outer_arguments outer;
    delete_arguments delete_nlcpy;
    insert_arguments insert;
    tile_arguments tile;
    repeat_arguments repeat;
    roll_arguments roll;
    diff_arguments diff;
    copy_op_arguments copy;
    copy_masked_op_arguments copy_masked;
    sca_op_arguments sca;
    sort_multi_op_arguments sort_multi;
    shuffle_op_arguments shuffle;
    clip_op_arguments clip;
    fill_diagonal_op_arguments fill_diagonal;
    block_op_arguments block;
    domain_mask_arguments domain_mask;
    simple_fnorm_arguments simple_fnorm;
    fnorm_arguments fnorm;
    /* creation, manipulation, indexing, and more...  */
} ve_arguments;


typedef struct request_package_tag {
    /* to select what function execute */
    uint64_t funcnum;
    /* to select function type */
    uint64_t functype;
    /* arguments for request */
    ve_arguments arguments;
} request_package;

#define SIZEOF_REQUEST_PACKAGE (int)sizeof(request_package)
#define N_REQUEST_PACKAGE (int)sizeof(request_package) / sizeof(uint64_t)

#endif /* VE_REQUEST_H_INCLUDED */
