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
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <limits.h>
#include <alloca.h>
#include <assert.h>

#include "nlcpy.h"
#include <inc_i64/asl.h>

#define DBG_PRT(...)
//#define ERR_PRT(...)
//#define DBG_PRT printf
#define ERR_PRT printf

//#define ASL_RANDOM_ERR (1)


asl_random_t rng;

uint64_t random_generate(ve_array* ve_input){
    asl_error_t err = ASL_ERROR_OK;
    int64_t num = ve_input->size;

#ifdef _OPENMP
    const int nt = omp_get_max_threads();
#else
    const int nt = 1;
#endif /* _OPENMP */
    err = asl_library_set_thread_count(nt);

    switch( ve_input->dtype ){
    case ve_f64:
        {
            double *val = (double *)ve_input->ve_adr;
            err = asl_random_generate_d(rng, num, val);
            if (err != ASL_ERROR_OK ){
                ERR_PRT("[ERR]asl_random_generate_d(hnd=%d, num=%d, val_ptr=%p) ret = 0x%x \n", rng, num, val, err);
                return (uint64_t)err;
            }
        }
        break;
    case ve_f32:
        {
            float *val = (float *)ve_input->ve_adr;
            err = asl_random_generate_s(rng, num, val);
            if (err != ASL_ERROR_OK ){
                ERR_PRT("[ERR]asl_random_generate_s(hnd=%d, num=%d, val_ptr=%p) ret = 0x%x \n", rng, num, val, err);
                return (uint64_t)err;
            }
        }
        break;
    case ve_i32:
    case ve_u32:
        {
            asl_int_t *val = (asl_int_t *)ve_input->ve_adr;
            err = asl_random_generate_i(rng, num, val);
            if (err != ASL_ERROR_OK ){
                ERR_PRT("[ERR]asl_random_generate_i(hnd=%d, num=%d, val_ptr=%p) ret = 0x%x \n", rng, num, val, err);
                return (uint64_t)err;
            }
        }
        break;
    case ve_i64:
    case ve_u64:
        {
            asl_int64_t *val = (asl_int64_t *)ve_input->ve_adr;
            err = asl_random_generate_i(rng, num, val);
            if (err != ASL_ERROR_OK ){
                ERR_PRT("[ERR]asl_random_generate_i(hnd=%d, num=%d, val_ptr=%p) ret = 0x%x \n", rng, num, val, err);
                return (uint64_t)err;
            }
        }
        break;
    default:
        ERR_PRT("[ERR]asl_random_generate_? cant execute dtype=%d \n", ve_input->dtype);
        err = ASL_ERROR_ARGUMENT;
        return (uint64_t)err;
        break;
    }
    return (uint64_t)err;
}

uint64_t nlcpy_random_generate_uniform_f64(void* out, int32_t *psw) {
    //asl_library_set_logging_level(ASL_LOGLEVEL_INFO);
    asl_error_t err = ASL_ERROR_OK;
    int64_t num = ((ve_array*)out)->size;
    if (!asl_random_is_valid(rng)) {
        err = asl_random_create(&rng, ASL_RANDOMMETHOD_MT19937_64);
        if (err != ASL_ERROR_OK ){
            ERR_PRT("[ERR]asl_random_create() ret = 0x%04x \n", err);
            return (uint64_t)err;
        }
    }

    err = asl_random_distribute_uniform(rng);
    if (err != ASL_ERROR_OK ){
        ERR_PRT("[ERR]asl_random_distribute_uniform(hnd=%d) ret = 0x%x \n", rng,err);
        return (uint64_t)err;
    }
    err = random_generate((ve_array*)out);
    retrieve_fpe_flags(psw);
    return err;
}

uint64_t nlcpy_random_generate_integers(void* out, void* work, int64_t low, uint64_t range, int32_t *psw) {
    asl_error_t err = ASL_ERROR_OK;
    int64_t num = ((ve_array*)work)->size;
    uint64_t *work_val = (uint64_t *)((ve_array*)work)->ve_adr;

    if (!asl_random_is_valid(rng)) {
        err = asl_random_create(&rng, ASL_RANDOMMETHOD_MT19937_64);
        if (err != ASL_ERROR_OK ){
            ERR_PRT("[ERR]asl_random_create() ret = 0x%04x \n", err);
            return (uint64_t)err;
        }
    }

#ifdef _OPENMP
    const int nt = omp_get_max_threads();
#else
    const int nt = 1;
#endif /* _OPENMP */
    err = asl_library_set_thread_count(nt);

    err = asl_random_generate_uniform_long_bits(rng, num, work_val);
    if (err != ASL_ERROR_OK ){
        ERR_PRT("[ERR]asl_random_generate_uniform_long_bits(hnd=%d) ret = 0x%x \n", rng,err);
        return (uint64_t)err;
    }

    switch(((ve_array*)out)->dtype){
    case ve_i64:
        {
            int64_t *val= (int64_t *)((ve_array*)out)->ve_adr;
            for (int64_t i=0; i<num; i++){
                val[i] = (int64_t)((int64_t)(work_val[i] % range) + (int64_t)low);
            }
            break;
        }
    case ve_i32:
        {
            int32_t *val= (int32_t *)((ve_array*)out)->ve_adr;
            for (int32_t i=0; i<num; i++){
                val[i] = (int32_t)((int32_t)(work_val[i] % range) + (int32_t)low);
            }
            break;
        }
    default:
        ERR_PRT("[ERR]cant execute dtype=%d \n", ((ve_array*)out)->dtype);
        err = ASL_ERROR_ARGUMENT;
        break;
    }
    retrieve_fpe_flags(psw);
    return err;
}

uint64_t nlcpy_random_generate_unsigned_integers(void* out, void* work, uint64_t low, uint64_t range, int32_t *psw) {
    asl_error_t err = ASL_ERROR_OK;
    int64_t num = ((ve_array*)work)->size;
    uint64_t *work_val = (uint64_t *)((ve_array*)work)->ve_adr;

    if (!asl_random_is_valid(rng)) {
        err = asl_random_create(&rng, ASL_RANDOMMETHOD_MT19937_64);
        if (err != ASL_ERROR_OK ){
            ERR_PRT("[ERR]asl_random_create() ret = 0x%04x \n", err);
            return (uint64_t)err;
        }
    }

#ifdef _OPENMP
    const int nt = omp_get_max_threads();
#else
    const int nt = 1;
#endif /* _OPENMP */
    err = asl_library_set_thread_count(nt);

    err = asl_random_generate_uniform_long_bits(rng, num, work_val);
    if (err != ASL_ERROR_OK ){
        ERR_PRT("[ERR]asl_random_generate_uniform_long_bits(hnd=%d) ret = 0x%x \n", rng,err);
        return (uint64_t)err;
    }

    switch(((ve_array*)out)->dtype){
    case ve_u64:
        {
            uint64_t *val= (uint64_t *)((ve_array*)out)->ve_adr;
            for (int64_t i=0; i<num; i++){
                val[i] = (uint64_t)((uint64_t)(work_val[i] % range) + (uint64_t)low);
            }
            break;
        }
    case ve_u32:
        {
            uint32_t *val= (uint32_t *)((ve_array*)out)->ve_adr;
            for (int32_t i=0; i<num; i++){
                val[i] = (uint32_t)((uint32_t)(work_val[i] % range) + (uint32_t)low);
            }
            break;
        }
    default:
        ERR_PRT("[ERR]cant execute dtype=%d \n", ((ve_array*)out)->dtype);
        err = ASL_ERROR_ARGUMENT;
        break;
    }
    retrieve_fpe_flags(psw);
    return err;
}

uint64_t nlcpy_random_generate_normal_f64(void* out, double m, double s, int32_t *psw){
    asl_error_t err = ASL_ERROR_OK;
    int64_t num = ((ve_array*)out)->size;
    if (!asl_random_is_valid(rng)) {
        err = asl_random_create(&rng, ASL_RANDOMMETHOD_MT19937_64);
        if (err != ASL_ERROR_OK ){
            ERR_PRT("[ERR]asl_random_create() ret = 0x%x \n", err);
            return (uint64_t)err;
        }
    }

//err = asl_random_get_parameter_normal(rng, &m, &s);
    err = asl_random_distribute_normal(rng, m, s);
    if (err != ASL_ERROR_OK ){
        ERR_PRT("[ERR]asl_random_distribute_normal(hnd=%d, m=%lf, s=%lf) ret = 0x%x \n", rng, m, s, err);
        return (uint64_t)err;
    }
    err = random_generate((ve_array*)out);
    retrieve_fpe_flags(psw);
    return err;
}

uint64_t nlcpy_random_generate_gamma_f64(void* out, double a, double b, int32_t *psw){
    asl_error_t err = ASL_ERROR_OK;
    int64_t num = ((ve_array*)out)->size;
    if (!asl_random_is_valid(rng)) {
        err = asl_random_create(&rng, ASL_RANDOMMETHOD_MT19937_64);
        if (err != ASL_ERROR_OK ){
            ERR_PRT("[ERR]asl_random_create() ret = 0x%x \n", err);
            return (uint64_t)err;
        }
    }

    //err = asl_random_get_parameter_gamma(rng, &a, &b);
    err = asl_random_distribute_gamma(rng, a, b);
    if (err != ASL_ERROR_OK ){
        ERR_PRT("[ERR]asl_random_distribute_gamma(hnd=%d, a=%lf, b=%lf) ret = 0x%x \n", rng, a, b, err);
        return (uint64_t)err;
    }
    err = random_generate((ve_array*)out);
    retrieve_fpe_flags(psw);
    return err;
}

uint64_t nlcpy_random_generate_poisson_f64(void* out, double m, int32_t *psw){
    asl_error_t err = ASL_ERROR_OK;
    int64_t num = ((ve_array*)out)->size;
    if (!asl_random_is_valid(rng)) {
        err = asl_random_create(&rng, ASL_RANDOMMETHOD_MT19937_64);
        if (err != ASL_ERROR_OK ){
            ERR_PRT("[ERR]asl_random_create() ret = 0x%x \n", err);
            return (uint64_t)err;
        }
    }

    err = asl_random_distribute_poisson(rng, m);
    if (err != ASL_ERROR_OK ){
        ERR_PRT("[ERR]asl_random_distribute_poisson(hnd=%d, m=%lf) ret = 0x%x \n", rng, m, err);
        return (uint64_t)err;
    }
    err = random_generate((ve_array*)out);
    retrieve_fpe_flags(psw);
    return err;
}

uint64_t nlcpy_random_generate_binomial_f64(void * out, int64_t m, double p, int32_t *psw){
    asl_error_t err = ASL_ERROR_OK;
    int64_t num = ((ve_array*)out)->size;
    if (!asl_random_is_valid(rng)) {
        err = asl_random_create(&rng, ASL_RANDOMMETHOD_MT19937_64);
        if (err != ASL_ERROR_OK ){
            ERR_PRT("[ERR]asl_random_create() ret = 0x%x \n", err);
            return (uint64_t)err;
        }
    }

    err = asl_random_distribute_binomial(rng, m,p);
    if (err != ASL_ERROR_OK ){
        ERR_PRT("[ERR]asl_random_distribute_binomial(hnd=%d,m=%d,p=%lf) ret = 0x%x \n", rng, m, p ,err);
        return (uint64_t)err;
    }
    err = random_generate((ve_array*)out);
    retrieve_fpe_flags(psw);
    return err;
}

uint64_t nlcpy_random_generate_cauchy_f64(void* out, double a, double b, int32_t *psw){
    asl_error_t err = ASL_ERROR_OK;
    int64_t num = ((ve_array*)out)->size;
    if (!asl_random_is_valid(rng)) {
        err = asl_random_create(&rng, ASL_RANDOMMETHOD_MT19937_64);
        if (err != ASL_ERROR_OK ){
            ERR_PRT("[ERR]asl_random_create() ret = 0x%x \n", err);
            return (uint64_t)err;
        }
    }

    err = asl_random_distribute_cauchy(rng, a, b);
    if (err != ASL_ERROR_OK ){
        ERR_PRT("[ERR]asl_random_distribute_cauchy(hnd=%d, a=%lf, b=%lf) ret = 0x%x \n", rng, a, b, err);
        return (uint64_t)err;
    }
    err = random_generate((ve_array*)out);
    retrieve_fpe_flags(psw);
    return err;
}

uint64_t nlcpy_random_generate_exponential_f64(void* out, double m, int32_t *psw){
    asl_error_t err = ASL_ERROR_OK;
    int64_t num = ((ve_array*)out)->size;
    if (!asl_random_is_valid(rng)) {
        err = asl_random_create(&rng, ASL_RANDOMMETHOD_MT19937_64);
        if (err != ASL_ERROR_OK ){
            ERR_PRT("[ERR]asl_random_create() ret = 0x%x \n", err);
            return (uint64_t)err;
        }
    }

    err = asl_random_distribute_exponential(rng, m);
    if (err != ASL_ERROR_OK ){
        ERR_PRT("[ERR]asl_random_distribute_exponential(hnd=%d, m=%lf) ret = 0x%x \n", rng, m, err);
        return (uint64_t)err;
    }
    err = random_generate((ve_array*)out);
    retrieve_fpe_flags(psw);
    return err;
}

uint64_t nlcpy_random_generate_geometric_f64(void* out, double p, int32_t *psw){
    asl_error_t err = ASL_ERROR_OK;
    int64_t num = ((ve_array*)out)->size;
    if (!asl_random_is_valid(rng)) {
        err = asl_random_create(&rng, ASL_RANDOMMETHOD_MT19937_64);
        if (err != ASL_ERROR_OK ){
            ERR_PRT("[ERR]asl_random_create() ret = 0x%x \n", err);
            return (uint64_t)err;
        }
    }

    err = asl_random_distribute_geometric(rng, p);
    if (err != ASL_ERROR_OK ){
        ERR_PRT("[ERR]asl_random_distribute_geometric(hnd=%d, p=%lf) ret = 0x%x \n", rng, p, err);
        return (uint64_t)err;
    }
    err = random_generate((ve_array*)out);
    retrieve_fpe_flags(psw);
    return err;
}

uint64_t nlcpy_random_generate_gumbel_f64(void* out, double a, double b, int32_t *psw){
    asl_error_t err = ASL_ERROR_OK;
    int64_t num = ((ve_array*)out)->size;
    if (!asl_random_is_valid(rng)) {
        err = asl_random_create(&rng, ASL_RANDOMMETHOD_MT19937_64);
        if (err != ASL_ERROR_OK ){
            ERR_PRT("[ERR]asl_random_create() ret = 0x%x \n", err);
            return (uint64_t)err;
        }
    }

    err = asl_random_distribute_gumbel(rng, a, b);
    if (err != ASL_ERROR_OK ){
        ERR_PRT("[ERR]asl_random_distribute_gumbel(hnd=%d, a=%lf, b=%lf) ret = 0x%x \n", rng, a, b, err);
        return (uint64_t)err;
    }
    err = random_generate((ve_array*)out);
    retrieve_fpe_flags(psw);
    return err;
}

uint64_t nlcpy_random_generate_logistic_f64(void* out, double a, double b, int32_t *psw){
    asl_error_t err = ASL_ERROR_OK;
    int64_t num = ((ve_array*)out)->size;
    if (!asl_random_is_valid(rng)) {
        err = asl_random_create(&rng, ASL_RANDOMMETHOD_MT19937_64);
        if (err != ASL_ERROR_OK ){
            ERR_PRT("[ERR]asl_random_create() ret = 0x%x \n", err);
            return (uint64_t)err;
        }
    }

    err = asl_random_distribute_logistic(rng, a, b);
    if (err != ASL_ERROR_OK ){
        ERR_PRT("[ERR]asl_random_distribute_logistic(hnd=%d, a=%lf, b=%lf) ret = 0x%x \n", rng, a, b, err);
        return (uint64_t)err;
    }
    err = random_generate((ve_array*)out);
    retrieve_fpe_flags(psw);
    return err;
}

uint64_t nlcpy_random_generate_lognormal_f64(void* out, double m, double s, int32_t *psw){
    asl_error_t err = ASL_ERROR_OK;
    int64_t num = ((ve_array*)out)->size;
    if (!asl_random_is_valid(rng)) {
        err = asl_random_create(&rng, ASL_RANDOMMETHOD_MT19937_64);
        if (err != ASL_ERROR_OK ){
            ERR_PRT("[ERR]asl_random_create() ret = 0x%x \n", err);
            return (uint64_t)err;
        }
    }

    err = asl_random_distribute_lognormal(rng, m, s);
    if (err != ASL_ERROR_OK ){
        ERR_PRT("[ERR]asl_random_distribute_lognormal(hnd=%d, m=%lf, s=%lf) ret = 0x%x \n", rng, m, s, err);
        return (uint64_t)err;
    }
    err = random_generate((ve_array*)out);
    retrieve_fpe_flags(psw);
    return err;
}

uint64_t nlcpy_random_generate_lognormal_box_muller_f64(void* out, double m, double s, int32_t *psw){
    asl_error_t err = ASL_ERROR_OK;
    int64_t num = ((ve_array*)out)->size;
    if (!asl_random_is_valid(rng)) {
        err = asl_random_create(&rng, ASL_RANDOMMETHOD_MT19937_64);
        if (err != ASL_ERROR_OK ){
            ERR_PRT("[ERR]asl_random_create() ret = 0x%x \n", err);
            return (uint64_t)err;
        }
    }

    err = asl_random_distribute_lognormal_box_muller(rng, m, s);
    if (err != ASL_ERROR_OK ){
        ERR_PRT("[ERR]asl_random_distribute_lognormal_box_muller(hnd=%d, m=%lf, s=%lf) ret = 0x%x \n", rng, m, s, err);
        return (uint64_t)err;
    }
    err = random_generate((ve_array*)out);
    retrieve_fpe_flags(psw);
    return err;
}

uint64_t nlcpy_random_generate_normal_box_muller_f64(void* out, double m, double s, int32_t *psw){
    asl_error_t err = ASL_ERROR_OK;
    int64_t num = ((ve_array*)out)->size;
    if (!asl_random_is_valid(rng)) {
        err = asl_random_create(&rng, ASL_RANDOMMETHOD_MT19937_64);
        if (err != ASL_ERROR_OK ){
            ERR_PRT("[ERR]asl_random_create() ret = 0x%x \n", err);
            return (uint64_t)err;
        }
    }

    err = asl_random_distribute_normal_box_muller(rng, m, s);
    if (err != ASL_ERROR_OK ){
        ERR_PRT("[ERR]asl_random_distribute_normal_box_muller(hnd=%d,m=%lf,s=%lf) ret = 0x%x \n", rng, m ,s, err);
        return (uint64_t)err;
    }
    err = random_generate((ve_array*)out);
    retrieve_fpe_flags(psw);
    return err;
}

uint64_t nlcpy_random_generate_weibull_f64(void* out, double a, double b, int32_t *psw){
    asl_error_t err = ASL_ERROR_OK;
    int64_t num = ((ve_array*)out)->size;
    if (!asl_random_is_valid(rng)) {
        err = asl_random_create(&rng, ASL_RANDOMMETHOD_MT19937_64);
        if (err != ASL_ERROR_OK ){
            ERR_PRT("[ERR]asl_random_create() ret = 0x%x \n", err);
            return (uint64_t)err;
        }
    }

    err = asl_random_distribute_weibull(rng, a, b);
    if (err != ASL_ERROR_OK ){
        ERR_PRT("[ERR]dsl_random_distribute_weibull(hnd=%d, a=%lf, b=%lf) ret = 0x%x \n", rng, a, b, err);
        return (uint64_t)err;
    }
    err = random_generate((ve_array*)out);
    retrieve_fpe_flags(psw);
    return err;
}

uint64_t random_destroy_handle() {
    asl_error_t err = ASL_ERROR_OK;

    if (asl_random_is_valid(rng)) {
        err = asl_random_destroy(rng);
        if (err != ASL_ERROR_OK ){
            ERR_PRT("[ERR]asl_random_destroy() ret = 0x%x \n", err);
            return (uint64_t)err;
        }
    }
    return (uint64_t)err;
}

uint64_t nlcpy_random_set_seed(ve_array *seed, int32_t *psw) {
    asl_error_t err = ASL_ERROR_OK;

    if (!asl_random_is_valid(rng)) {
        err = asl_random_create(&rng, ASL_RANDOMMETHOD_MT19937_64);
        if (err != ASL_ERROR_OK ){
            ERR_PRT("[ERR]asl_random_create() ret = 0x%04x \n", err);
            return (uint64_t)err;
        }
    }

    const asl_uint32_t *sp = (uint32_t *)seed->ve_adr;
    DBG_PRT("[DBG]nlcpy_random_set_seed seed(ptr=%p,size=%d,[0]=%u)\n", sp, seed->size, sp[0]);

    err = asl_random_initialize(rng, seed->size , sp);
#if ASL_RANDOM_ERR==1
    err = ASL_ERROR_ARGUMENT;
#endif //ASL_RANDOM_ERR==1
    if (err != ASL_ERROR_OK ){
        ERR_PRT("[ERR]asl_random_initialize(hnd=%d,len=%d,seed_ptr=%p) ret = 0x%x \n", rng, seed->size , sp ,err);
        return (uint64_t)err;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)err;
}

int64_t nlcpy_random_get_state_size(){
    return asl_random_get_state_size(rng);
}

uint64_t nlcpy_random_save_state(ve_array *state, int32_t *psw){
    asl_error_t err = ASL_ERROR_OK;

    if (!asl_random_is_valid(rng)) {
        err = asl_random_create(&rng, ASL_RANDOMMETHOD_MT19937_64);
        if (err != ASL_ERROR_OK ){
            ERR_PRT("[ERR]asl_random_create() ret = 0x%04x \n", err);
            return (uint64_t)err;
        }
    }

    asl_uint32_t *val = (uint32_t *)state->ve_adr;
    err = asl_random_save_state(rng, val);
#if ASL_RANDOM_ERR==1
    err = ASL_ERROR_ARGUMENT;
#endif //ASL_RANDOM_ERR==1
    if (err != ASL_ERROR_OK ){
        ERR_PRT("[ERR]asl_random_save_state(hnd=%d,val_ptr=%p) ret = 0x%04x \n",rng, val, err);
        return (uint64_t)err;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)err;
}

uint64_t nlcpy_random_restore_state(ve_array *state, int32_t *psw){
    asl_error_t err = ASL_ERROR_OK;

    if (!asl_random_is_valid(rng)) {
        err = asl_random_create(&rng, ASL_RANDOMMETHOD_MT19937_64);
        if (err != ASL_ERROR_OK ){
            ERR_PRT("[ERR]asl_random_create() ret = 0x%04x \n", err);
            return (uint64_t)err;
        }
    }

    const asl_uint32_t *val = (uint32_t *)state->ve_adr;
    err = asl_random_restore_state(rng, val);
#if ASL_RANDOM_ERR==1
    err = ASL_ERROR_ARGUMENT;
#endif //ASL_RANDOM_ERR==1
    if (err != ASL_ERROR_OK ){
        ERR_PRT("[ERR]asl_random_restore_state(hnd=%d,val_ptr=%p) ret = 0x%04x \n",rng, val, err);
        return (uint64_t)err;
    }

    retrieve_fpe_flags(psw);
    return (uint64_t)err;
}
