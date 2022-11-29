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

# numpy dtype character codes
# ? ... bool
# b ... int8
# h ... int16
# i ... int32
# l ... int64
# q ... int64
# B ... uint8
# H ... uint16
# I ... uint32
# L ... uint64
# Q ... uint64
# e ... float16
# f ... float32
# d ... float64
# F ... complex64
# D ... complex128


_add_types = (
    '??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
    'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d', 'gg->g', 'FF->F', 'DD->D', 'GG->G',
    'Mm->M', 'mm->m', 'mM->M', 'OO->O'
)

_subtract_types = (
    '??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I',
    'll->l', 'LL->L', 'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d',
    'gg->g', 'FF->F', 'DD->D', 'GG->G', 'Mm->M', 'mm->m', 'MM->m',
    'OO->O'
)

_multiply_types = (
    '??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
    'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d', 'gg->g', 'FF->F', 'DD->D', 'GG->G',
    'mq->m', 'qm->m', 'md->m', 'dm->m', 'OO->O'
)

_divide_types = (
    'bb->d', 'BB->d', 'hh->d', 'HH->d', 'ii->d', 'II->d', 'll->d',
    'LL->d', 'qq->d', 'QQ->d', 'ee->e', 'ff->f', 'dd->d', 'FF->F',
    'DD->D'
)

_true_divide_types = (
    'bb->d', 'BB->d', 'hh->d', 'HH->d', 'ii->d', 'II->d', 'll->d',
    'LL->d', 'qq->d', 'QQ->d', 'ee->e', 'ff->f', 'dd->d', 'FF->F',
    'DD->D'
)

_logaddexp_types = (
    'ee->e', 'ff->f', 'dd->d', 'gg->g'
)

_logaddexp2_types = (
    'ee->e', 'ff->f', 'dd->d', 'gg->g'
)

_bitwise_and_types = (
    '??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I',
    'll->l', 'LL->L', 'qq->q', 'QQ->Q'
)

_bitwise_or_types = (
    '??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I',
    'll->l', 'LL->L', 'qq->q', 'QQ->Q'
)

_bitwise_xor_types = (
    '??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I',
    'll->l', 'LL->L', 'qq->q', 'QQ->Q'
)

_right_shift_types = (
    'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
    'LL->L', 'qq->q', 'QQ->Q'
)

_left_shift_types = (
    'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
    'LL->L', 'qq->q', 'QQ->Q'
)

_greater_types = (
    '??->?', 'bb->?', 'BB->?', 'hh->?', 'HH->?', 'ii->?', 'II->?',
    'll->?', 'LL->?', 'qq->?', 'QQ->?', 'ee->?', 'ff->?', 'dd->?',
    'gg->?', 'FF->?', 'DD->?', 'GG->?', 'OO->?', 'mm->?', 'MM->?',
    'OO->O', 'OO->?'
)

_greater_equal_types = (
    '??->?', 'bb->?', 'BB->?', 'hh->?', 'HH->?', 'ii->?', 'II->?',
    'll->?', 'LL->?', 'qq->?', 'QQ->?', 'ee->?', 'ff->?', 'dd->?',
    'gg->?', 'FF->?', 'DD->?', 'GG->?', 'OO->?', 'mm->?', 'MM->?',
    'OO->O', 'OO->?'
)

_less_types = (
    '??->?', 'bb->?', 'BB->?', 'hh->?', 'HH->?', 'ii->?', 'II->?',
    'll->?', 'LL->?', 'qq->?', 'QQ->?', 'ee->?', 'ff->?', 'dd->?',
    'gg->?', 'FF->?', 'DD->?', 'GG->?', 'OO->?', 'mm->?', 'MM->?',
    'OO->O', 'OO->?'
)

_less_equal_types = (
    '??->?', 'bb->?', 'BB->?', 'hh->?', 'HH->?', 'ii->?', 'II->?',
    'll->?', 'LL->?', 'qq->?', 'QQ->?', 'ee->?', 'ff->?', 'dd->?',
    'gg->?', 'FF->?', 'DD->?', 'GG->?', 'OO->?', 'mm->?', 'MM->?',
    'OO->O', 'OO->?'
)

_not_equal_types = (
    '??->?', 'bb->?', 'BB->?', 'hh->?', 'HH->?', 'ii->?', 'II->?',
    'll->?', 'LL->?', 'qq->?', 'QQ->?', 'ee->?', 'ff->?', 'dd->?',
    'gg->?', 'FF->?', 'DD->?', 'GG->?', 'OO->?', 'mm->?', 'MM->?',
    'OO->O', 'OO->?'
)

_equal_types = (
    '??->?', 'bb->?', 'BB->?', 'hh->?', 'HH->?', 'ii->?', 'II->?',
    'll->?', 'LL->?', 'qq->?', 'QQ->?', 'ee->?', 'ff->?', 'dd->?',
    'gg->?', 'FF->?', 'DD->?', 'GG->?', 'OO->?', 'mm->?', 'MM->?',
    'OO->O', 'OO->?'
)

_logical_and_types = (
    '??->?', 'bb->?', 'BB->?', 'hh->?', 'HH->?', 'ii->?', 'II->?',
    'll->?', 'LL->?', 'qq->?', 'QQ->?', 'ee->?', 'ff->?', 'dd->?',
    'gg->?', 'FF->?', 'DD->?', 'GG->?', 'OO->O', 'OO->?'
)

_logical_or_types = (
    '??->?', 'bb->?', 'BB->?', 'hh->?', 'HH->?', 'ii->?', 'II->?',
    'll->?', 'LL->?', 'qq->?', 'QQ->?', 'ee->?', 'ff->?', 'dd->?',
    'gg->?', 'FF->?', 'DD->?', 'GG->?', 'OO->O', 'OO->?'
)

_logical_xor_types = (
    '??->?', 'bb->?', 'BB->?', 'hh->?', 'HH->?', 'ii->?', 'II->?',
    'll->?', 'LL->?', 'qq->?', 'QQ->?', 'ee->?', 'ff->?', 'dd->?',
    'gg->?', 'FF->?', 'DD->?', 'GG->?', 'OO->O'
)

_logical_not_types = (
    '?->?', 'b->?', 'B->?', 'h->?', 'H->?', 'i->?', 'I->?', 'l->?',
    'L->?', 'q->?', 'Q->?', 'e->?', 'f->?', 'd->?', 'g->?', 'F->?',
    'D->?', 'G->?', 'O->O', 'O->?'
)

_minimum_types = (
    '??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
    'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d', 'gg->g', 'FF->F', 'DD->D', 'GG->G',
    'mm->m', 'MM->M', 'OO->O'
)

_maximum_types = (
    '??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
    'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d', 'gg->g', 'FF->F', 'DD->D', 'GG->G',
    'mm->m', 'MM->M', 'OO->O'
)

_fmax_types = (
    '??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
    'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d', 'gg->g', 'FF->F', 'DD->D', 'GG->G',
    'mm->m', 'MM->M', 'OO->O'
)

_fmin_types = (
    '??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
    'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d', 'gg->g', 'FF->F', 'DD->D', 'GG->G',
    'mm->m', 'MM->M', 'OO->O'
)

_sin_types = (
    'e->e', 'f->f', 'f->f', 'd->d', 'g->g', 'F->F', 'D->D', 'G->G', 'O->O'
)

_cos_types = (
    'e->e', 'f->f', 'f->f', 'd->d', 'g->g', 'F->F', 'D->D', 'G->G', 'O->O'
)

_tan_types = (
    'e->e', 'f->f', 'd->d', 'g->g', 'F->F', 'D->D', 'G->G', 'O->O'
)

_arcsin_types = (
    'e->e', 'f->f', 'd->d', 'g->g', 'F->F', 'D->D', 'G->G', 'O->O'
)

_arccos_types = (
    'e->e', 'f->f', 'd->d', 'g->g', 'F->F', 'D->D', 'G->G', 'O->O'
)

_arctan_types = (
    'e->e', 'f->f', 'd->d', 'g->g', 'F->F', 'D->D', 'G->G', 'O->O'
)

_sinh_types = (
    'e->e', 'f->f', 'd->d', 'g->g', 'F->F', 'D->D', 'G->G', 'O->O'
)

_cosh_types = (
    'e->e', 'f->f', 'd->d', 'g->g', 'F->F', 'D->D', 'G->G', 'O->O'
)

_tanh_types = (
    'e->e', 'f->f', 'd->d', 'g->g', 'F->F', 'D->D', 'G->G', 'O->O'
)

_arcsinh_types = (
    'e->e', 'f->f', 'd->d', 'g->g', 'F->F', 'D->D', 'G->G', 'O->O'
)

_arccosh_types = (
    'e->e', 'f->f', 'd->d', 'g->g', 'F->F', 'D->D', 'G->G', 'O->O'
)

_arctanh_types = (
    'e->e', 'f->f', 'd->d', 'g->g', 'F->F', 'D->D', 'G->G', 'O->O'
)

_negative_types = (
    '?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I',
    'l->l', 'L->L', 'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d',
    'F->F', 'D->D'
)

_positive_types = (
    '?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I',
    'l->l', 'L->L', 'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d',
    'F->F', 'D->D'
)

_power_types = (
    'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
    'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d', 'gg->g', 'FF->F', 'DD->D',
    'GG->G', 'OO->O'
)

_remainder_types = (
    'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
    'LL->L', 'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d'
)

_mod_types = (
    'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
    'LL->L', 'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d'
)

_fmod_types = (
    '??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I',
    'll->l', 'LL->L', 'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d'
)

_absolute_types = (
    '?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
    'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d', 'g->g', 'm->m', 'F->f', 'D->d',
    'G->g', 'O->O'
)

_fabs_types = (
    'e->e', 'f->f', 'd->d',
)

_rint_types = (
    'e->e', 'f->f', 'd->d', 'f->f', 'd->d', 'g->g', 'F->F', 'D->D',
    'G->G', 'O->O'
)

_sign_types = (
    '?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l',
    'L->L', 'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d', 'g->g', 'F->F',
    'D->D',
)

_heaviside_types = (
    'ee->e', 'ff->f', 'dd->d', 'gg->g'
)

_conjugate_types = (
    'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
    'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d', 'g->g', 'F->F', 'D->D',
    'G->G', 'O->O'
)

_exp_types = (
    'e->e', 'f->f', 'd->d', 'f->f', 'd->d', 'g->g', 'F->F', 'D->D',
    'G->G', 'O->O'
)

_exp2_types = (
    'e->e', 'f->f', 'd->d', 'g->g', 'F->F', 'D->D', 'G->G', 'O->O'
)

_log_types = (
    'e->e', 'f->f', 'd->d', 'g->g', 'F->F', 'D->D', 'G->G', 'O->O'
)

_log2_types = (
    'e->e', 'f->f', 'd->d', 'g->g', 'F->F', 'D->D', 'G->G', 'O->O'
)

_log10_types = (
    'e->e', 'f->f', 'd->d', 'g->g', 'F->F', 'D->D', 'G->G', 'O->O'
)

_expm1_types = (
    'e->e', 'f->f', 'd->d', 'g->g', 'F->F', 'D->D', 'G->G', 'O->O'
)

_log1p_types = (
    'e->e', 'f->f', 'd->d', 'g->g', 'F->F', 'D->D', 'G->G', 'O->O'
)

_sqrt_types = (
    'e->e', 'f->f', 'd->d', 'f->f', 'd->d', 'g->g', 'F->F', 'D->D',
    'G->G', 'O->O'
)

_square_types = (
    'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
    'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d', 'g->g', 'F->F', 'D->D',
    'G->G', 'O->O'
)

_cbrt_types = (
    'e->e', 'f->f', 'd->d',
)

_reciprocal_types = (
    'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
    'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d', 'g->g', 'F->F', 'D->D',
    'G->G', 'O->O'
)

_deg2rad_types = (
    'e->e', 'f->f', 'd->d',
)

_rad2deg_types = (
    'e->e', 'f->f', 'd->d',
)

_degrees_types = (
    'e->e', 'f->f', 'd->d',
)

_radians_types = (
    'e->e', 'f->f', 'd->d',
)

_isfinite_types = (
    '?->?', 'b->?', 'B->?', 'h->?', 'H->?', 'i->?', 'I->?', 'l->?', 'L->?',
    'q->?', 'Q->?', 'e->?', 'f->?', 'd->?', 'g->?', 'F->?', 'D->?', 'G->?',
    'm->?', 'M->?'
)

_isinf_types = (
    '?->?', 'b->?', 'B->?', 'h->?', 'H->?', 'i->?', 'I->?', 'l->?', 'L->?',
    'q->?', 'Q->?', 'e->?', 'f->?', 'd->?', 'g->?', 'F->?', 'D->?', 'G->?',
    'm->?', 'M->?'
)

_isnan_types = (
    '?->?', 'b->?', 'B->?', 'h->?', 'H->?', 'i->?', 'I->?', 'l->?', 'L->?',
    'q->?', 'Q->?', 'e->?', 'f->?', 'd->?', 'g->?', 'F->?', 'D->?', 'G->?',
    'm->?', 'M->?'
)

_signbit_types = (
    'e->?', 'f->?', 'd->?', 'g->?'
)

_copysign_types = (
    'ee->e', 'ff->f', 'dd->d', 'gg->g'
)

_nextafter_types = (
    'ee->e', 'ff->f', 'dd->d', 'gg->g'
)

_spacing_types = (
    'e->e', 'f->f', 'd->d', 'g->g'
)

_ldexp_types = (
    'ei->e', 'fi->f', 'el->e', 'fl->f', 'di->d', 'dl->d', 'gi->g', 'gl->g'
)

_invert_types = (
    '?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l',
    'L->L', 'q->q', 'Q->Q',
)

_floor_types = (
    'e->e', 'f->f', 'd->d',
)

_floor_divide_types = (
    'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
    'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d', 'gg->g', 'FF->F', 'DD->D',
    'GG->G', 'mq->m', 'md->m', 'mm->q', 'OO->O'
)

_ceil_types = (
    'e->e', 'f->f', 'd->d',
)

_trunc_types = (
    'e->e', 'f->f', 'd->d', 'f->f', 'd->d', 'g->g', 'O->O'
)

_fmod_types = (
    'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
    'LL->L', 'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d',
)

_arctan2_types = (
    'ee->e', 'ff->f', 'dd->d',
)

_hypot_types = (
    'ee->e', 'ff->f', 'dd->d',
)
