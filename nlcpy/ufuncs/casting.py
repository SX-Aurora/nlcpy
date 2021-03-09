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


_subtract_types = (
    '??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I',
    'll->l', 'LL->L', 'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d',
    'gg->g', 'FF->F', 'DD->D', 'GG->G', 'Mm->M', 'mm->m', 'MM->m',
    'OO->O'
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
    '?->?', 'b->b', 'i->i', 'q->q', 'e->e', 'f->f', 'd->d',
    'F->f', 'D->d',
)

_fabs_types = (
    'e->e', 'f->f', 'd->d',
)

_sign_types = (
    '?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l',
    'L->L', 'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d', 'g->g', 'F->F',
    'D->D',
)

_cbrt_types = (
    'e->e', 'f->f', 'd->d',
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

_invert_types = (
    '?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l',
    'L->L', 'q->q', 'Q->Q',
)

_floor_types = (
    'e->e', 'f->f', 'd->d',
)

_ceil_types = (
    'e->e', 'f->f', 'd->d',
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
