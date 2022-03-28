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

import numpy
import nlcpy


def _casting_check_binary(dtype_out, in_args, name, casting):
    for i, x in enumerate(in_args):
        if not numpy.can_cast(x, dtype_out, casting=casting):
            raise TypeError(
                "Cannot cast ufunc \'{}\' input {} from dtype(\'{}\') "
                "to dtype(\'{}\') with casting rule \'{}\'".format(
                    name, i, x.dtype, dtype_out, casting)
            )


def _casting_check_unary(dtype_out, in_args, name, casting):
    for x in in_args:
        if not numpy.can_cast(x, dtype_out, casting=casting):
            raise TypeError("Cannot cast ufunc \'{}\' input from dtype(\'{}\') "
                            "to dtype(\'{}\') with casting rule \'{}\'".format(
                                name, x.dtype, dtype_out, casting))


def _casting_check_without_msg(dtype_out, in_args, casting):
    for x in in_args:
        if numpy.isscalar(x):
            if not numpy.can_cast(x, dtype_out, casting=casting):
                return False
        else:
            if not numpy.can_cast(x.dtype, dtype_out, casting=casting):
                return False
    return True


def _casting_check_binary_except(dtype_out, in_args, casting):
    if in_args[0].dtype.kind == 'u' or in_args[1].dtype.kind == 'u':
        for x in in_args:
            if not numpy.can_cast(x, dtype_out, casting=casting):
                return False
    else:
        for x in in_args:
            if not numpy.can_cast(x.dtype, dtype_out, casting=casting):
                return False
    return True


def _casting_check_unary_except(dtype_out, in_args, casting):
    if in_args[0].dtype.kind == 'u':
        for x in in_args:
            if not numpy.can_cast(x, dtype_out, casting=casting):
                return False
    else:
        for x in in_args:
            if not numpy.can_cast(x.dtype, dtype_out, casting=casting):
                return False
    return True


def _casting_check_out(dtype_out, out_dtype, name, casting):
    if not numpy.can_cast(dtype_out, out_dtype, casting=casting):
        raise TypeError("ufunc '{}' output (typecode '{}') could not be "
                        "coerced to provided output parameter (typecode '{}') "
                        "according to the casting rule ''{}''".format(
                            name, dtype_out.char, out_dtype.char, casting))


def _casting_check_out_without_msg(dtype_out, out_dtype, casting):
    if numpy.can_cast(dtype_out, out_dtype, casting=casting):
        return True
    else:
        return False


def _raise_no_loop_matching(name):
    raise TypeError("No loop matching the specified signature and casting "
                    "was found for ufunc {}".format(name))


def _raise_nlcpy_original_error(ari_dtype, name):
    raise TypeError("\'{}\' is not supported as a calculation dtype "
                    "of ufunc \'{}\'".format(ari_dtype, name))


def _binary_check_case1(*args):
    dt = args[0]
    in_args = args[1]
    name = args[2]
    casting = args[3]
    # valid = args[4]
    check_cast = args[5]
    check_out = args[6]
    out = args[7]
    if check_cast:
        _casting_check_binary(dt, in_args, name, casting)
    if check_out and not _casting_check_out_without_msg(dt, out.dtype, casting):
        raise TypeError("Cannot cast ufunc \'{}\' output from dtype(\'{}\') "
                        "to dtype(\'{}\') with casting rule \'{}\'".format(
                            name, dt, out.dtype, casting))


def _binary_check_case2(*args):
    dt = args[0]
    in_args = args[1]
    name = args[2]
    casting = args[3]
    valid = args[4]
    check_cast = args[5]
    check_out = args[6]
    out = args[7]
    if check_cast and not _casting_check_without_msg(dt, in_args, casting):
        _raise_no_loop_matching(name)
    if check_out:
        if out.dtype.kind in ('b', 'i', 'u'):
            if not in_args[0].dtype.kind in ('f', 'c') and \
                    not in_args[1].dtype.kind in ('f', 'c'):
                _raise_no_loop_matching(name)
        if check_cast and not numpy.can_cast(dt, out.dtype, casting=casting):
            _raise_no_loop_matching(name)
        else:
            _casting_check_out(dt, out.dtype, name, casting)
    if valid is False or dt.kind in ('b', 'i', 'u'):
        _raise_no_loop_matching(name)


def _binary_check_case3(*args):
    dt = args[0]
    in_args = args[1]
    name = args[2]
    casting = args[3]
    valid = args[4]
    check_cast = args[5]
    check_out = args[6]
    out = args[7]
    if valid is False or numpy.dtype(dt).kind in ('b', 'i', 'u'):
        _raise_no_loop_matching(name)
    if check_cast and not _casting_check_without_msg(dt, in_args, casting):
        _raise_no_loop_matching(name)
    if check_out:
        if check_cast and not numpy.can_cast(dt, out.dtype, casting=casting):
            _raise_no_loop_matching(name)
        else:
            _casting_check_out(dt, out.dtype, name, casting)


def _binary_check_case4(*args):
    dt = args[0]
    in_args = args[1]
    name = args[2]
    casting = args[3]
    valid = args[4]
    check_cast = args[5]
    check_out = args[6]
    out = args[7]
    if dt.kind in ('u'):
        if in_args[0].dtype.kind in ('b') and in_args[1].dtype.kind not in ('b', 'u'):
            _raise_no_loop_matching(name)
        elif in_args[0].dtype.kind not in ('b', 'u') and in_args[1].dtype.kind in ('b'):
            _raise_no_loop_matching(name)
        elif in_args[0].dtype.kind not in ('b', 'u') and \
                in_args[1].dtype.kind not in ('b', 'u'):
            _raise_no_loop_matching(name)
    if check_cast and not _casting_check_without_msg(dt, in_args, casting):
        _raise_no_loop_matching(name)
    if check_out:
        if check_cast and not numpy.can_cast(dt, out.dtype, casting=casting):
            _raise_no_loop_matching(name)
        elif dt not in (numpy.int8,) and out.dtype.kind in ('u'):
            _casting_check_out(dt, out.dtype, name, casting)
        if out.dtype.kind in ('b') or in_args[0].dtype.kind in ('f', 'c') or \
                in_args[1].dtype.kind in ('f', 'c'):
            _casting_check_out(dt, out.dtype, name, casting)
        if in_args[1].dtype.char == 'L' and out.dtype.kind == 'i' and \
           in_args[0].size != 1 and in_args[1].size != 1:
            raise TypeError("ufunc '{}' output (typecode '{}') could not be "
                            "coerced to provided output parameter (typecode '{}') "
                            "according to the casting rule ''{}''".format(
                                name, in_args[1].dtype.char, out.dtype.char, casting))

    if valid is False or dt.kind in ('b'):
        _raise_no_loop_matching(name)


def _binary_check_case5(*args):
    dt = args[0]
    in_args = args[1]
    name = args[2]
    casting = args[3]
    valid = args[4]
    check_cast = args[5]
    check_out = args[6]
    out = args[7]
    # x = in_args[0]
    # y = in_args[1]
    if check_cast and not _casting_check_binary_except(dt, in_args, casting):
        _raise_no_loop_matching(name)
    if check_out:
        if check_cast and not numpy.can_cast(dt, out.dtype, casting=casting):
            _raise_no_loop_matching(name)
        elif not (dt == numpy.int8 and out.dtype.kind in ('u')):
            _casting_check_out(dt, out.dtype, name, casting)
    if valid is False or numpy.dtype(dt).kind in ('b'):
        _raise_no_loop_matching(name)


def _binary_check_case6(*args):
    dt = args[0]
    in_args = args[1]
    name = args[2]
    casting = args[3]
    valid = args[4]
    check_cast = args[5]
    check_out = args[6]
    out = args[7]
    # x = in_args[0]
    # y = in_args[1]
    if check_cast and not _casting_check_binary_except(dt, in_args, casting):
        _raise_no_loop_matching(name)
    if check_out:
        if check_cast and not numpy.can_cast(dt, out.dtype, casting=casting):
            _raise_no_loop_matching(name)
        elif not (dt == numpy.int8 and out.dtype.kind in ('u')):
            _casting_check_out(dt, out.dtype, name, casting)
    if valid is False or numpy.dtype(dt).kind in ('f', 'c'):
        _raise_no_loop_matching(name)


def _binary_check_case7(*args):
    dt = args[0]
    in_args = args[1]
    name = args[2]
    casting = args[3]
    valid = args[4]
    check_cast = args[5]
    check_out = args[6]
    out = args[7]
    # x = in_args[0]
    # y = in_args[1]
    if check_cast and not _casting_check_binary_except(dt, in_args, casting):
        _raise_no_loop_matching(name)
    if check_out:
        if check_cast and not numpy.can_cast(dt, out.dtype, casting=casting):
            _raise_no_loop_matching(name)
        elif not (dt == numpy.int8 and out.dtype.kind in ('u')):
            _casting_check_out(dt, out.dtype, name, casting)
    if valid is False or numpy.dtype(dt).kind in ('b', 'f', 'c'):
        _raise_no_loop_matching(name)


def _binary_check_case8(*args):
    dt = args[0]
    in_args = args[1]
    name = args[2]
    casting = args[3]
    valid = args[4]
    check_cast = args[5]
    check_out = args[6]
    out = args[7]
    if check_cast:
        _casting_check_binary(numpy.dtype(dt), in_args, name, casting)
    if check_out:
        if not _casting_check_out_without_msg(dt, out.dtype, casting):
            raise TypeError("Cannot cast ufunc \'{}\' output from dtype(\'{}\') "
                            "to dtype(\'{}\') with casting rule \'{}\'".format(
                                name, dt, out.dtype, casting))
    if valid is False:
        _raise_no_loop_matching(name)


def _binary_check_case9(*args):
    dt = args[0]
    in_args = args[1]
    name = args[2]
    casting = args[3]
    valid = args[4]
    check_cast = args[5]
    check_out = args[6]
    out = args[7]
    if check_cast and not _casting_check_without_msg(dt, in_args, casting):
        _raise_no_loop_matching(name)
    if check_out:
        if check_cast and not numpy.can_cast(dt, out.dtype, casting=casting):
            _raise_no_loop_matching(name)
        else:
            _casting_check_out(dt, out.dtype, name, casting)
    if valid is False:
        _raise_no_loop_matching(name)


def _binary_check_case10(*args):
    dt = args[0]
    in_args = args[1]
    name = args[2]
    casting = args[3]
    # valid = args[4]
    check_cast = args[5]
    if check_cast:
        _casting_check_binary(numpy.dtype(dt), in_args, name, casting)


def _unary_check_case1(*args):
    dt = args[0]
    in_args = args[1]
    name = args[2]
    casting = args[3]
    valid = args[4]
    check_cast = args[5]
    check_out = args[6]
    out = args[7]
    ari_dtype = args[8]
    if check_cast and not _casting_check_unary_except(dt, in_args, casting):
        _raise_no_loop_matching(name)
    if check_out:
        if check_cast and not numpy.can_cast(dt, out.dtype, casting=casting):
            _raise_no_loop_matching(name)
        else:
            _casting_check_out(dt, out.dtype, name, casting)
    if valid is False or numpy.dtype(dt).kind in ('b'):
        _raise_no_loop_matching(name)
    # NLCPy original error
    if ari_dtype.kind == 'c':
        _raise_nlcpy_original_error(ari_dtype, name)


def _unary_check_case2(*args):
    dt = args[0]
    in_args = args[1]
    name = args[2]
    casting = args[3]
    valid = args[4]
    check_cast = args[5]
    check_out = args[6]
    out = args[7]
    if check_cast and not _casting_check_unary_except(dt, in_args, casting):
        _raise_no_loop_matching(name)
    if check_out:
        if check_cast and not numpy.can_cast(dt, out.dtype, casting=casting):
            _raise_no_loop_matching(name)
        elif not (dt == numpy.int8 and out.dtype.kind in ('u')):
            _casting_check_out(dt, out.dtype, name, casting)
    if valid is False or numpy.dtype(dt).kind in ('b'):
        _raise_no_loop_matching(name)


def _unary_check_case3(*args):
    dt = args[0]
    in_args = args[1]
    name = args[2]
    casting = args[3]
    valid = args[4]
    check_cast = args[5]
    check_out = args[6]
    out = args[7]
    if check_cast and not _casting_check_unary_except(dt, in_args, casting):
        _raise_no_loop_matching(name)
    if check_out:
        if check_cast and not numpy.can_cast(dt, out.dtype, casting=casting):
            _raise_no_loop_matching(name)
        else:
            _casting_check_out(dt, out.dtype, name, casting)
    if valid is False or numpy.dtype(dt).kind in ('b'):
        _raise_no_loop_matching(name)


def _unary_check_case4(*args):
    dt = args[0]
    in_args = args[1]
    name = args[2]
    casting = args[3]
    valid = args[4]
    check_cast = args[5]
    check_out = args[6]
    out = args[7]
    if check_cast and not _casting_check_unary_except(dt, in_args, casting):
        _raise_no_loop_matching(name)
    if check_out:
        if check_cast and not numpy.can_cast(dt, out.dtype, casting=casting):
            _raise_no_loop_matching(name)
        else:
            _casting_check_out(dt, out.dtype, name, casting)
    if valid is False or numpy.dtype(dt).kind in ('f'):
        _raise_no_loop_matching(name)


def _unary_check_case5(*args):
    dt = args[0]
    in_args = args[1]
    name = args[2]
    casting = args[3]
    valid = args[4]
    check_cast = args[5]
    check_out = args[6]
    out = args[7]
    if check_cast and not _casting_check_unary_except(dt, in_args, casting):
        _raise_no_loop_matching(name)
    if check_out:
        if check_cast and not numpy.can_cast(dt, out.dtype, casting=casting):
            _raise_no_loop_matching(name)
        else:
            _casting_check_out(dt, out.dtype, name, casting)
    if valid is False:
        _raise_no_loop_matching(name)


def _unary_check_case6(*args):
    # dt = args[0]
    # in_args = args[1]
    name = args[2]
    # casting = args[3]
    valid = args[4]
    # check_cast = args[5]
    if valid is False:
        _raise_no_loop_matching(name)


def _add_error_check(*args):
    _binary_check_case1(*args)


def _subtract_error_check(*args):
    _binary_check_case1(*args)
    dt = args[0]
    if numpy.dtype(dt) == numpy.dtype('bool'):
        raise TypeError("nlcpy boolean subtract, the `-` operator, is deprecated, "
                        "use the bitwise_xor, the `^` operator, or the "
                        "logical_xor function instead.")


def _multiply_error_check(*args):
    _binary_check_case1(*args)


def _logaddexp_error_check(*args):
    _binary_check_case3(*args)


def _logaddexp2_error_check(*args):
    _binary_check_case3(*args)


def _true_divide_error_check(*args):
    _binary_check_case2(*args)


def _floor_divide_error_check(*args):
    _binary_check_case4(*args)


def _negative_error_check(*args):
    dt = args[0]
    in_args = args[1]
    name = args[2]
    casting = args[3]
    # valid = args[4]
    check_cast = args[5]
    check_out = args[6]
    out = args[7]
    if check_cast:
        _casting_check_unary(dt, in_args, name, casting)
    if check_out and not _casting_check_out_without_msg(dt, out.dtype, casting):
        raise TypeError("Cannot cast ufunc \'{}\' output from dtype(\'{}\') "
                        "to dtype(\'{}\') with casting rule \'{}\'".format(
                            name, dt, out.dtype, casting))
    if numpy.dtype(dt) == numpy.dtype('bool'):
        raise TypeError("The nlcpy boolean negative, the `-` operator, "
                        "is not supported, use the `~` operator or the "
                        "logical_not function instead.")


def _positive_error_check(*args):
    dt = args[0]
    in_args = args[1]
    name = args[2]
    casting = args[3]
    # valid = args[4]
    check_cast = args[5]
    check_out = args[6]
    out = args[7]
    if check_cast:
        _casting_check_unary(dt, in_args, name, casting)
    if check_out and not _casting_check_out_without_msg(dt, out.dtype, casting):
        raise TypeError("Cannot cast ufunc \'{}\' output from dtype(\'{}\') "
                        "to dtype(\'{}\') with casting rule \'{}\'".format(
                            name, dt, out.dtype, casting))
    if numpy.dtype(dt) == numpy.dtype('bool'):
        raise TypeError("ufunc \'positive\' did not contain a loop with "
                        "signature matching types dtype(\'{}\') -> "
                        "dtype(\'{}\')".format(
                            in_args[0].dtype, numpy.dtype(dt)))


def _power_error_check(*args):
    _binary_check_case5(*args)
    dt = args[0]
    in_args = args[1]
    x = in_args[0]
    y = in_args[1]
    if x.dtype.kind in ('b', 'i', 'u') and dt.kind in ('b', 'i', 'u'):
        if y.dtype.kind == 'i':
            if nlcpy.any(y < 0):
                raise ValueError("Integers to negative integer powers are not allowed.")


def _remainder_error_check(*args):
    _binary_check_case5(*args)


def _fmod_error_check(*args):
    _binary_check_case5(*args)


def _divmod_error_check(*args):
    dt = args[0]
    in_args = args[1]
    name = args[2]
    casting = args[3]
    valid = args[4]
    check_cast = args[5]
    if valid is False or numpy.dtype(dt).kind in ('b'):
        _raise_no_loop_matching(name)
    if check_cast and not _casting_check_without_msg(dt, in_args, casting):
        _raise_no_loop_matching(name)


# TODO: refine error check
def _absolute_error_check(*args):
    dt = args[0]
    in_args = args[1]
    name = args[2]
    casting = args[3]
    valid = args[4]
    check_cast = args[5]
    check_out = args[6]
    out = args[7]

    if in_args[0].dtype.kind == 'c':
        if valid is False or dt.kind not in ('f'):
            _raise_no_loop_matching(name)
    if check_cast:
        if in_args[0].dtype.kind != 'c':
            _casting_check_unary(dt, in_args, name, casting)
    if check_out:
        if check_cast and not \
            numpy.can_cast(dt, out.dtype, casting=casting) and \
                in_args[0].dtype.kind == 'c':
            _raise_no_loop_matching(name)
        if in_args[0].dtype.kind == 'c':
            _casting_check_out(dt, out.dtype, name, casting)
        elif not _casting_check_out_without_msg(dt, out.dtype, casting):
            raise TypeError("Cannot cast ufunc \'{}\' output from dtype(\'{}\') "
                            "to dtype(\'{}\') with casting rule \'{}\'".format(
                                name, dt, out.dtype, casting))
    if valid is False or numpy.dtype(dt).kind in ('c'):
        raise TypeError("ufunc \'absolute\' did not contain a loop with "
                        "signature matching types dtype(\'{}\') -> "
                        "dtype(\'{}\')".format(
                            str(numpy.dtype(dt)),
                            str(numpy.dtype(dt))
                        ))


def _fabs_error_check(*args):
    _binary_check_case3(*args)
    """
    dt = args[0]
    in_args = args[1]
    name = args[2]
    casting = args[3]
    valid = args[4]
    check_cast = args[5]
    check_out = args[6]
    out = args[7]
    if check_cast and not _casting_check_without_msg(dt, in_args, casting):
        _raise_no_loop_matching(name)
    if check_out:
        if check_cast and not numpy.can_cast(dt, out.dtype, casting=casting):
            _raise_no_loop_matching(name)
        else:
            _casting_check_out(dt, out.dtype, name, casting)
    if valid is False or numpy.dtype(dt).kind in ('b'):
        _raise_no_loop_matching(name)
    """


def _rint_error_check(*args):
    _binary_check_case1(*args)
    """
    dt = args[0]
    in_args = args[1]
    name = args[2]
    casting = args[3]
    valid = args[4]
    check_cast = args[5]
    check_out = args[6]
    out = args[7]
    ari_dtype = args[8]
    if check_cast and not _casting_check_without_msg(dt, in_args, casting):
        _raise_no_loop_matching(name)
    if check_out:
        if check_cast and not numpy.can_cast(dt, out.dtype, casting=casting):
            _raise_no_loop_matching(name)
        else:
            _casting_check_out(dt, out.dtype, name, casting)
    if valid is False or numpy.dtype(dt).kind in ('b'):
        _raise_no_loop_matching(name)
    # NLCPy original error
    if ari_dtype.kind == 'c':
        raise TypeError("\'{}\' is not supported as a calculation dtype "
                        "of ufunc \'{}\'".format(ari_dtype, name))
    """


def _sign_error_check(*args):
    dt = args[0]
    in_args = args[1]
    name = args[2]
    casting = args[3]
    valid = args[4]
    check_cast = args[5]
    check_out = args[6]
    out = args[7]
    if check_cast:
        _casting_check_unary(numpy.dtype(dt), in_args, name, casting)
    if check_out and not _casting_check_out_without_msg(dt, out.dtype, casting):
        raise TypeError("Cannot cast ufunc \'{}\' output from dtype(\'{}\') "
                        "to dtype(\'{}\') with casting rule \'{}\'".format(
                            name, dt, out.dtype, casting))
    if valid is False or numpy.dtype(dt).kind in ('b'):
        raise TypeError("ufunc \'sign\' did not contain a loop with signature "
                        "matching types dtype(\'{}\') -> dtype(\'{}\')".format(
                            in_args[0].dtype, numpy.dtype(dt)))


def _heaviside_error_check(*args):
    _binary_check_case5(*args)


def _conj_error_check(*args):
    dt = args[0]
    in_args = args[1]
    name = args[2]
    casting = args[3]
    valid = args[4]
    check_cast = args[5]
    if valid is False or numpy.dtype(dt).kind in ('b'):
        _raise_no_loop_matching(name)
    if check_cast and not _casting_check_unary_except(dt, in_args, casting):
        _raise_no_loop_matching(name)


def _conjugate_error_check(*args):
    _unary_check_case2(*args)


def _exp_error_check(*args):
    _unary_check_case3(*args)


def _exp2_error_check(*args):
    _unary_check_case1(*args)


def _log_error_check(*args):
    _unary_check_case2(*args)


def _log2_error_check(*args):
    _unary_check_case1(*args)


def _log10_error_check(*args):
    _unary_check_case1(*args)


def _expm1_error_check(*args):
    _unary_check_case1(*args)


def _log1p_error_check(*args):
    _unary_check_case1(*args)


def _sqrt_error_check(*args):
    _unary_check_case2(*args)


def _square_error_check(*args):
    _unary_check_case2(*args)


def _cbrt_error_check(*args):
    _unary_check_case3(*args)


def _reciprocal_error_check(*args):
    _unary_check_case2(*args)


def _sin_error_check(*args):
    _unary_check_case3(*args)


def _cos_error_check(*args):
    _unary_check_case3(*args)


def _tan_error_check(*args):
    _unary_check_case3(*args)


def _arcsin_error_check(*args):
    _unary_check_case3(*args)


def _arccos_error_check(*args):
    _unary_check_case3(*args)


def _arctan_error_check(*args):
    _unary_check_case3(*args)


def _arctan2_error_check(*args):
    _binary_check_case5(*args)


def _hypot_error_check(*args):
    _binary_check_case5(*args)


def _sinh_error_check(*args):
    _unary_check_case3(*args)


def _cosh_error_check(*args):
    _unary_check_case3(*args)


def _tanh_error_check(*args):
    _unary_check_case3(*args)


def _arcsinh_error_check(*args):
    _unary_check_case3(*args)


def _arccosh_error_check(*args):
    _unary_check_case3(*args)


def _arctanh_error_check(*args):
    _unary_check_case3(*args)


def _deg2rad_error_check(*args):
    _unary_check_case3(*args)


def _rad2deg_error_check(*args):
    _unary_check_case3(*args)


def _degrees_error_check(*args):
    _unary_check_case3(*args)


def _radians_error_check(*args):
    _unary_check_case3(*args)


def _bitwise_and_error_check(*args):
    _binary_check_case6(*args)


def _bitwise_or_error_check(*args):
    _binary_check_case6(*args)


def _bitwise_xor_error_check(*args):
    _binary_check_case6(*args)


def _invert_error_check(*args):
    _unary_check_case4(*args)


def _left_shift_error_check(*args):
    _binary_check_case7(*args)


def _right_shift_error_check(*args):
    _binary_check_case7(*args)


def _greater_error_check(*args):
    _binary_check_case10(*args)


def _greater_equal_error_check(*args):
    _binary_check_case10(*args)


def _less_error_check(*args):
    _binary_check_case10(*args)


def _less_equal_error_check(*args):
    _binary_check_case10(*args)


def _not_equal_error_check(*args):
    _binary_check_case10(*args)


def _equal_error_check(*args):
    _binary_check_case10(*args)


def _logical_and_error_check(*args):
    _binary_check_case10(*args)


def _logical_or_error_check(*args):
    _binary_check_case10(*args)


def _logical_xor_error_check(*args):
    _binary_check_case10(*args)


def _logical_not_error_check(*args):
    _unary_check_case6(*args)


def _minimum_error_check(*args):
    _binary_check_case8(*args)


def _maximum_error_check(*args):
    _binary_check_case8(*args)


def _fmax_error_check(*args):
    _binary_check_case8(*args)


def _fmin_error_check(*args):
    _binary_check_case8(*args)


def _isfinite_error_check(*args):
    _unary_check_case6(*args)


def _isinf_error_check(*args):
    _unary_check_case6(*args)


def _isnan_error_check(*args):
    _unary_check_case6(*args)


def _isnat_error_check(*args):
    _unary_check_case6(*args)


def _signbit_error_check(*args):
    dt = args[0]
    in_args = args[1]
    name = args[2]
    # casting = args[3]
    valid = args[4]
    check_cast = args[5]
    if valid is False or numpy.dtype(dt).kind not in ('b'):
        _raise_no_loop_matching(name)
    if check_cast and in_args[0].dtype.kind == 'c':
        _raise_no_loop_matching(name)


def _copysign_error_check(*args):
    _binary_check_case9(*args)


def _nextafter_error_check(*args):
    _binary_check_case9(*args)


def _spacing_error_check(*args):
    _unary_check_case5(*args)


def _modf_error_check(*args):
    _unary_check_case6(*args)


def _ldexp_error_check(*args):
    dt = args[0]
    in_args = args[1]
    name = args[2]
    casting = args[3]
    valid = args[4]
    check_cast = args[5]
    check_out = args[6]
    out = args[7]
    if in_args[1].dtype.kind not in ('b', 'i', 'u'):
        _raise_no_loop_matching(name)
    if check_cast and not _casting_check_without_msg(dt, in_args, casting):
        _raise_no_loop_matching(name)
    if check_out:
        if check_cast and not numpy.can_cast(dt, out.dtype, casting=casting):
            _raise_no_loop_matching(name)
        else:
            _casting_check_out(dt, out.dtype, name, casting)
    if valid is False or numpy.dtype(dt).kind not in ('f'):
        _raise_no_loop_matching(name)


def _floor_error_check(*args):
    _unary_check_case5(*args)


def _ceil_error_check(*args):
    _unary_check_case5(*args)


def _trunc_error_check(*args):
    dt = args[0]
    in_args = args[1]
    name = args[2]
    casting = args[3]
    valid = args[4]
    check_cast = args[5]
    check_out = args[6]
    out = args[7]
    if check_cast and not _casting_check_without_msg(dt, in_args, casting):
        _raise_no_loop_matching(name)
    if check_out:
        _casting_check_out(dt, out.dtype, name, casting)
    if valid is False:
        _raise_no_loop_matching(name)
