#
# * The source code in this file is based on the soure code of NumPy and CuPy.
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
# # NumPy License #
#
#     Copyright (c) 2005-2020, NumPy Developers.
#     All rights reserved.
#
#     Redistribution and use in source and binary forms, with or without
#     modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#     * Neither the name of the NumPy Developers nor the names of any contributors may be
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
# # CuPy License #
#
#     Copyright (c) 2015 Preferred Infrastructure, Inc.
#     Copyright (c) 2015 Preferred Networks, Inc.
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy
#     of this software and associated documentation files (the "Software"), to deal
#     in the Software without restriction, including without limitation the rights
#     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#     copies of the Software, and to permit persons to whom the Software is
#     furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included in
#     all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#     THE SOFTWARE.
#

from nlcpy.random import generator


def rand(*size):
    """This function has the same `nlcpy.random.RandomState.rand`

    See Also
    --------
    nlcpy.random.RandomState.rand : Random values in a given shape.

    """
    rs = generator._get_rand()
    return rs.rand(None if len(size) == 0 else size)


def randn(*size):
    """This function has the same `nlcpy.random.RandomState.randn`

    See Also
    --------
    nlcpy.random.RandomState.randn : Returns a sample (or samples) from the "standard
        normal" distribution.

    """
    rs = generator._get_rand()
    return rs.randn(None if len(size) == 0 else size)


def randint(low, high=None, size=None, dtype=int):
    """This function has the same `nlcpy.random.RandomState.randint`

    See Also
    --------
    nlcpy.random.RandomState.randint : Returns random integers from low (inclusive) to
        high (exclusive).

    """
    rs = generator._get_rand()
    return rs.randint(low, high, size, dtype)


def ranf(*size):
    """This function has the same `nlcpy.random.RandomState.ranf`

    See Also
    --------
    nlcpy.random.RandomState.ranf : This is an alias of random_sample.

    """
    rs = generator._get_rand()
    return rs.ranf(None if len(size) == 0 else size)


def sample(*size):
    """This function has the same `nlcpy.random.RandomState.sample`

    See Also
    --------
    nlcpy.random.RandomState.sample : This is an alias of random_sample.

    """
    rs = generator._get_rand()
    return rs.sample(None if len(size) == 0 else size)


def random(size=None):
    """This function has the same `nlcpy.random.RandomState.random`

    See Also
    --------
    nlcpy.random.RandomState.random : Returns random floats in the half-open interval
        [0.0, 1.0).

    """
    rs = generator._get_rand()
    return rs.random(size=size)


def random_integers(low, high=None, size=None):
    """This function has the same `nlcpy.random.randint` interval [0.0, 1.0]

    See Also
    --------
    nlcpy.random.randint : Returns random integers from low (inclusive) to high
        (exclusive).

    """
    if high is None:
        high = low
        low = 1
    return randint(low, high + 1, size)


def random_sample(size=None):
    """This function has the same `nlcpy.random.RandomState.random_sample`

    See Also
    --------
    nlcpy.random.RandomState.random_sample : Returns random floats in the half-open
        interval [0.0, 1.0).

    """
    rs = generator._get_rand()
    return rs.random_sample(size)


def bytes(length):
    """This function has the same `nlcpy.random.RandomState.bytes`

    See Also
    --------
    nlcpy.random.RandomState.bytes : Returns random bytes.

    """
    rs = generator._get_rand()
    return rs.bytes(length)
