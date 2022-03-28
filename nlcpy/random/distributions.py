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


def binomial(n, p, size=None):
    """This function has the same `nlcpy.random.RandomState.binomial`

    See Also
    --------
    nlcpy.random.RandomState.binomial : Draws samples from a binomial distribution.

    """
    rs = generator._get_rand()
    return rs.binomial(n, p, size=size)


def exponential(scale, size=None):
    """This function has the same `nlcpy.random.RandomState.exponential`

    See Also
    --------
    nlcpy.random.RandomState.exponential : Draws samples from an exponential
        distribution.

    """
    rs = generator._get_rand()
    return rs.exponential(scale, size=size)


def gamma(shape, scale=1.0, size=None):
    """This function has the same `nlcpy.random.RandomState.gamma`

    See Also
    --------
    nlcpy.random.RandomState.gamma : Draws samples from a Gamma distribution.

    """
    rs = generator._get_rand()
    return rs.gamma(shape, scale, size=size)


def geometric(p, size=None):
    """This function has the same `nlcpy.random.RandomState.geometric`

    See Also
    --------
    nlcpy.random.RandomState.geometric : Draws samples from a geometric distribution.

    """
    rs = generator._get_rand()
    return rs.geometric(p, size=size)


def gumbel(loc=0.0, scale=1.0, size=None):
    """This function has the same `nlcpy.random.RandomState.gumbel`

    See Also
    --------
    nlcpy.random.RandomState.gumbel : Draws samples from a Gumbel distribution.

    """
    rs = generator._get_rand()
    return rs.gumbel(loc, scale, size=size)


def logistic(loc=0.0, scale=1.0, size=None):
    """This function has the same `nlcpy.random.RandomState.logistic`

    See Also
    --------
    nlcpy.random.RandomState.logistic : Draws samples from a logistic distribution.

    """
    rs = generator._get_rand()
    return rs.logistic(loc, scale, size=size)


def lognormal(mean=0.0, sigma=1.0, size=None):
    """This function has the same `nlcpy.random.RandomState.lognormal`

    See Also
    --------
    nlcpy.random.RandomState.lognormal : Draws samples from a log-normal distribution.

    """
    rs = generator._get_rand()
    return rs.lognormal(mean, sigma, size=size)


def normal(loc=0.0, scale=1.0, size=None):
    """This function has the same `nlcpy.random.RandomState.normal`

    See Also
    --------
    generator.RandomState.normal : Draws random samples from a normal (Gaussian)
        distribution.

    """
    rs = generator._get_rand()
    return rs.normal(loc=loc, scale=scale, size=size)


def poisson(lam=1.0, size=None):
    """This function has the same `nlcpy.random.RandomState.poisson`

    See Also
    --------
    nlcpy.random.RandomState.poisson : Draws samples from a Poisson distribution.

    """
    rs = generator._get_rand()
    return rs.poisson(lam, size=size)


def standard_cauchy(size=None):
    """This function has the same `nlcpy.random.RandomState.standard_cauchy`

    See Also
    --------
    nlcpy.random.RandomState.standard_cauchy : Draws samples from a standard Cauchy
        distribution with mode = 0.

    """
    rs = generator._get_rand()
    return rs.standard_cauchy(size=size)


def standard_exponential(size=None):
    """This function has the same `nlcpy.random.RandomState.standard_exponential`

    See Also
    --------
    nlcpy.random.RandomState.standard_exponential : Draws samples from a standard
        exponential distribution.

    """
    rs = generator._get_rand()
    return rs.standard_exponential(size=size)


def standard_gamma(shape, size=None):
    """This function has the same `nlcpy.random.RandomState.standard_gamma`

    See Also
    --------
    nlcpy.random.RandomState.standard_gamma : Draws samples from a standard Gamma
        distribution.

    """
    rs = generator._get_rand()
    return rs.standard_gamma(shape, size=size)


def standard_normal(size=None):
    """This function has the same `nlcpy.random.RandomState.standard_normal`

    See Also
    --------
    nlcpy.random.RandomState.standard_normal : Draws samples from a standard Normal
        distribution (mean=0, stdev=1).

    """
    rs = generator._get_rand()
    return rs.standard_normal(size=size)


def uniform(low=0.0, high=1.0, size=None):
    """This function has the same `nlcpy.random.RandomState.uniform`

    See Also
    --------
    nlcpy.random.RandomState.uniform : Draws samples from a uniform distribution.

    """
    rs = generator._get_rand()
    return rs.uniform(low, high, size=size)


def weibull(a, size=None):
    """This function has the same `nlcpy.random.RandomState.weibull`

    See Also
    --------
    nlcpy.random.RandomState.weibull : Draws samples from a Weibull distribution.

    """
    rs = generator._get_rand()
    return rs.weibull(a, size=size)
