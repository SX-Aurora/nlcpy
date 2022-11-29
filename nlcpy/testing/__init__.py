#
# * The source code in this file is based on the soure code of CuPy.
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

from nlcpy.testing import array  # NOQA
from nlcpy.testing import helper  # NOQA
from nlcpy.testing import parameterized  # NOQA
from nlcpy.testing import random  # NOQA

from nlcpy.testing.array import assert_allclose  # NOQA
from nlcpy.testing.array import assert_array_almost_equal  # NOQA
from nlcpy.testing.array import assert_array_almost_equal_nulp  # NOQA
from nlcpy.testing.array import assert_array_equal  # NOQA
from nlcpy.testing.array import assert_array_less  # NOQA
from nlcpy.testing.array import assert_array_list_equal  # NOQA
from nlcpy.testing.array import assert_array_max_ulp  # NOQA
from nlcpy.testing.helper import assert_warns  # NOQA
from nlcpy.testing.helper import for_all_dtypes  # NOQA
from nlcpy.testing.helper import for_all_dtypes_combination  # NOQA
from nlcpy.testing.helper import for_CF_orders  # NOQA
from nlcpy.testing.helper import for_complex_dtypes  # NOQA
from nlcpy.testing.helper import for_dtypes  # NOQA
from nlcpy.testing.helper import for_dtypes_combination  # NOQA
from nlcpy.testing.helper import for_float_dtypes  # NOQA
from nlcpy.testing.helper import for_int_dtypes  # NOQA
from nlcpy.testing.helper import for_int_dtypes_combination  # NOQA
from nlcpy.testing.helper import for_all_axis  # NOQA
from nlcpy.testing.helper import for_broadcast  # NOQA
from nlcpy.testing.helper import set_random_seed  # NOQA
from nlcpy.testing.helper import for_orders  # NOQA
from nlcpy.testing.helper import for_signed_dtypes  # NOQA
from nlcpy.testing.helper import for_signed_dtypes_combination  # NOQA
from nlcpy.testing.helper import for_unsigned_dtypes  # NOQA
from nlcpy.testing.helper import for_unsigned_dtypes_combination  # NOQA
from nlcpy.testing.helper import numpy_nlcpy_allclose  # NOQA
from nlcpy.testing.helper import numpy_nlcpy_array_almost_equal  # NOQA
from nlcpy.testing.helper import numpy_nlcpy_array_almost_equal_nulp  # NOQA
from nlcpy.testing.helper import numpy_nlcpy_array_equal  # NOQA
from nlcpy.testing.helper import numpy_nlcpy_array_less  # NOQA
from nlcpy.testing.helper import numpy_nlcpy_array_list_equal  # NOQA
from nlcpy.testing.helper import numpy_nlcpy_array_max_ulp  # NOQA
from nlcpy.testing.helper import numpy_nlcpy_check_for_unary_ufunc  # NOQA
from nlcpy.testing.helper import numpy_nlcpy_check_for_binary_ufunc  # NOQA
from nlcpy.testing.helper import numpy_nlcpy_equal  # NOQA
from nlcpy.testing.helper import numpy_nlcpy_raises  # NOQA
from nlcpy.testing.helper import numpy_satisfies  # NOQA
from nlcpy.testing.helper import NumpyAliasBasicTestBase  # NOQA
from nlcpy.testing.helper import NumpyAliasValuesTestBase  # NOQA
from nlcpy.testing.helper import numpy_nlcpy_errstate  # NOQA
from nlcpy.testing.helper import shaped_arange  # NOQA
from nlcpy.testing.helper import shaped_rearrange_for_broadcast  # NOQA
from nlcpy.testing.helper import create_shape_and_axis_set  # NOQA
from nlcpy.testing.helper import shaped_random  # NOQA
from nlcpy.testing.helper import shaped_reverse_arange  # NOQA
from nlcpy.testing.helper import with_requires  # NOQA
from nlcpy.testing.helper import asnumpy  # NOQA
from nlcpy.testing.device import multi_ve  # NOQA
from nlcpy.testing.parameterized import parameterize  # NOQA
from nlcpy.testing.parameterized import product  # NOQA
from nlcpy.testing.random import fix_random  # NOQA
from nlcpy.testing.random import generate_seed  # NOQA
