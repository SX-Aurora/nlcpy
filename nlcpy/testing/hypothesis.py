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
import numpy


def chi_square_test(observed, expected, alpha=0.05, df=None):
    """Testing Goodness-of-fit Test with Pearson's Chi-squared Test.

    Args:
        observed (list of ints): List of # of counts each element is observed.
        expected (list of floats): List of # of counts each element is expected
            to be observed.
        alpha (float): Significance level. Currently,
            only 0.05 and 0.01 are acceptable.
        df (int): Degree of freedom. If ``None``,
            it is set to the length of ``observed`` minus 1.

    Returns:
        bool: ``True`` if null hypothesis is **NOT** reject.
        Otherwise, ``False``.
    """
    if df is None:
        df = observed.size - 1

    if alpha == 0.01:
        alpha_idx = 0
    elif alpha == 0.05:
        alpha_idx = 1
    else:
        raise ValueError('support only alpha == 0.05 or 0.01')

    chi_square = numpy.sum((observed - expected) ** 2 / expected)
    return chi_square < chi_square_table[alpha_idx][df]


# https://www.medcalc.org/manual/chi-square-table.php
chi_square_table = [
    [None,
     6.635, 9.210, 11.345, 13.277, 15.086,
     16.812, 18.475, 20.090, 21.666, 23.209,
     24.725, 26.217, 27.688, 29.141, 30.578,
     32.000, 33.409, 34.805, 36.191, 37.566,
     38.932, 40.289, 41.638, 42.980, 44.314,
     45.642, 46.963, 48.278, 49.588, 50.892,
     52.191, 53.486, 54.776, 56.061, 57.342,
     58.619, 59.893, 61.162, 62.428, 63.691,
     64.950, 66.206, 67.459, 68.710, 69.957,
     71.201, 72.443, 73.683, 74.919, 76.154,
     77.386, 78.616, 79.843, 81.069, 82.292,
     83.513, 84.733, 85.950, 87.166, 88.379,
     89.591, 90.802, 92.010, 93.217, 94.422,
     95.626, 96.828, 98.028, 99.228, 100.425,
     101.621, 102.816, 104.010, 105.202, 106.393,
     107.583, 108.771, 109.958, 111.144, 112.329,
     113.512, 114.695, 115.876, 117.057, 118.236,
     119.414, 120.591, 121.767, 122.942, 124.116,
     125.289, 126.462, 127.633, 128.803, 129.973,
     131.141, 132.309, 133.476, 134.642, 135.807,
     136.971, 138.134, 139.297, 140.459, 141.620,
     142.780, 143.940, 145.099, 146.257, 147.414,
     148.571, 149.727, 150.882, 152.037, 153.191,
     154.344, 155.496, 156.648, 157.800, 158.950,
     160.100, 161.250, 162.398, 163.546, 164.694,
     165.841, 166.987, 168.133, 169.278, 170.423,
     171.567, 172.711, 173.854, 174.996, 176.138,
     177.280, 178.421, 179.561, 180.701, 181.840,
     182.979, 184.118, 185.256, 186.393, 187.530,
     188.666, 189.802, 190.938, 192.073, 193.208,
     194.342, 195.476, 196.609, 197.742, 198.874,
     200.006, 201.138, 202.269, 203.400, 204.530,
     205.660, 206.790, 207.919, 209.047, 210.176,
     211.304, 212.431, 213.558, 214.685, 215.812,
     216.938, 218.063, 219.189, 220.314, 221.438,
     222.563, 223.687, 224.810, 225.933, 227.056,
     228.179, 229.301, 230.423, 231.544, 232.665,
     233.786, 234.907, 236.027, 237.147, 238.266,
     239.386, 240.505, 241.623, 242.742, 243.860,
     244.977, 246.095, 247.212, 248.329, 249.445,
     250.561, 251.677, 252.793, 253.908, 255.023,
     256.138, 257.253, 258.367, 259.481, 260.595,
     261.708, 262.821, 263.934, 265.047, 266.159,
     267.271, 268.383, 269.495, 270.606, 271.717,
     272.828, 273.939, 275.049, 276.159, 277.269,
     278.379, 279.488, 280.597, 281.706, 282.814,
     283.923, 285.031, 286.139, 287.247, 288.354,
     289.461, 290.568, 291.675, 292.782, 293.888,
     294.994, 296.100, 297.206, 298.311, 299.417,
     300.522, 301.626, 302.731, 303.835, 304.940],
    [None,
     3.841, 5.991, 7.815, 9.488, 11.070,
     12.592, 14.067, 15.507, 16.919, 18.307,
     19.675, 21.026, 22.362, 23.685, 24.996,
     26.296, 27.587, 28.869, 30.144, 31.410,
     32.671, 33.924, 35.172, 36.415, 37.652,
     38.885, 40.113, 41.337, 42.557, 43.773,
     44.985, 46.194, 47.400, 48.602, 49.802,
     50.998, 52.192, 53.384, 54.572, 55.758,
     56.942, 58.124, 59.304, 60.481, 61.656,
     62.830, 64.001, 65.171, 66.339, 67.505,
     68.669, 69.832, 70.993, 72.153, 73.311,
     74.468, 75.624, 76.778, 77.931, 79.082,
     80.232, 81.381, 82.529, 83.675, 84.821,
     85.965, 87.108, 88.250, 89.391, 90.531,
     91.670, 92.808, 93.945, 95.081, 96.217,
     97.351, 98.484, 99.617, 100.749, 101.879,
     103.010, 104.139, 105.267, 106.395, 107.522,
     108.648, 109.773, 110.898, 112.022, 113.145,
     114.268, 115.390, 116.511, 117.632, 118.752,
     119.871, 120.990, 122.108, 123.225, 124.342,
     125.458, 126.574, 127.689, 128.804, 129.918,
     131.031, 132.144, 133.257, 134.369, 135.480,
     136.591, 137.701, 138.811, 139.921, 141.030,
     142.138, 143.246, 144.354, 145.461, 146.567,
     147.674, 148.779, 149.885, 150.989, 152.094,
     153.198, 154.302, 155.405, 156.508, 157.610,
     158.712, 159.814, 160.915, 162.016, 163.116,
     164.216, 165.316, 166.415, 167.514, 168.613,
     169.711, 170.809, 171.907, 173.004, 174.101,
     175.198, 176.294, 177.390, 178.485, 179.581,
     180.676, 181.770, 182.865, 183.959, 185.052,
     186.146, 187.239, 188.332, 189.424, 190.516,
     191.608, 192.700, 193.791, 194.883, 195.973,
     197.064, 198.154, 199.244, 200.334, 201.423,
     202.513, 203.602, 204.690, 205.779, 206.867,
     207.955, 209.042, 210.130, 211.217, 212.304,
     213.391, 214.477, 215.563, 216.649, 217.735,
     218.820, 219.906, 220.991, 222.076, 223.160,
     224.245, 225.329, 226.413, 227.496, 228.580,
     229.663, 230.746, 231.829, 232.912, 233.994,
     235.077, 236.159, 237.240, 238.322, 239.403,
     240.485, 241.566, 242.647, 243.727, 244.808,
     245.888, 246.968, 248.048, 249.128, 250.207,
     251.286, 252.365, 253.444, 254.523, 255.602,
     256.680, 257.758, 258.837, 259.914, 260.992,
     262.070, 263.147, 264.224, 265.301, 266.378,
     267.455, 268.531, 269.608, 270.684, 271.760,
     272.836, 273.911, 274.987, 276.062, 277.138,
     278.213, 279.288, 280.362, 281.437, 282.511,
     283.586, 284.660, 285.734, 286.808, 287.882]]
