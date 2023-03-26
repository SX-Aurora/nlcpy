#
# * The source code in this file is based on the soure code of NumPy.
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
import sys
import re
import os
import io
import platform
import nlcpy
from nlcpy.veo import _veo
import numpy


def get_nlc_lib_path(arch):
    nlc_path = os.environ.get('VE_LD_LIBRARY_PATH', None)
    if nlc_path is not None:
        pat = re.compile(r'nlc/[0-9]+\.[0-9]+\.[0-9]+')
        for _nlc_path in nlc_path.split(':'):
            if pat.search(_nlc_path):
                return _nlc_path

    if arch == 1:
        default_nlc_path = '/opt/nec/ve/lib'
    elif arch == 3:
        default_nlc_path = '/opt/nec/ve3/lib'

    if not os.path.exists(default_nlc_path):
        return None
    require_nlc_libs = [
        'libasl_openmp_i64.so',
        'libaslfftw3_i64.so',
        'liblapack_i64.so',
        'libblas_openmp_i64.so',
        'libsca_openmp_i64.so',
        'libheterosolver_openmp_i64.so',
        'libsblas_openmp_i64.so',
        'libcblas_i64.so',
    ]
    for _file in os.listdir(default_nlc_path):
        if _file in require_nlc_libs:
            require_nlc_libs.pop(require_nlc_libs.index(_file))
    if len(require_nlc_libs) == 0:
        return default_nlc_path
    else:
        return None


def get_nlcpy_ver():
    return nlcpy.__version__


def get_numpy_ver():
    return numpy.__version__


def get_ncc_build_ver(arch):
    try:
        if arch == 1:
            from nlcpy_ve1_kernel import build_info
        elif arch == 3:
            from nlcpy_ve3_kernel import build_info
        else:
            raise RuntimeError('Unknown arch:', arch)
    except ImportError:
        return None
    try:
        ncc_ver = build_info.ncc_build_version
    except AttributeError:
        return None
    return ncc_ver


class pkg_info(object):

    def __init__(self):
        self.nve = nlcpy.venode.get_num_available_venodes()
        self.ve_pids = [nlcpy.venode.VE(i).pid for i in range(self.nve)]
        self.ve_archs = [nlcpy.venode.VE(i).arch for i in range(self.nve)]
        self.ve_ncores = [nlcpy.venode.VE(i).ncore for i in range(self.nve)]
        self.ve_tot_mems = [nlcpy.venode.VE(i).meminfo['kb_main_total']
                            for i in range(self.nve)]
        self.ve_used_mems = [nlcpy.venode.VE(i).meminfo['kb_main_used']
                             for i in range(self.nve)]

        _ve = nlcpy.venode.VE()
        self.ve_pid = _ve.pid
        self.ve_arch = _ve.arch
        self.records = {
            'OS': platform.platform(),
            'Python Version': platform.python_version(),
            'NLC Library Path': get_nlc_lib_path(self.ve_arch),
            'NLCPy Kernel Path': _ve.libpath._lib_dir,
            'NLCPy Version': get_nlcpy_ver(),
            'NumPy Version': get_numpy_ver(),
            'ncc Build Version': get_ncc_build_ver(self.ve_arch),
            'VEO API Version': _veo._veo_api_version,
            'VEO Version': _veo._veo_version,
            'Assigned VE IDs': self.ve_pids,
            'VE Arch': self.ve_archs,
            'VE ncore': self.ve_ncores,
            'VE Total Mem[KB]': self.ve_tot_mems,
            'VE Used  Mem[KB]': self.ve_used_mems,
        }

    def __str__(self):
        max_width = max([len(k) for k in self.records.keys()])
        fmt = '{:' + str(max_width) + '}: {}\n'
        with io.StringIO() as s:
            for k, v in self.records.items():
                s.write(fmt.format(k, v))
            ret = s.getvalue()
        return ret


def show_config():
    """Shows various information in the system on which NLCPy is running.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.show_config()  # doctest: +SKIP
    OS                : Linux-4.18.0-372.32.1.el8_6.x86_64-x86_64-with-glibc2.10
    Python Version    : 3.8.12
    NLC Library Path  : /opt/nec/ve3/nlc/3.0.0/lib
    NLCPy Kernel Path : /opt/nec/ve/nlcpy/3.0.0/lib/python3.6/nlcpy_ve1_kernel
    NLCPy Version     : 3.0.0
    NumPy Version     : 1.19.5
    ncc Build Version : 4.0.83
    VEO API Version   : 15
    VEO Version       : 2.13.0
    Assigned VE IDs  : [0, 1, 2, 3]
    VE Arch          : [1, 1, 1, 1]
    VE ncore         : [8, 8, 8, 8]
    VE Total Mem[KB] : [50331648, 50331648, 50331648, 50331648]
    VE Used  Mem[KB] : [4085760, 131072, 131072, 131072]
    """
    sys.stdout.write(str(pkg_info()))
    sys.stdout.flush()
