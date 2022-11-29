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
import re
from nlcpy import _path
import subprocess
import os
import numpy as np

pkg_ver = {'NLC': '', 'ASL': '', 'BLAS': '',
           'LAPACK': '', 'SCA': '', 'NUMPY': '', 'NCC': ''}


def chk_data(data):
    ans = ""

    if data == "":
        ans = "Not Available"
    else:
        pat = r"[\d+]+[\w.]*(?:[-\s]+(?:alpha|beta|build)[\w.]*)?"
        mat = re.match(pat, data)
        if mat is None:
            ans = "Not Available"
        else:
            ans = data

    return ans


def exec_cmd(get_com):
    try:
        proc = subprocess.check_output(get_com, shell=True)
        std_proc = proc
    except subprocess.CalledProcessError:
        std_proc = ""

    if std_proc != "":
        out = std_proc.decode().rstrip()
    else:
        out = ""

    return out


def get_nlc_ver():
    ld_path = os.getenv('VE_LD_LIBRARY_PATH')
    if ld_path is None:
        gp = None
    else:
        pat = r'nlc/([0-9]+\.){1}[0-9]+(\.[0-9]+)?'
        p = re.search(pat, ld_path)
        gp = p.group(0)
        gp = gp[4:]

    if gp is None:
        pkg_ver['NLC'] = ""
    else:
        pkg_ver['NLC'] = gp
    return gp


def get_asl_ver(base_ver):
    cmd = '/usr/bin/rpm -qi nec-asl-ve-' + base_ver
    get_cmd = cmd + "| /usr/bin/grep Version | " + \
        "/usr/bin/awk -F':' '{print $2}'"
    get_cmd = get_cmd + " | /usr/bin/sed 's/^[[:blank:]]*//'"
    m_v = exec_cmd(get_cmd)

    cmd = '/usr/bin/rpm -qi nec-asl-ve-' + base_ver
    get_cmd = cmd + "| /usr/bin//grep Release | " + \
        "/usr/bin/awk -F':' '{print $2}'"
    get_cmd = get_cmd + " | /usr/bin/sed 's/^[[:blank:]]*//'"
    n_v = exec_cmd(get_cmd)

    ans = m_v + "-" + n_v

    pkg_ver['ASL'] = ans


def get_blas_ver(base_ver):
    cmd = '/usr/bin/rpm -qi nec-blas-ve-' + base_ver
    get_cmd = cmd + "| /usr/bin/grep Version | " + \
        "/usr/bin/awk -F':' '{print $2}'"
    get_cmd = get_cmd + " | /usr/bin/sed 's/^[[:blank:]]*//'"
    m_v = exec_cmd(get_cmd)

    cmd = '/usr/bin/rpm -qi nec-blas-ve-' + base_ver
    get_cmd = cmd + "| /usr/bin/grep Release | " + \
        "/usr/bin/awk -F':' '{print $2}'"
    get_cmd = get_cmd + " | /usr/bin/sed 's/^[[:blank:]]*//'"
    n_v = exec_cmd(get_cmd)

    ans = m_v + "-" + n_v

    pkg_ver['BLAS'] = ans


def get_lapack_ver(base_ver):
    cmd = '/usr/bin/rpm -qi nec-lapack-ve-' + base_ver
    get_cmd = cmd + "| /usr/bin/grep Version | " + \
        "/usr/bin/awk -F':' '{print $2}'"
    get_cmd = get_cmd + " | /usr/bin/sed 's/^[[:blank:]]*//'"
    m_v = exec_cmd(get_cmd)

    cmd = 'rpm -qi nec-lapack-ve-' + base_ver
    get_cmd = cmd + "| /usr/bin/grep Release | " + \
        "/usr/bin/awk -F':' '{print $2}'"
    get_cmd = get_cmd + " | /usr/bin/sed 's/^[[:blank:]]*//'"
    n_v = exec_cmd(get_cmd)

    ans = m_v + "-" + n_v

    pkg_ver['LAPACK'] = ans


def get_sca_ver(base_ver):
    cmd = '/usr/bin/rpm -qi nec-sca-ve-' + base_ver
    get_cmd = cmd + "| /usr/bin/grep Version |" + \
        "/usr/bin/awk -F':' '{print $2}'"
    get_cmd = get_cmd + " | /usr/bin/sed 's/^[[:blank:]]*//'"
    m_v = exec_cmd(get_cmd)

    cmd = '/usr/bin/rpm -qi nec-sca-ve-' + base_ver
    get_cmd = cmd + "| /usr/bin/grep Release | " + \
        "/usr/bin/awk -F':' '{print $2}'"
    get_cmd = get_cmd + " | /usr/bin/sed 's/^[[:blank:]]*//'"
    n_v = exec_cmd(get_cmd)

    ans = m_v + "-" + n_v

    pkg_ver['SCA'] = ans


def get_numpy_ver():
    pkg_ver['NUMPY'] = np.__version__


def get_ncc_ver():
    cnv_path = _path._common_kernel_path
    get_com = "/opt/nec/ve/bin/nreadelf -dW " + cnv_path + \
        "|/usr/bin/grep \"/opt/nec/ve/ncc\" | " \
        "/usr/bin/grep -o -E \"([0-9]+\\.){1}[0-9]+(\\.[0-9]+)?\" | " \
        "/usr/bin/head -n1"
    out = exec_cmd(get_com)
    ans = chk_data(out)
    pkg_ver['NCC'] = ans


def get_pkg_version():
    base_ver = ''

    get_nlc_ver()
    if pkg_ver['NLC'] == '':
        for key in pkg_ver:
            pkg_ver[key] = 'Not Available'
    else:
        base_ver = pkg_ver['NLC']

        get_asl_ver(base_ver)
        get_blas_ver(base_ver)
        get_lapack_ver(base_ver)
        get_sca_ver(base_ver)
        get_numpy_ver()
        get_ncc_ver()


def show_config():
    """Shows library versions in the system on which NLCPy is running.

    This function prints the versions of the following
    libraries and compiler on which NLCPy is running.

    - NLC: Numeric Library Collection
    - ASL: Advanced Scientific Library
    - BLAS: Basic Linear Algebra Subprograms
    - LAPACK: Linear Algebra PACKage
    - SCA: Stencil Code Accelerator
    - NumPy: Fundamental package for scientific computing in Python
    - ncc: NEC C/C++ compiler

    Here, the version of ncc indicates that it was used
    when the shared objects of NLCPy were built.

    Examples
    --------
    >>> import nlcpy as vp
    >>> vp.show_config()  # doctest: +SKIP
    NLC             : 2.2.0
         ASL        : 2.2-4.el7
         BLAS       : 2.3-1.el7
         LAPACK     : 2.1-1.el7
         SCA        : 3.2-2.el7
    NumPy           : 1.20.3
    ncc(build)      : 3.3.0

    """
    get_pkg_version()

    print('{0:15} : {1:9}'.format("NLC", pkg_ver['NLC']))
    print('     {0:10} : {1:9}'.format("ASL", pkg_ver['ASL']))
    print('     {0:10} : {1:9}'.format("BLAS", pkg_ver['BLAS']))
    print('     {0:10} : {1:9}'.format("LAPACK", pkg_ver['LAPACK']))
    print('     {0:10} : {1:9}'.format("SCA", pkg_ver['SCA']))
    print('{0:15} : {1:9}'.format("NumPy", pkg_ver['NUMPY']))
    print('{0:15} : {1:9}'.format("ncc(build)", pkg_ver['NCC']))
