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


import nlcpy
from nlcpy.jit import kernel
from nlcpy.__config__ import get_nlc_ver

import os
import datetime
import subprocess
import tempfile

from nlcpy.venode import VE


class nccException(Exception):
    pass


class nfortException(Exception):
    pass


class ncppException(Exception):
    pass


def _exec_cmd(cmd, log_stream=None):
    try:
        env = os.environ
        log = subprocess.check_output(cmd, env=env,
                                      stderr=subprocess.STDOUT,
                                      universal_newlines=True)
        if log_stream is not None:
            msg = 'cmd: {}\n'.format(' '.join(cmd)) + log
            log_stream.write(msg)
        return log
    except subprocess.CalledProcessError as e:
        msg = ('`{0}` command returns non-zero exit status. \n'
               'command: {1}\n'
               'return-code: {2}\n'
               'stdout/stderr: \n'
               '{3}'.format(' '.join(cmd),
                            e.cmd,
                            e.returncode,
                            e.output))
        if log_stream is not None:
            log_stream.write(msg)
        if 'ncc' in os.path.basename(cmd[0]):
            raise nccException(msg) from None
        elif 'nfort' in os.path.basename(cmd[0]):
            raise nfortException(msg) from None
        elif 'nc++' in os.path.basename(cmd[0]):
            raise ncppException(msg) from None
        else:
            raise  # pragma: no cover
    except Exception:
        raise


def get_id():
    now = datetime.datetime.now()
    return (str(os.getpid()) + '_' +
            str(now.date()) + '.' + str(now.time()))


def get_default_cflags(openmp=True, opt_level=2, debug=False):
    """Gets default compiler flags.

    Parameters
    ----------
    openmp : bool
        Enables OpenMP or not.
        Defaults to ``True``.
    opt_level : int
        Optimization level.
        Defaults to ``2``.
    debug : bool
        Adding ``-g`` option or not.
        Defaults to ``False``.

    Returns
    -------
    cflags : tuple of str
        Compiler flags.

    Examples
    --------
    >>> import nlcpy
    >>> from pprint import pprint
    >>> cflags = nlcpy.jit.get_default_cflags(openmp=True, opt_level=2, debug=False)
    >>> pprint(cflags)  # doctest: +SKIP
    ('-c',
     '-fpic',
     '-O2',
     '-I',
     '/your/path/to/nlcpy/include',
     '-fopenmp')

    """

    if type(opt_level) is not int:
        raise TypeError('Invalid type of opt_level. Expected int.')
    _opt_level = int(opt_level)
    if _opt_level < 0 or _opt_level > 4:
        raise ValueError('Invalid value of opt_level. '
                         'Expected 0 <= opt_level <= 4.')
    cflags = (
        '-c',
        '-fpic',
        '-O' + str(_opt_level),
        '-I',
        nlcpy.get_include(),
    )
    if openmp:
        cflags += ('-fopenmp',)
    if debug:
        cflags += ('-g',)
    return cflags


def get_default_ldflags(openmp=True):
    """Gets default linker flags.

    Parameters
    ----------
    openmp : bool
        Enables OpenMP or not.
        Defaults to ``True``.

    Returns
    -------
    ldflags : tuple of str
        Linker flags.

    Examples
    --------
    >>> import nlcpy
    >>> nlcpy.jit.get_default_ldflags(openmp=True)
    ('-fpic', '-shared', '-fopenmp')

    """

    ldflags = (
        '-fpic',
        '-shared',
    )
    if openmp:
        ldflags += ('-fopenmp',)
    return ldflags


class CustomVELibrary:
    """Custom VE library class.

    This class provides simple wrapper functionalities related to
    compiling by ncc/nc++/nfort and linking with shared objects for the VE.

    It can be used to either compile and load from a C/C++/Fortran source
    or load from a pre-built shared object.

    The instance of this class holds a handle of the loaded VE library.
    To retrieve a callable VE function from the instance,
    please call a method :meth:`get_function`.

    Parameters
    ----------
    code : str
        C/C++/Fortran source code. Mutually exclusive with ``path``.
    path : str
        Pre-built shared object path. Mutually exclusive with ``code``.
    cflags : tuple of str
        Compilation flags.
        The default is given by :func:`nlcpy.jit.get_default_cflags`.
    ldflags : tuple of str
        Linking flags.
        The default is given by :func:`nlcpy.jit.get_default_ldflags`.
    log_stream : object
        Pass either ``sys.stdout`` or a writable file object to which
        the compiler output will be written.
        Defaults to ``None``.
    compiler : str
        Command to be used for compiling and linking.
        Defaults to ``'/opt/nec/be/bin/ncc'``.
        You can also specify ``'nfort'`` or ``'nc++'``.
    use_nlc : bool
        Whether the VE source links with NLC or not.
        If set to ``True``, the OpenMP and 64bit integer library of NLC
        will be set to the ``cflags`` and ``ldflags`` internally.
        Defaults to ``False``.
    ftrace : bool
        Whether the VE source links with ftrace or not.
        Defaults to ``False``.
    dist_dir : str
        Directory path that stores the source code, the object file,
        and the shared object file.
        By default, the files are stored into a temporary directory
        and are removed after constructing this instance.

    Note
    ----
        Only the OpenMP & 64bit integer version of the NLC can be used.

    """

    def __init__(self, *, code=None, path=None,
                 cflags=None, ldflags=None,
                 log_stream=None, compiler='/opt/nec/ve/bin/ncc',
                 use_nlc=False, ftrace=False, dist_dir=None):
        if (code is None) == (path is None):
            raise TypeError(
                'Do not specify kwargs code and path at the same time.'
            )

        if path is None:
            if type(code) not in (str, bytes):
                raise TypeError('code must be given str or bytes.')
            elif type(code) is bytes:
                code = code.decode()
        if code is None:
            if type(path) not in (str, bytes):
                raise TypeError('path must be given str or bytes.')
            elif type(path) is bytes:
                path = path.decode()
        self._code = code
        self._path = path

        self._src_path = None
        self._obj_path = None
        self._lib_path = None

        if path is None:
            self._id = get_id()
        else:
            self._id = None
        if type(cflags) in (tuple, list):
            self._cflags = tuple(cflags)
        elif cflags is None:
            self._cflags = get_default_cflags()
        else:
            raise TypeError('cflags must be given tuple or list.')

        if type(ldflags) in (tuple, list):
            self._ldflags = tuple(ldflags)
        elif ldflags is None:
            self._ldflags = get_default_ldflags()
        else:
            raise TypeError('ldflags must be given tuple or list.')

        self._log_stream = log_stream

        if type(compiler) is not str:
            raise TypeError('compiler must be given str.')
        if 'ncc' in os.path.basename(compiler):
            self._suffix = '.c'
        elif 'nfort' in os.path.basename(compiler):
            self._suffix = '.f03'
        elif 'nc++' in os.path.basename(compiler):
            self._suffix = '.cpp'
        else:
            raise ValueError('unknown compiler command: `{}`'.format(compiler))
        self._compiler = compiler
        self._use_nlc = use_nlc
        self._ftrace = ftrace

        if dist_dir is not None or path is not None:
            self._load_lib(dist_dir, self._id)
        else:
            # make tempdir
            with tempfile.TemporaryDirectory() as dist_dir:
                self._load_lib(dist_dir, self._id)

        self._valid = True

    def get_function(self, func_name, args_type=(),
                     ret_type=nlcpy.ve_types.void):
        """Retrieve a VE function by its name from a library.

        Parameters
        ----------
        func_name : str
            Name of the VE function.
        args_type : tuple of str
            Data types for the arguments of the VE function.
            You can also specify this from
            :ref:`constants <label_ve_types_consts>`.
        ret_type : str
            Data type for the return value of the VE function.
            You can also specify this from
            :ref:`constants <label_ve_types_consts>`.

        Returns
        -------
        kernel : :class:`nlcpy.jit.CustomVEKernel`
            The callable VE kernel.
        """

        if type(func_name) not in (str, bytes):
            raise TypeError('func_name must be given str or bytes.')
        if type(args_type) not in (list, tuple):
            raise TypeError('args_type must be given list or tuple.')
        if type(ret_type) is not str:
            raise TypeError('ret_type must be given str.')

        if type(func_name) is str:
            func_name = func_name.encode('utf-8')
        func = self._lib.find_function(func_name)
        func.args_type(*args_type)
        func.ret_type(ret_type)
        return kernel.CustomVEKernel(func, self)

    def _check_dist(self, dirc, name):
        dist_dir = os.path.abspath(dirc)
        dist_path = os.path.join(dist_dir, name)
        if not os.path.isdir(dist_dir):
            os.makedirs(dist_dir, exist_ok=True)
        self._src_path = dist_path + self._suffix
        self._obj_path = dist_path + '.o'
        self._lib_path = dist_path + '.so'

    def _make_src(self):
        with open(self._src_path, 'w') as src:
            src.write(self._code)

    def _make_obj(self):
        cmd = (self._compiler, self._src_path) + self._cflags + ('-o', self._obj_path)
        if self._use_nlc:
            nlc_ver = get_nlc_ver()
            if 'ncc' in os.path.basename(self._compiler) or \
                    'nc++' in os.path.basename(self._compiler):
                cmd += ('-I/opt/nec/ve/nlc/{}/include/inc_i64/'.format(nlc_ver),)
            if 'nfort' in os.path.basename(self._compiler):
                cmd += ('-I/opt/nec/ve/nlc/{}/include/mod_i64/'.format(nlc_ver),)
        if self._ftrace:
            cmd += ('-ftrace',)
        _ = _exec_cmd(cmd, log_stream=self._log_stream)

    def _make_so(self):
        cmd = (self._compiler, self._obj_path) + self._ldflags + ('-o', self._lib_path)
        if self._use_nlc:
            cmd += (
                '-lasl_openmp_i64',
                '-laslfftw3_i64',
                '-llapack_i64',
                '-lblas_openmp_i64',
                '-lsca_openmp_i64',
                '-lheterosolver_openmp_i64',
                '-lsblas_openmp_i64',
                '-lcblas_i64',
            )
        if self._ftrace:
            cmd += ('-ftrace', '-lveftrace_p')
            # if '-fopenmp' in self._cflags:
            #     cmd += ('-ftrace', '-lveftrace_p')
            # else:
            #     cmd += ('-ftrace', '-lveftrace_t')
        _ = _exec_cmd(cmd, log_stream=self._log_stream)

    def _src2so(self):
        self._make_src()
        self._make_obj()
        self._make_so()

    def _load_lib(self, dist_dir, file_name):
        if self._path is None:
            self._check_dist(dist_dir, file_name)
            self._src2so()
        else:
            self._lib_path = self._path
        self._venode = VE()
        self._lib = self._venode.proc.load_library(
            self._lib_path.encode('utf-8'))

    def _is_valid(self):
        return self._valid

    def _deactivate(self):
        self._valid = False

    @property
    def id(self):
        return self._id

    def __repr__(self):  # pragma: no cover
        return '<CustomVELibrary({}\n)>'.format(
            '\n'.join([
                '\n* code:\n{}'.format(self._code),
                '* path: {}'.format(self._path),
                '* cflags: {}'.format(
                    ''.join([' {}'.format(s) for s in self._cflags])),
                '* ldflags: {}'.format(
                    ''.join([' {}'.format(s) for s in self._ldflags])),
                '* log_stream: {}'.format(self._log_stream),
                '* compiler: {}'.format(self._compiler),
                '* use_nlc: {}'.format(self._use_nlc),
                '* ftrace: {}'.format(self._ftrace),
                '* ID: {}'.format(self._id),
                '* src_path: {}'.format(self._src_path),
                '* obj_path: {}'.format(self._obj_path),
                '* lib_path: {}'.format(self._lib_path),
                '* lib: {}'.format(self._lib),
                '* venode: {}'.format(self._venode),
            ]))


def unload_library(ve_lib):
    """Unloads the shared library.

    Parameters
    ----------
    ve_lib : nlcpy.jit.CustomVELibrary
        Customized VE library.

    Restriction
    -----------
    Please avoid unloading the shared library linked with FTRACE.
    Otherwise, SIGSEGV may occur.

    """
    if type(ve_lib) is not CustomVELibrary:
        raise TypeError('unrecognized input type: `{}`'.format(type(ve_lib)))
    nlcpy.request.flush()
    if ve_lib._is_valid():
        ve_lib._venode.proc.unload_library(ve_lib._lib)
    ve_lib._deactivate()
