#####################################################
# #### Usage ####
#
# ## for development ##
# $ python setup.py build_ext -i -> build inplace
# or,
# $ sh build_inplace.sh
#
# ## create wheel package and install ##
# $ python setup.py bdist_wheel -> create wheel package
# $ cd dist
# $ pip install *.whl -> install nlcpy to your python directory
#
# ## quick install(build works on /tmp directory) ##
# $ pip install . -> install nlcpy to your python directory
#

import shutil
from setuptools import setup, Extension
from setuptools.command import build_ext
try:
    from setuptools.command import build
    from setuptools._distutils import dir_util
except ImportError:
    from distutils.command import build
    from distutils import dir_util
from setuptools.command.sdist import sdist
from wheel.bdist_wheel import bdist_wheel
import numpy
from os import path
from io import open
import os
import os.path
import subprocess
import sys
import argparse
from Cython.Build import cythonize

# get directroy where this file exists on
here = path.abspath(path.dirname(__file__))

# get __version__ from _version.py
exec(open(os.path.join(here, 'nlcpy/_version.py')).read())

# get long_description from README.md
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# find NLC directory
NLC_PATH = os.environ.get("NLC_HOME")
if NLC_PATH is None:
    raise RuntimeError('Undefined environment variable NLC_HOME. Please execute '
                       '`source /opt/nec/ve[3]/nlc/X.X.X/bin/nlcvars.sh`')


#####################################################
# parse arguments
#####################################################

targs_ref = ('vh', 've1', 've3')


def parse_targs(targs):
    ret = []
    for t in targs.split(','):
        if t not in targs_ref:
            raise ValueError
        ret.append(t)
    return ret


parser = argparse.ArgumentParser()
parser.add_argument(
    '--targ', metavar="COMMA_SEPARATED_TARGS", required=True,
    nargs=1, type=parse_targs,
    help='select comma separated build targets')
parser.add_argument(
    '--profile', action='store_true', default=False,
    help='enable profiling for Cython code')
parser.add_argument(
    '--debug', action='store_true', default=False,
    help='enable debugging for Cython code')
args, sys.argv = parser.parse_known_args(sys.argv)

#####################################################
# set macros and compiler directives
#####################################################
compiler_directives = {
    'embedsignature': True,
    'c_api_binop_methods': True,
}

define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

if args.profile:
    compiler_directives['profile'] = True
    compiler_directives['linetrace'] = True
    define_macros.append(('CYTHON_TRACE_NOGIL', '1'))
    define_macros.append(('CYTHON_TRACE', '1'))
if args.debug:
    compiler_directives['boundscheck'] = True
    compiler_directives['initializedcheck'] = True
    compiler_directives['nonecheck'] = True
    compiler_directives['overflowcheck'] = True
    define_macros.append(('_GLIBCXX_ASSERTIONS', None))
    define_macros.append(('_GLIBCXX_DEBUG', None))


#####################################################
# make extensions
#####################################################
extensions = []
extra_compile_args = ['-O2']
if args.debug:
    extra_compile_args = extra_compile_args + ['-g']

ext_modules = {
    'veo': [
        'nlcpy.veo._veo',
    ],
    'veosinfo': [
        'nlcpy.veosinfo._veosinfo',
    ],
    'venode': [
        'nlcpy.venode._venode',
    ],
    'mempool': [
        'nlcpy.mempool.mempool',
    ],
    'random': [
        'nlcpy.random.generator',
        'nlcpy.random._generator',
    ],
    'fft': [
        'nlcpy.fft._fft',
    ],
    'request': [
        'nlcpy.request.ve_kernel',
        'nlcpy.request.request',
    ],
    'others': [
        'nlcpy.core.core',
        'nlcpy.core.internal',
        'nlcpy.core.core',
        'nlcpy.core.vememory',
        'nlcpy.core.internal',
        'nlcpy.core.broadcast',
        'nlcpy.kernel_register.ve_kernel_register',
        'nlcpy.core.dtype',
        'nlcpy.core.manipulation',
        'nlcpy.core.sorting',
        'nlcpy.core.indexing',
        'nlcpy.ufuncs.ufuncs',
        'nlcpy.core.scalar',
        'nlcpy.ufuncs.reduce',
        'nlcpy.core.searching',
        'nlcpy.core.math',
        'nlcpy.linalg.cblas_wrapper',
        'nlcpy.math.math',
        'nlcpy.ufuncs.outer',
        'nlcpy.ufuncs.reduceat',
        'nlcpy.ufuncs.accumulate',
        'nlcpy.statistics.function_base',
        'nlcpy.statistics.average',
        'nlcpy.statistics.correlating',
        'nlcpy.statistics.histograms',
        'nlcpy.sca.utility',
        'nlcpy.sca.handle',
        'nlcpy.sca.kernel',
        'nlcpy.sca.internal',
        'nlcpy.sca.descriptor',
        'nlcpy.sca.description',
    ],
}

include_dirs = {
    'veo': [
        '/opt/nec/ve/veos/include',
        numpy.get_include(),
    ],
    'veosinfo': [
        '/opt/nec/ve/veos/include',
    ],
    'venode': [
        '/opt/nec/ve/veos/include',
        numpy.get_include(),
    ],
    'mempool': [
        '/opt/nec/ve/veos/include',
        'nlcpy/mempool',
        numpy.get_include(),
    ],
    'random': [
        os.path.join(NLC_PATH, 'include'),
        numpy.get_include(),
    ],
    'fft': [
        os.path.join(NLC_PATH, 'include'),
        numpy.get_include(),
    ],
    'request': [
        '/opt/nec/ve/veos/include',
        numpy.get_include(),
    ],
    'others': [
        '/opt/nec/ve/veos/include',
        numpy.get_include(),
    ],
}

libraries = {
    'veo': [
        'veo',
    ],
    'veosinfo': [
        'veosinfo',
    ],
    'venode': [
    ],
    'mempool': [
        'veo',
    ],
    'random': [
    ],
    'fft': [
    ],
    'request': [
    ],
    'others': [
    ],
}

library_dirs = {
    'veo': [
        'veo', '/opt/nec/ve/veos/lib64',
    ],
    'veosinfo': [
        'veosinfo', '/opt/nec/ve/veos/lib64',
    ],
    'venode': [
    ],
    'mempool': [
        'veo', '/opt/nec/ve/veos/lib64',
    ],
    'random': [
    ],
    'fft': [
    ],
    'request': [
    ],
    'others': [
    ],
}

extra_link_args = {
    'veo': [
        '-Wl,-rpath=/opt/nec/ve/veos/lib64', '-Wl,--enable-new-dtags'
    ],
    'veosinfo': [
        '-Wl,-rpath=/opt/nec/ve/veos/lib64', '-Wl,--enable-new-dtags'
    ],
    'venode': [
    ],
    'mempool': [
        '-Wl,-rpath=/opt/nec/ve/veos/lib64', '-Wl,--enable-new-dtags'
    ],
    'random': [
    ],
    'fft': [
    ],
    'request': [
    ],
    'others': [
    ],
}


for key in ext_modules:
    for module in ext_modules[key]:
        src = module.replace('.', '/') + '.pyx'
        ext = Extension(
            module,
            [src, ],
            extra_compile_args=extra_compile_args,
            libraries=libraries[key],
            library_dirs=library_dirs[key],
            include_dirs=include_dirs[key],
            extra_link_args=extra_link_args[key],
            define_macros=define_macros,
        )
        extensions.append(ext)


#####################################################
# show build stats
#####################################################

def show_build_stats():
    # get ncc version
    ret = subprocess.run(
        'ncc --version |& grep -o "[0-9]\.[0-9]\.[0-9]"',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True)
    if ret.returncode != 0:
        raise RuntimeError('Cannot find \'ncc\' command.')
    ncc_version = ret.stdout.decode().split('\n')[0]
    # get gcc version
    import platform
    ret = subprocess.run(
        'gcc --version |& grep -o "[0-9]\.[0-9]\.[0-9]"',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True)
    if ret.returncode != 0:
        raise RuntimeError('Cannot find \'gcc\' command.')
    gcc_version = ret.stdout.decode().split('\n')[0]
    # get python version
    python_version = platform.python_version()
    # get cython version
    import cython
    cython_version = cython.__version__
    # get numpy version
    numpy_version = numpy.__version__

    print("===============BUILDING STATS==============")
    print("  NCC    version:", ncc_version)
    print("  GCC    version:", gcc_version)
    print("  PYTHON version:", python_version)
    print("  CYTHON version:", cython_version)
    print("  NUMPY  version:", numpy_version)
    print("===========================================")


show_build_stats()


#####################################################
# set custom cmdclass
#####################################################

CFG_DIR = 'config'

# customize bdist_wheel


class vh_custom_bdist_wheel(bdist_wheel):
    def run(self):
        shutil.copyfile(os.path.join(CFG_DIR, 'MANIFEST.in.wheel.vh'), 'MANIFEST.in')
        super().run()


class ve1_custom_bdist_wheel(bdist_wheel):
    def run(self):
        shutil.copyfile(os.path.join(CFG_DIR, 'MANIFEST.in.wheel.ve1'), 'MANIFEST.in')
        ret = subprocess.call(['make', 'ARCH=ve1', '-f',
                               os.path.join(os.getcwd(), 'Makefile')])
        if ret != 0:
            raise RuntimeError('Failed to build VE1 kernel.')
        super().run()


class ve3_custom_bdist_wheel(bdist_wheel):
    def run(self):
        shutil.copyfile(os.path.join(CFG_DIR, 'MANIFEST.in.wheel.ve3'), 'MANIFEST.in')
        ret = subprocess.call(['make', 'ARCH=ve3', '-f',
                               os.path.join(os.getcwd(), 'Makefile')])
        if ret != 0:
            raise RuntimeError('Failed to build VE3 kernel.')
        super().run()


# customize build


class vh_custom_build(build.build):
    def run(self):
        # as-is
        super().run()


class ve1_custom_build(build.build):
    def run(self):
        self.build_lib += '-ve1'
        super().run()


class ve3_custom_build(build.build):
    def run(self):
        self.build_lib += '-ve3'
        super().run()


# customize build_ext


class vh_custom_build_ext(build_ext.build_ext):
    def run(self):
        # as-is
        super().run()

    def build_extensions(self):
        try:
            self.compiler.compiler_so.remove('-Wstrict-prototypes')
        except ValueError:
            pass
        super().build_extensions()


class ve1_custom_build_ext(build_ext.build_ext):
    def run(self):
        ret = subprocess.call(['make', 'ARCH=ve1', '-f',
                               os.path.join(os.getcwd(), 'Makefile')])
        if ret != 0:
            raise RuntimeError('Failed to build VE1 kernel.')
        super().run()


class ve3_custom_build_ext(build_ext.build_ext):
    def run(self):
        ret = subprocess.call(['make', 'ARCH=ve3', '-f',
                               os.path.join(os.getcwd(), 'Makefile')])
        if ret != 0:
            raise RuntimeError('Failed to build VE3 kernel.')
        super().run()


# customize sdist


class custom_sdist(sdist):
    def run(self):
        shutil.copyfile('MANIFEST.in.sdist', 'MANIFEST.in')
        super().run()


cmdclass = {}

flat_targs = sum(args.targ, [])
print("Build Targets:", flat_targs)
for bt in flat_targs:
    print("\n===Begin Building for %s ===" % bt)
    shutil.copyfile(os.path.join(CFG_DIR, f'setup_{bt}.cfg'), 'setup.cfg')

    cmdclass['bdist_wheel'] = locals()[f'{bt}_custom_bdist_wheel']
    cmdclass['build_ext'] = locals()[f'{bt}_custom_build_ext']
    cmdclass['build'] = locals()[f'{bt}_custom_build']
    cmdclass['sdist'] = custom_sdist
    kwargs = {'version': __version__,  # NOQA
              'cmdclass': cmdclass}
    if bt == 'vh':
        kwargs['install_requires'] = ['numpy>=1.17',
                                      'nlcpy_ve1_kernel==' + __version__, # NOQA
                                      'nlcpy_ve3_kernel==' + __version__] # NOQA
        kwargs['ext_modules'] = cythonize(
            extensions,
            compiler_directives=compiler_directives,
            language_level=3)

    setup(**kwargs)
    dir_util._path_created.clear()
    print("===End Building for %s ===" % bt)
