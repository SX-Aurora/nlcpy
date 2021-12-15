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
from setuptools.command.sdist import sdist
from wheel.bdist_wheel import bdist_wheel
import numpy
from os import path
from io import open
import os
import os.path
import subprocess
from Cython.Build import cythonize

# get directroy where this file exists on
here = path.abspath(path.dirname(__file__))

# get __version__ from _version.py
exec(open(here + '/nlcpy/_version.py').read())

# get long_description from README.md
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# find NLC directory
base = '/opt/nec/ve/nlc'
files = os.listdir(base)
dirs = [os.path.join(base, f)
        for f in files
        if os.path.isdir(os.path.join(base, f))]
dirs.sort(key=lambda s: [int(u) for u in os.path.basename(s).split('.')], reverse=True)
NLC_PATH = dirs[0]

#####################################################
# make extensions
#####################################################
extensions = []

extra_compile_args = ['-O2']

ext_modules = {
    'veo': [
        'nlcpy.veo._veo',
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
        'veo', '/opt/nec/ve/veos/include',
        numpy.get_include(),
    ],
    'mempool': [
        'veo', '/opt/nec/ve/veos/include',
        numpy.get_include(),
    ],
    'random': [
        NLC_PATH + '/include',
        numpy.get_include(),
    ],
    'fft': [
        NLC_PATH + '/include',
        numpy.get_include(),
    ],
    'request': [
        'veo', '/opt/nec/ve/veos/include',
        numpy.get_include(),
    ],
    'others': [
        'veo', '/opt/nec/ve/veos/include',
        numpy.get_include(),
    ],
}

libraries = {
    'veo': [
        'veo',
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
        '-Wl,-rpath=/opt/nec/ve/veos/lib64'
    ],
    'mempool': [
        '-Wl,-rpath=/opt/nec/ve/veos/lib64'
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
        )
        extensions.append(ext)


#####################################################
# set custom cmdclass
#####################################################

class custom_bdist_wheel(bdist_wheel):
    def run(self):
        shutil.copyfile('MANIFEST.in.wheel', 'MANIFEST.in')
        ret = subprocess.call(['make', '-f', os.getcwd() + '/Makefile'])
        if ret != 0:
            raise RuntimeError('Failed to build VE kernel.')
        bdist_wheel.run(self)


class custom_build_ext(build_ext.build_ext):
    def run(self):
        ret = subprocess.call(['make', '-f', os.getcwd() + '/Makefile'])
        if ret != 0:
            raise RuntimeError('Failed to build VE kernel.')
        build_ext.build_ext.run(self)


class custom_sdist(sdist):
    def run(self):
        shutil.copyfile('MANIFEST.in.sdist', 'MANIFEST.in')
        sdist.run(self)


cmdclass = {}
cmdclass['bdist_wheel'] = custom_bdist_wheel
cmdclass['build_ext'] = custom_build_ext
cmdclass['sdist'] = custom_sdist


#####################################################
# show build stats
#####################################################

def show_build_stats(NLC_PATH):
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
    print("  NLC    path   :", NLC_PATH)
    print("  GCC    version:", gcc_version)
    print("  PYTHON version:", python_version)
    print("  CYTHON version:", cython_version)
    print("  NUMPY  version:", numpy_version)
    print("===========================================")


show_build_stats(NLC_PATH)


setup(
    version=__version__,  # NOQA
    cmdclass=cmdclass,
    ext_modules=cythonize(
        extensions,
        compiler_directives={'embedsignature': True},
        language_level=3
    ),
)
