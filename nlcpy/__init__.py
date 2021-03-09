import numpy  # NOQA
from numpy import _NoValue  # NOQA
import os  # NOQA
from distutils.version import StrictVersion  # NOQA

# -----------------------------------------------------------------------------
# set environment variable
# -----------------------------------------------------------------------------
ve_num_threads = os.environ.get('VE_OMP_NUM_THREADS', '8')
os.environ['VE_OMP_NUM_THREADS'] = ve_num_threads
from nlcpy import _path  # NOQA
_here = _path._here
_ve_ld_library_path = os.environ.get('VE_LD_LIBRARY_PATH', '')
os.environ['VE_LD_LIBRARY_PATH'] = _here + '/lib:' + _ve_ld_library_path

# XXX: This is a temporary workaround
# find NCC directory and set VE_LD_PRELOAD
base = '/opt/nec/ve/ncc'
files = os.listdir(base)
dirs = [os.path.join(base, f)
        for f in files
        if os.path.isdir(os.path.join(base, f))]
dirs.sort(key=lambda s: [int(u) for u in os.path.basename(s).split('.')], reverse=True)
NCC_PATH = dirs[0]
NCC_COMP_VER = '3.0.7'
if StrictVersion(os.path.basename(NCC_PATH)) >= StrictVersion(NCC_COMP_VER):
    _ve_ld_preload = os.environ.get('VE_LD_PRELOAD', '')
    os.environ['VE_LD_PRELOAD'] = NCC_PATH + '/lib/libncc.so.2:' + _ve_ld_preload

# --------------------------------------------------
# parameter
# --------------------------------------------------
from nlcpy.core import set_boundary_size  # NOQA
from nlcpy.core import get_boundary_size  # NOQA
from nlcpy import _version  # NOQA
__version__ = _version.__version__

# --------------------------------------------------
# ndarray
# --------------------------------------------------
from nlcpy.core import ndarray  # NOQA

# --------------------------------------------------
# ve offload
# --------------------------------------------------
from nlcpy import veo  # NOQA

# --------------------------------------------------
# Array Creation routines
# --------------------------------------------------
from nlcpy.creation.basic import empty   # NOQA
from nlcpy.creation.basic import empty_like  # NOQA
from nlcpy.creation.basic import eye  # NOQA
from nlcpy.creation.basic import identity  # NOQA
from nlcpy.creation.basic import ones  # NOQA
from nlcpy.creation.basic import ones_like  # NOQA
from nlcpy.creation.basic import zeros  # NOQA
from nlcpy.creation.basic import zeros_like  # NOQA
from nlcpy.creation.basic import full  # NOQA
from nlcpy.creation.basic import full_like  # NOQA
from nlcpy.creation.from_data import array  # NOQA
from nlcpy.creation.from_data import asarray  # NOQA
from nlcpy.creation.from_data import asanyarray  # NOQA
from nlcpy.creation.from_data import copy  # NOQA
from nlcpy.creation.from_data import fromfile  # NOQA
from nlcpy.creation.from_data import loadtxt  # NOQA
from nlcpy.creation.ranges import arange  # NOQA
from nlcpy.creation.ranges import linspace  # NOQA
from nlcpy.creation.ranges import logspace  # NOQA
from nlcpy.creation.ranges import meshgrid  # NOQA
from nlcpy.creation.matrices import diag  # NOQA

# --------------------------------------------------
# Array Manipulation routines
# --------------------------------------------------
from nlcpy.manipulation.basic import shape  # NOQA
from nlcpy.manipulation.shape import reshape  # NOQA
from nlcpy.manipulation.shape import ravel  # NOQA
from nlcpy.manipulation.trans import moveaxis  # NOQA
from nlcpy.manipulation.trans import rollaxis  # NOQA
from nlcpy.manipulation.trans import transpose  # NOQA
from nlcpy.manipulation.dims import broadcast_to  # NOQA
from nlcpy.manipulation.dims import expand_dims  # NOQA
from nlcpy.manipulation.dims import squeeze  # NOQA
from nlcpy.manipulation.join import concatenate  # NOQA
from nlcpy.manipulation.join import stack  # NOQA
from nlcpy.manipulation.join import hstack  # NOQA
from nlcpy.manipulation.join import vstack  # NOQA
from nlcpy.manipulation.add_remove import resize  # NOQA
from nlcpy.manipulation.add_remove import append  # NOQA
from nlcpy.manipulation.add_remove import insert  # NOQA
from nlcpy.manipulation.add_remove import delete  # NOQA
from nlcpy.manipulation.tiling import tile  # NOQA
from nlcpy.manipulation.tiling import repeat  # NOQA
from nlcpy.manipulation.basic import copyto  # NOQA
from nlcpy.manipulation.rearranging import flip  # NOQA
from nlcpy.manipulation.rearranging import fliplr  # NOQA
from nlcpy.manipulation.rearranging import flipud  # NOQA

# --------------------------------------------------
# ufunc operations
# --------------------------------------------------
from nlcpy.ufuncs import *  # NOQA
from nlcpy.ufuncs import ufunc  # NOQA

# --------------------------------------------------
# mathmatical functions
# --------------------------------------------------
from nlcpy.math.math import *  # NOQA

# --------------------------------------------------
# statistics
# --------------------------------------------------
from nlcpy.statistics.order import amax  # NOQA
from nlcpy.statistics.order import amin  # NOQA
from nlcpy.statistics.order import nanmax  # NOQA
from nlcpy.statistics.order import nanmin  # NOQA
from nlcpy.statistics.order import ptp # NOQA
from nlcpy.statistics.order import percentile # NOQA
from nlcpy.statistics.order import nanpercentile # NOQA
from nlcpy.statistics.order import quantile # NOQA
from nlcpy.statistics.order import nanquantile # NOQA
from nlcpy.statistics.average import median # NOQA
from nlcpy.statistics.average import average # NOQA
from nlcpy.statistics.average import mean # NOQA
from nlcpy.statistics.average import std # NOQA
from nlcpy.statistics.average import var # NOQA
from nlcpy.statistics.average import nanmedian # NOQA
from nlcpy.statistics.average import nanmean # NOQA
from nlcpy.statistics.average import nanstd # NOQA
from nlcpy.statistics.average import nanvar # NOQA
from nlcpy.statistics.correlating import corrcoef # NOQA
from nlcpy.statistics.correlating import correlate # NOQA
from nlcpy.statistics.correlating import cov # NOQA
from nlcpy.statistics.histograms import histogram # NOQA
from nlcpy.statistics.histograms import histogram2d # NOQA
from nlcpy.statistics.histograms import histogramdd # NOQA
from nlcpy.statistics.histograms import bincount # NOQA
from nlcpy.statistics.histograms import histogram_bin_edges # NOQA
from nlcpy.statistics.histograms import digitize # NOQA
# --------------------------------------------------
# logic functions
# --------------------------------------------------
from nlcpy.logic.testing import all  # NOQA
from nlcpy.logic.testing import any  # NOQA

# --------------------------------------------------
# linear algebra
# --------------------------------------------------
from nlcpy.linalg.products import dot  # NOQA
from nlcpy.linalg.products import inner  # NOQA
from nlcpy.linalg.products import outer  # NOQA

# --------------------------------------------------
# Indexing functions
# --------------------------------------------------
from nlcpy.indexing.generate import where  # NOQA
from nlcpy.indexing.indexing import take  # NOQA
from nlcpy.indexing.indexing import diagonal  # NOQA

# --------------------------------------------------
# searching
# --------------------------------------------------
from nlcpy.core.searching import argmax  # NOQA
from nlcpy.core.searching import argmin  # NOQA
from nlcpy.core.searching import nonzero  # NOQA
from nlcpy.core.searching import argwhere  # NOQA

# --------------------------------------------------
# sorting
# --------------------------------------------------
from nlcpy.sorting.sort import sort  # NOQA
from nlcpy.sorting.sort import argsort  # NOQA

# --------------------------------------------------
# misc
# --------------------------------------------------
from nlcpy.core.core import may_share_memory  # NOQA

# --------------------------------------------------
# Input and Output
# --------------------------------------------------
from nlcpy.io.npz import NpzFile  # NOQA
from nlcpy.io.npz import load  # NOQA

# =============================================================================
# Data types (borrowed from NumPy)
#
# https://docs.scipy.org/doc/numpy/reference/arrays.scalars.html
# =============================================================================

# -----------------------------------------------------------------------------
# Generic types
# -----------------------------------------------------------------------------
from numpy import complexfloating  # NOQA
from numpy import floating  # NOQA
from numpy import generic  # NOQA
from numpy import inexact  # NOQA
from numpy import integer  # NOQA
from numpy import number  # NOQA
from numpy import signedinteger  # NOQA
from numpy import unsignedinteger  # NOQA

# -----------------------------------------------------------------------------
# Booleans
# -----------------------------------------------------------------------------
from numpy import bool  # NOQA
from numpy import bool_  # NOQA
from numpy import bool8  # NOQA

# -----------------------------------------------------------------------------
# Integers
# -----------------------------------------------------------------------------
from numpy import byte  # NOQA
from numpy import short  # NOQA
from numpy import intc  # NOQA
from numpy import int_  # NOQA
from numpy import longlong  # NOQA
from numpy import intp  # NOQA
from numpy import int8  # NOQA
from numpy import int16  # NOQA
from numpy import int32  # NOQA
from numpy import int64  # NOQA
from numpy import int  # NOQA

# -----------------------------------------------------------------------------
# Unsigned integers
# -----------------------------------------------------------------------------
from numpy import ubyte  # NOQA
from numpy import ushort  # NOQA
from numpy import uintc  # NOQA
from numpy import uint  # NOQA
from numpy import ulonglong  # NOQA
from numpy import uintp  # NOQA
from numpy import uint8  # NOQA
from numpy import uint16  # NOQA
from numpy import uint32  # NOQA
from numpy import uint64  # NOQA

# -----------------------------------------------------------------------------
# Floating-point numbers
# -----------------------------------------------------------------------------
from numpy import half  # NOQA
from numpy import single  # NOQA
from numpy import double  # NOQA
from numpy import float_  # NOQA
from numpy import longfloat  # NOQA
from numpy import float16  # NOQA
from numpy import float32  # NOQA
from numpy import float64  # NOQA
from numpy import float  # NOQA

# from numpy import float96
# from numpy import float128

# -----------------------------------------------------------------------------
# Complex floating-point numbers
# -----------------------------------------------------------------------------
from numpy import csingle  # NOQA
from numpy import complex  # NOQA
from numpy import complex_  # NOQA
from numpy import complex64  # NOQA
from numpy import complex128  # NOQA

# from numpy import complex192
# from numpy import complex256
# from numpy import clongfloat


# =============================================================================
# Constants (borrowed from NumPy)
#
# https://docs.scipy.org/doc/numpy/reference/constants.html
# =============================================================================
from numpy import Inf  # NOQA
from numpy import Infinity  # NOQA
from numpy import NAN  # NOQA
from numpy import NINF  # NOQA
from numpy import NZERO  # NOQA
from numpy import NaN  # NOQA
from numpy import PINF  # NOQA
from numpy import PZERO  # NOQA
from numpy import e  # NOQA
from numpy import euler_gamma  # NOQA
from numpy import inf  # NOQA
from numpy import infty  # NOQA
from numpy import nan  # NOQA
from numpy import newaxis  # NOQA
from numpy import pi  # NOQA

# -----------------------------------------------------------------------------
# Generic types
# -----------------------------------------------------------------------------
from numpy import complexfloating  # NOQA

# -----------------------------------------------------------------------------
# Data Types
# -----------------------------------------------------------------------------
from numpy import dtype  # NOQA
from nlcpy.datatype.getlimits import iinfo  # NOQA
from nlcpy.datatype.getlimits import finfo  # NOQA

# -----------------------------------------------------------------------------
# Other modules
# -----------------------------------------------------------------------------
from numpy import set_printoptions  # NOQA
from numpy import get_printoptions  # NOQA
from nlcpy.error_handler.error_handler import *  # NOQA

# --------------------------------------------------
# Call veo initialization when nlcpy is imported
# --------------------------------------------------
v = veo.VeoAlloc()  # initialize veo process
args = ()
try:
    req = v.lib.func[b"asl_library_initialize"](v.ctx, None)
    req.wait_result()
except KeyError:
    pass


# -----------------------------------------------------------------------------
# random
# -----------------------------------------------------------------------------
from nlcpy import random  # NOQA


# -----------------------------------------------------------------------------
# fft
# -----------------------------------------------------------------------------
from nlcpy import fft # NOQA


# -----------------------------------------------------------------------------
# warm up
# -----------------------------------------------------------------------------
from nlcpy._warmup import _warmup  # NOQA
_warmup()
