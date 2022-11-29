import numpy  # NOQA
from numpy import _NoValue  # NOQA
from nlcpy.__config__ import show_config  # NOQA
from nlcpy import _path

# -----------------------------------------------------------------------------
# set environment variable
# -----------------------------------------------------------------------------
from nlcpy import _environment
_environment._set_ve_ld_library_path()
_environment._set_ve_ld_preload()

# --------------------------------------------------
# parameter
# --------------------------------------------------
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
# MaskedArray
# --------------------------------------------------
from nlcpy.ma import MaskedArray  # NOQA

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
from nlcpy.creation.matrices import diagflat  # NOQA
from nlcpy.creation.matrices import tri  # NOQA
from nlcpy.creation.matrices import tril  # NOQA
from nlcpy.creation.matrices import triu  # NOQA

# --------------------------------------------------
# Array Manipulation routines
# --------------------------------------------------
from nlcpy.manipulation.basic import shape  # NOQA
from nlcpy.manipulation.shape import reshape  # NOQA
from nlcpy.manipulation.shape import ravel  # NOQA
from nlcpy.manipulation.trans import moveaxis  # NOQA
from nlcpy.manipulation.trans import rollaxis  # NOQA
from nlcpy.manipulation.trans import swapaxes  # NOQA
from nlcpy.manipulation.trans import transpose  # NOQA
from nlcpy.manipulation.dims import atleast_1d  # NOQA
from nlcpy.manipulation.dims import atleast_2d  # NOQA
from nlcpy.manipulation.dims import atleast_3d  # NOQA
from nlcpy.manipulation.dims import broadcast_arrays  # NOQA
from nlcpy.manipulation.dims import broadcast_to  # NOQA
from nlcpy.manipulation.dims import expand_dims  # NOQA
from nlcpy.manipulation.dims import squeeze  # NOQA
from nlcpy.manipulation.join import block  # NOQA
from nlcpy.manipulation.join import concatenate  # NOQA
from nlcpy.manipulation.join import stack  # NOQA
from nlcpy.manipulation.join import hstack  # NOQA
from nlcpy.manipulation.join import vstack  # NOQA
from nlcpy.manipulation.split import split  # NOQA
from nlcpy.manipulation.split import hsplit  # NOQA
from nlcpy.manipulation.split import vsplit  # NOQA
from nlcpy.manipulation.add_remove import resize  # NOQA
from nlcpy.manipulation.add_remove import append  # NOQA
from nlcpy.manipulation.add_remove import insert  # NOQA
from nlcpy.manipulation.add_remove import delete  # NOQA
from nlcpy.manipulation.add_remove import unique  # NOQA
from nlcpy.manipulation.tiling import tile  # NOQA
from nlcpy.manipulation.tiling import repeat  # NOQA
from nlcpy.manipulation.basic import copyto  # NOQA
from nlcpy.manipulation.rearranging import flip  # NOQA
from nlcpy.manipulation.rearranging import fliplr  # NOQA
from nlcpy.manipulation.rearranging import flipud  # NOQA
from nlcpy.manipulation.rearranging import roll  # NOQA

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
from nlcpy.statistics.order import max  # NOQA
from nlcpy.statistics.order import amin  # NOQA
from nlcpy.statistics.order import min  # NOQA
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
from nlcpy.indexing.generate import diag_indices  # NOQA
from nlcpy.indexing.generate import where  # NOQA
from nlcpy.indexing.indexing import take  # NOQA
from nlcpy.indexing.indexing import diagonal  # NOQA
from nlcpy.indexing.indexing import select  # NOQA
from nlcpy.indexing.inserting import fill_diagonal  # NOQA

# --------------------------------------------------
# searching
# --------------------------------------------------
from nlcpy.core.searching import argmax  # NOQA
from nlcpy.core.searching import argmin  # NOQA
from nlcpy.core.searching import nonzero  # NOQA
from nlcpy.core.searching import argwhere  # NOQA
from nlcpy.sorting.search import nanargmax  # NOQA
from nlcpy.sorting.search import nanargmin  # NOQA

# --------------------------------------------------
# sorting
# --------------------------------------------------
from nlcpy.sorting.sort import sort  # NOQA
from nlcpy.sorting.sort import argsort  # NOQA

# --------------------------------------------------
# counting
# --------------------------------------------------
from nlcpy.sorting.count import count_nonzero  # NOQA

# --------------------------------------------------
# misc
# --------------------------------------------------
from nlcpy.core.core import may_share_memory  # NOQA

# --------------------------------------------------
# Input and Output
# --------------------------------------------------
from nlcpy.io.npz import NpzFile  # NOQA
from nlcpy.io.npz import load  # NOQA
from nlcpy.io.npz import save  # NOQA
from nlcpy.io.npz import savez  # NOQA
from nlcpy.io.npz import savez_compressed  # NOQA
from nlcpy.io.text import savetxt  # NOQA

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

# from numpy import float96
# from numpy import float128

# -----------------------------------------------------------------------------
# Complex floating-point numbers
# -----------------------------------------------------------------------------
from numpy import csingle  # NOQA
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
# create VE process and initialize
# --------------------------------------------------
from nlcpy.venode._venode import _create_venode_pool  # NOQA
_create_venode_pool()  # create veo process on VE node.


# -----------------------------------------------------------------------------
# get include file path
# -----------------------------------------------------------------------------
def get_include():
    """Returns the directory path that contains the NLCPy \\*.h header files.

    """
    return _path._include_path

# -----------------------------------------------------------------------------
# random
# -----------------------------------------------------------------------------
from nlcpy import random  # NOQA


# -----------------------------------------------------------------------------
# SCA
# -----------------------------------------------------------------------------
from nlcpy import sca  # NOQA


# -----------------------------------------------------------------------------
# fft
# -----------------------------------------------------------------------------
from nlcpy import fft # NOQA


# --------------------------------------------------
# JIT
# --------------------------------------------------

from nlcpy import ve_types  # NOQA
from nlcpy import jit  # NOQA


# -----------------------------------------------------------------------------
# numpy wrap
# -----------------------------------------------------------------------------
from nlcpy.wrapper.numpy_wrap import _make_wrap_func  # NOQA
from nlcpy.wrapper.numpy_wrap import _make_wrap_method  # NOQA


def __getattr__(attr):
    if attr in (
            'asmatrix',
            'byte_bounds',
            'get_array_wrap',
            'getbufsize',
            'geterrcall',
            'mafromtxt',
            'maximum_sctype',
            'memmap',
            'min_scalar_type',
            'mintypecode',
            'nditer',
            'nested_iters',
            'set_numeric_ops',
            'setbufsize',
            'seterrcall',
            'shares_memory',
            'test',
            'trim_zeros',
            'vectorize',
            'who'
    ):
        raise AttributeError("module 'nlcpy' has no attribute '{}'.".format(attr))
    try:
        f = getattr(numpy, attr)
    except AttributeError as _err:
        raise AttributeError(
            "module 'nlcpy' has no attribute '{}'.".format(attr)) from _err
    if not callable(f):
        raise AttributeError("module 'nlcpy' has no attribute '{}'.".format(attr))
    return _make_wrap_func(f)
