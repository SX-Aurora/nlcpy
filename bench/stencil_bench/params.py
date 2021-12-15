from pprint import pprint

ITER = 1000
STENCIL_SCALE = 6
COEF = .01
DTYPES = ('f4',)

nlcpy_sca = [
    'NLCPy(SCA) / VE',
    'red',
    'o'
]
cupy_fusion = [
    'CuPy(Fusion) / GPU',
    'darkseagreen',
    'D'
]
numba_cuda = [
    'Numba / GPU',
    'darkgreen',
    'v'
]
pystencils_gpu = [
    'pystencils / GPU',
    'green',
    '^'
]
pystencils_cpu = [
    'pystencils / CPU',
    'darkblue',
    '<'
]
numba_cpu = [
    'Numba / CPU',
    'blue',
    '>'
]

TARGS = (
    nlcpy_sca,
    # pystencils_gpu,
    # cupy_fusion,
    # numba_cuda,
    pystencils_cpu,
    # numba_cpu,
)
PALETTES = tuple([_targ[1] for _targ in TARGS])
MARKERS = tuple([_targ[2] for _targ in TARGS])

SHAPES = {
    '2d': (
        (  64,   64),
        ( 128,  128),
        ( 256,  256),
        ( 512,  512),
        (1024, 1024),
    ),
    '3d': (
        (  64,   64,   64),
        ( 128,  128,  128),
        ( 256,  256,  256),
        ( 512,  512,  512),
        (1024, 1024, 1024),
    ),
}


def print_params():
    print('dtype:', DTYPES)
    print('iter:', ITER)
    print('stencil_scale:', STENCIL_SCALE)
    print('target:', TARGS)
    print('shape:', SHAPES)
