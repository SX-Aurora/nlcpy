import argparse
import numpy as np
import nlcpy as vp
import params
# from compute_xya import stencil_xya
# from compute_xyza import stencil_xyza
import gen_graph


result = {
    'target': [],
    'size': [],
    'sec': [],
    'GFLOPS': [],
    'max': [],
    'min': [],
    'avg': [],
    'std': []
}


def initialize(shape, dtype, targ):
    scale = .001
    if targ == params.nlcpy_sca[0]:
        y = vp.sca.create_optimized_array(shape, dtype=dtype)
        _x = vp.arange(y.size, dtype=dtype).reshape(shape) * scale
        x = vp.sca.convert_optimized_array(_x)
    elif targ in (params.numba_cpu[0], params.pystencils_cpu[0],
                  params.pystencils_gpu[0]):
        y = np.zeros(shape, dtype=dtype)
        x = np.arange(y.size, dtype=dtype).reshape(shape) * scale
    elif targ in (params.cupy_fusion[0], params.numba_cuda[0]):
        import cupy as cp
        y = cp.zeros(shape, dtype=dtype)
        x = cp.arange(y.size, dtype=dtype).reshape(shape) * scale
    else:
        raise ValueError

    return x, y


def check(data):
    return data.max(), data.min(), data.mean(), data.std()


def sec2gflops(sec, stencil_size, compute_size, i):
    return stencil_size * compute_size * i * 1e-9 / sec


def stencil_xya(x, y, coef, N, I, targ):
    if targ == params.nlcpy_sca[0]:
        from compute_nlcpy_sca import nlcpy_2d_sca_impl
        rt, res = nlcpy_2d_sca_impl(x, y, coef, N, I)
    elif targ == params.numba_cpu[0]:
        from compute_numba_cpu import numba_2d_impl
        rt, res = numba_2d_impl(x, y, coef, N, I)
    elif targ == params.numba_cuda[0]:
        from compute_numba_cuda import numba_2d_cuda_impl
        _, _ = numba_2d_cuda_impl(x, y, coef, N, I=1)
        rt, res = numba_2d_cuda_impl(x, y, coef, N, I)
    elif targ == params.cupy_fusion[0]:
        from compute_cupy_fusion import cupy_2d_fusion_impl
        _, _ = cupy_2d_fusion_impl(x, y, coef, N, I=1)
        rt, res = cupy_2d_fusion_impl(x, y, coef, N, I)
    elif targ == params.pystencils_cpu[0]:
        from compute_pystencils import pystencils_2d_cpu_impl
        rt, res = pystencils_2d_cpu_impl(x, y, coef, N, I)
    elif targ == params.pystencils_gpu[0]:
        from compute_pystencils import pystencils_2d_gpu_impl
        rt, res = pystencils_2d_gpu_impl(x, y, coef, N, I)
    else:
        raise ValueError
    return rt, np.asarray(res)


def stencil_xyza(x, y, coef, N, I, targ):
    if targ == params.nlcpy_sca[0]:
        from compute_nlcpy_sca import nlcpy_3d_sca_impl
        rt, res = nlcpy_3d_sca_impl(x, y, coef, N, I)
    elif targ == params.numba_cpu[0]:
        from compute_numba_cpu import numba_3d_impl
        rt, res = numba_3d_impl(x, y, coef, N, I)
    elif targ == params.numba_cuda[0]:
        from compute_numba_cuda import numba_3d_cuda_impl
        _, _ = numba_3d_cuda_impl(x, y, coef, N, I=1)
        rt, res = numba_3d_cuda_impl(x, y, coef, N, I)
    elif targ == params.cupy_fusion[0]:
        from compute_cupy_fusion import cupy_3d_fusion_impl
        _, _ = cupy_3d_fusion_impl(x, y, coef, N, I=1)
        rt, res = cupy_3d_fusion_impl(x, y, coef, N, I)
    elif targ == params.pystencils_cpu[0]:
        from compute_pystencils import pystencils_3d_cpu_impl
        rt, res = pystencils_3d_cpu_impl(x, y, coef, N, I)
    elif targ == params.pystencils_gpu[0]:
        from compute_pystencils import pystencils_3d_gpu_impl
        rt, res = pystencils_3d_gpu_impl(x, y, coef, N, I)
    else:
        raise ValueError
    return rt, np.asarray(res)


def compute(dim, targ):
    print('targ:', targ)
    for _dtype in params.DTYPES:
        print('  dtype:', _dtype)
        for _shape in params.SHAPES[dim]:
            print('    shape:', _shape)
            x, y = initialize(_shape, _dtype, targ)
            if dim == '2d':
                rt, res = stencil_xya(
                    x,
                    y,
                    params.COEF,
                    params.STENCIL_SCALE,
                    params.ITER,
                    targ
                )
                gflops = sec2gflops(
                    rt,
                    params.STENCIL_SCALE * 4 * 2 + 1,
                    (
                        (res.shape[-1] - params.STENCIL_SCALE * 2) *
                        (res.shape[-2] - params.STENCIL_SCALE * 2)
                    ),
                    params.ITER
                )
            elif dim == '3d':
                rt, res = stencil_xyza(
                    x,
                    y,
                    params.COEF,
                    params.STENCIL_SCALE,
                    params.ITER,
                    targ
                )
                gflops = sec2gflops(
                    rt,
                    params.STENCIL_SCALE * 6 * 2 + 1,
                    (
                        (res.shape[-1] - params.STENCIL_SCALE * 2) *
                        (res.shape[-2] - params.STENCIL_SCALE * 2) *
                        (res.shape[-3] - params.STENCIL_SCALE * 2)
                    ),
                    params.ITER
                )
            else:
                raise ValueError

            print('      elapsed:', rt)
            max, min, avg, std = check(res)
            print('      max: {}, min: {}, avg: {}, std: {}'.format(
                max, min, avg, std)
            )
            print('      GFLOPS:', gflops)
            result['target'].append(targ)
            result['size'].append(str(_shape))
            result['sec'].append(str(rt))
            result['GFLOPS'].append(gflops)
            result['max'].append(max)
            result['min'].append(min)
            result['avg'].append(avg)
            result['std'].append(std)


def run_stencil_bench(dim):
    for _targ in params.TARGS:
        compute(dim, _targ[0])


def dict_clean():
    result['target'] = []
    result['size'] = []
    result['sec'] = []
    result['GFLOPS'] = []
    result['max'] = []
    result['min'] = []
    result['avg'] = []
    result['std'] = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--xya', action='store_true', default=False)
    parser.add_argument('--xyza', action='store_true', default=False)
    args = parser.parse_args()
    d = vars(args)
    print('params: ', str(d))

    if not args.xya and not args.xyza:
        raise ValueError('Please specify the argument `--xya` or `--xyza`.')

    params.print_params()

    # 2d bench
    if args.xya:
        print('\n**********************')
        print('*** Start 2d bench ***')
        print('**********************\n')
        run_stencil_bench('2d')
        gen_graph.gen(result, params.PALETTES, params.MARKERS, 'perf-xya')

    dict_clean()

    # 3d bench
    if args.xyza:
        print('\n**********************')
        print('*** Start 3d bench ***')
        print('**********************\n')
        run_stencil_bench('3d')
        gen_graph.gen(result, params.PALETTES, params.MARKERS, 'perf-xyza')
