import importlib
import numpy
from bench_core import gen_data
from bench_core import run_benchmark
from pprint import pprint

modules = ['numpy', 'nlcpy']
shapes = [(1000, 1000), (10000, 10000)]

result = {
    'module': [],
    'operations': [],
    'array_size': [],
    'runtime': [],
    'speedup': [],
}


def set_dict(module, op, nbytes, runtime):
    result['module'].append(module)
    result['operations'].append(op)
    result['array_size'].append(str(int(nbytes/1e6))+'MB')
    result['runtime'].append(runtime)
    if module is 'numpy':
        result['speedup'].append(1)
    else:
        result['speedup'].append(result['runtime'][-2] / runtime)


def bench_sum():
    print("computing sum", end="", flush=True)
    for shape in shapes:
        for module in modules:
            m = importlib.import_module(module)
            data = gen_data(numpy, numpy.random.random, shape)
            data = m.asarray(data)
            res, rt = run_benchmark(m, m.sum, data)
            if module is 'numpy':
                res_np = res
            elif module is 'nlcpy':
                res_vp = res
            set_dict(module, 'sum', data.nbytes, rt)
            print(".", end="", flush=True)
        numpy.testing.assert_allclose(res_np, res_vp)
    print("done", flush=True)


def bench_std():
    print("computing std", end="", flush=True)
    for shape in shapes:
        for module in modules:
            m = importlib.import_module(module)
            data = gen_data(numpy, numpy.random.random, shape)
            data = m.asarray(data)
            res, rt = run_benchmark(m, m.std, data)
            if module is 'numpy':
                res_np = res
            elif module is 'nlcpy':
                res_vp = res
            set_dict(module, 'standard\ndeviation', data.nbytes, rt)
            print(".", end="", flush=True)
        numpy.testing.assert_allclose(res_np, res_vp)
    print("done", flush=True)


def bench_add():
    compute_func = lambda data: data + data

    print("computing add", end="", flush=True)
    for shape in shapes:
        for module in modules:
            m = importlib.import_module(module)
            data = gen_data(numpy, numpy.random.random, shape)
            data = m.asarray(data)
            res, rt = run_benchmark(m, compute_func, data)
            if module is 'numpy':
                res_np = res
            elif module is 'nlcpy':
                res_vp = res
            set_dict(module, 'element-wise\nadd', data.nbytes, rt)
            print(".", end="", flush=True)
        numpy.testing.assert_allclose(res_np, res_vp)
    print("done", flush=True)


def bench_matmul():
    compute_func = lambda data: data @ data

    print("computing matmul", end="", flush=True)
    for shape in shapes:
        for module in modules:
            m = importlib.import_module(module)
            data = gen_data(numpy, numpy.random.random, shape)
            data = m.asarray(data)
            res, rt = run_benchmark(m, compute_func, data)
            if module is 'numpy':
                res_np = res
            elif module is 'nlcpy':
                res_vp = res
            set_dict(module, 'matrix\nmultiplication', data.nbytes, rt)
            print(".", end="", flush=True)
        numpy.testing.assert_allclose(res_np, res_vp)
    print("done", flush=True)


def bench_copy():
    compute_func = lambda data: data.copy()

    print("computing copy", end="", flush=True)
    for shape in shapes:
        for module in modules:
            m = importlib.import_module(module)
            data = gen_data(numpy, numpy.random.random, shape)
            data = m.asarray(data)
            res, rt = run_benchmark(m, compute_func, data)
            if module is 'numpy':
                res_np = res
            elif module is 'nlcpy':
                res_vp = res
            set_dict(module, 'data copy', data.nbytes, rt)
            print(".", end="", flush=True)
        numpy.testing.assert_allclose(res_np, res_vp)
    print("done", flush=True)


def bench_fft():
    print("computing fft", end="", flush=True)
    for shape in shapes:
        for module in modules:
            m = importlib.import_module(module)
            data = gen_data(numpy, numpy.random.random, shape)
            data = m.asarray(data)
            res, rt = run_benchmark(m, m.fft.fft, data)
            if module is 'numpy':
                res_np = res
            elif module is 'nlcpy':
                res_vp = res
            set_dict(module, 'multiple\n1-D fft', data.nbytes, rt)
            print(".", end="", flush=True)
        numpy.testing.assert_allclose(res_np, res_vp)
    print("done", flush=True)


def bench_solve():
    print("computing solve", end="", flush=True)
    for shape in shapes:
        for module in modules:
            m = importlib.import_module(module)
            data1 = gen_data(numpy, numpy.random.random, shape)
            data2 = gen_data(numpy, numpy.random.random, shape)
            data1 = m.asarray(data1)
            data2 = m.asarray(data2)
            res, rt = run_benchmark(m, m.linalg.solve, (data1, data2))
            if module is 'numpy':
                res_np = res
            elif module is 'nlcpy':
                res_vp = res
            set_dict(module, 'solve\nlinear equation', data1.nbytes, rt)
            print(".", end="", flush=True)
        numpy.testing.assert_allclose(res_np, res_vp, atol=1e-6)
    print("done", flush=True)


def get_runtime():
    return result

def write_runtime(path='result/basic_op_result.pickle'):
    import pickle
    with open(path, mode='wb') as fo:
        pickle.dump(result, fo)
