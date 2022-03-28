import importlib
import numpy
from bench_core import gen_data
from bench_core import run_benchmark
from pprint import pprint

modules = ['numpy', 'nlcpy', 'cupy']
shapes = [(10000, 10000)]

result = {}
for m in modules:
    result[m] = {
        'operations': [],
        'array_size': [],
        'runtime': [],
    }

def set_dict(module, op, nbytes, runtime):
    attr = result[module]
    attr['operations'].append(op)
    attr['array_size'].append(str(int(nbytes/1e6))+'MB')
    attr['runtime'].append(runtime)


def bench_sum():
    numpy.random.seed(0)
    print("computing sum", end="", flush=True)
    for shape in shapes:
        for module in modules:
            m = importlib.import_module(module)
            data = gen_data(numpy, numpy.random.random, shape)
            data = m.asarray(data)
            res, rt = run_benchmark(m, m.sum, data)
            if module == 'numpy':
                res_np = res
            elif module == 'nlcpy':
                res_vp = res
            elif module == 'cupy':
                res_cp = res
            set_dict(module, 'sum', data.nbytes, rt)
            print(".", end="", flush=True)
        if 'numpy' in modules and 'nlcpy' in modules:
            numpy.testing.assert_allclose(res_np, res_vp)
        elif 'nlcpy' in modules and 'cupy' in modules:
            numpy.testing.assert_allclose(res_vp.get(), res_cp.get())
        elif 'numpy' in modules and 'cupy' in modules:
            numpy.testing.assert_allclose(res_np, res_cp.get())
    print("done", flush=True)


def bench_std():
    numpy.random.seed(0)
    print("computing std", end="", flush=True)
    for shape in shapes:
        for module in modules:
            m = importlib.import_module(module)
            data = gen_data(numpy, numpy.random.random, shape)
            data = m.asarray(data)
            res, rt = run_benchmark(m, m.std, data)
            if module == 'numpy':
                res_np = res
            elif module == 'nlcpy':
                res_vp = res
            elif module == 'cupy':
                res_cp = res
            set_dict(module, 'standard\ndeviation', data.nbytes, rt)
            print(".", end="", flush=True)
        if 'numpy' in modules and 'nlcpy' in modules:
            numpy.testing.assert_allclose(res_np, res_vp)
        elif 'nlcpy' in modules and 'cupy' in modules:
            numpy.testing.assert_allclose(res_vp.get(), res_cp.get())
        elif 'numpy' in modules and 'cupy' in modules:
            numpy.testing.assert_allclose(res_np, res_cp.get())
    print("done", flush=True)


def bench_add():
    compute_func = lambda data: data + data
    numpy.random.seed(0)

    print("computing add", end="", flush=True)
    for shape in shapes:
        for module in modules:
            m = importlib.import_module(module)
            data = gen_data(numpy, numpy.random.random, shape)
            data = m.asarray(data)
            res, rt = run_benchmark(m, compute_func, data)
            if module == 'numpy':
                res_np = res
            elif module == 'nlcpy':
                res_vp = res
            elif module == 'cupy':
                res_cp = res
            set_dict(module, 'element-wise\nadd', data.nbytes, rt)
            print(".", end="", flush=True)
        if 'numpy' in modules and 'nlcpy' in modules:
            numpy.testing.assert_allclose(res_np, res_vp)
        elif 'nlcpy' in modules and 'cupy' in modules:
            numpy.testing.assert_allclose(res_vp.get(), res_cp.get())
        elif 'numpy' in modules and 'cupy' in modules:
            numpy.testing.assert_allclose(res_np, res_cp.get())
    print("done", flush=True)


def bench_matmul():
    compute_func = lambda data: data @ data
    numpy.random.seed(0)

    print("computing matmul", end="", flush=True)
    for shape in shapes:
        for module in modules:
            m = importlib.import_module(module)
            data = gen_data(numpy, numpy.random.random, shape)
            data = m.asarray(data)
            res, rt = run_benchmark(m, compute_func, data)
            if module == 'numpy':
                res_np = res
            elif module == 'nlcpy':
                res_vp = res
            elif module == 'cupy':
                res_cp = res
            set_dict(module, 'matrix\nmultiplication', data.nbytes, rt)
            print(".", end="", flush=True)
        if 'numpy' in modules and 'nlcpy' in modules:
            numpy.testing.assert_allclose(res_np, res_vp)
        elif 'nlcpy' in modules and 'cupy' in modules:
            numpy.testing.assert_allclose(res_vp.get(), res_cp.get())
        elif 'numpy' in modules and 'cupy' in modules:
            numpy.testing.assert_allclose(res_np, res_cp.get())
    print("done", flush=True)


def bench_copy():
    compute_func = lambda data: data.copy()
    numpy.random.seed(0)

    print("computing copy", end="", flush=True)
    for shape in shapes:
        for module in modules:
            m = importlib.import_module(module)
            data = gen_data(numpy, numpy.random.random, shape)
            data = m.asarray(data)
            res, rt = run_benchmark(m, compute_func, data)
            if module == 'numpy':
                res_np = res
            elif module == 'nlcpy':
                res_vp = res
            elif module == 'cupy':
                res_cp = res
            set_dict(module, 'data copy', data.nbytes, rt)
            print(".", end="", flush=True)
        if 'numpy' in modules and 'nlcpy' in modules:
            numpy.testing.assert_allclose(res_np, res_vp)
        elif 'nlcpy' in modules and 'cupy' in modules:
            numpy.testing.assert_allclose(res_vp.get(), res_cp.get())
        elif 'numpy' in modules and 'cupy' in modules:
            numpy.testing.assert_allclose(res_np, res_cp.get())
    print("done", flush=True)


def bench_fft():
    numpy.random.seed(0)
    print("computing fft", end="", flush=True)
    for shape in shapes:
        for module in modules:
            m = importlib.import_module(module)
            data = gen_data(numpy, numpy.random.random, shape)
            data = m.asarray(data)
            res, rt = run_benchmark(m, m.fft.fft, data)
            if module == 'numpy':
                res_np = res
            elif module == 'nlcpy':
                res_vp = res
            elif module == 'cupy':
                res_cp = res
            set_dict(module, 'multiple\n1-D fft', data.nbytes, rt)
            print(".", end="", flush=True)
        if 'numpy' in modules and 'nlcpy' in modules:
            numpy.testing.assert_allclose(res_np, res_vp)
        elif 'nlcpy' in modules and 'cupy' in modules:
            numpy.testing.assert_allclose(res_vp.get(), res_cp.get())
        elif 'numpy' in modules and 'cupy' in modules:
            numpy.testing.assert_allclose(res_np, res_cp.get())
    print("done", flush=True)


def bench_solve():
    numpy.random.seed(0)
    print("computing solve", end="", flush=True)
    for shape in shapes:
        for module in modules:
            m = importlib.import_module(module)
            data1 = gen_data(numpy, numpy.random.random, shape)
            data2 = gen_data(numpy, numpy.random.random, shape)
            data1 = m.asarray(data1)
            data2 = m.asarray(data2)
            res, rt = run_benchmark(m, m.linalg.solve, (data1, data2))
            if module == 'numpy':
                res_np = res
            elif module == 'nlcpy':
                res_vp = res
            elif module == 'cupy':
                res_cp = res
            set_dict(module, 'solve\nlinear equation', data1.nbytes, rt)
            print(".", end="", flush=True)
        if 'numpy' in modules and 'nlcpy' in modules:
            numpy.testing.assert_allclose(res_np, res_vp, atol=1e-10)
        elif 'nlcpy' in modules and 'cupy' in modules:
            numpy.testing.assert_allclose(res_vp.get(), res_cp.get(), atol=1e-10)
        elif 'numpy' in modules and 'cupy' in modules:
            numpy.testing.assert_allclose(res_np, res_cp.get(), atol=1e-10)
    print("done", flush=True)


def get_runtime():
    return result

def write_runtime(path='result/{}_result.pickle'):
    import pickle, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for m in modules:
        filepath = path.format(m)
        with open(filepath, mode='wb') as fo:
            pickle.dump(result[m], fo)
