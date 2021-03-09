import importlib
import numpy
import scipy
from bench_core import gen_data
from bench_core import run_benchmark
from gen_graph import dw_graph
from pprint import pprint

# 10,000-10,000,000
#
# 10000
# 100000
# 1000000
# 10000000

shapes = []
modules = ['numpy', 'scipy', 'nlcpy']
# 1-D
shapes.append([
    (10000,),
    (100000,),
    (1000000,),
    (10000000,),
])
# 2-D
shapes.append([
    (100,  100),
    (100,  1000,),
    (1000, 1000,),
    (1000, 10000,),
])
# 3-D
shapes.append([
    (10,  10,  100),
    (10,  100, 100,),
    (100, 100, 100),
    (100, 100, 1000,),
])

result = {
    'module': [],
    'operation': [],
    'runtime': [],
    'speedup': [],
    'size': [],
}


def set_dict(module, op, runtime, size):
    result['module'].append(module)
    result['operation'].append(op)
    result['runtime'].append(runtime)
    if module is 'numpy':
        result['speedup'].append(1)
    elif module is 'scipy':
        result['speedup'].append(result['runtime'][-2] / runtime)
    elif module is 'nlcpy':
        result['speedup'].append(result['runtime'][-3] / runtime)
    result['size'].append(size)

def cleanup_dict():
    result['module'] = []
    result['operation'] = []
    result['runtime'] = []
    result['speedup'] = []
    result['size'] = []

def bench_fft():
    name = 'fft'
    print("computing {}".format(name), end="", flush=True)
    fix_num = 1
    for dim in shapes :
        for shape in dim :
            for module in modules:
                m = importlib.import_module(module)
                data = gen_data(numpy, numpy.random.random, shape)
                data = m.asarray(data)
                if module is 'scipy':
                    res, rt = run_benchmark(m, m.fft.fft, data)
                else:
                    res, rt = run_benchmark(m, m.fft.fft, data)

                if module is 'numpy':
                    res_np = res
                elif module is 'nlcpy':
                    res_vp = res
                elif module is 'scipy':
                    res_sp = res
                set_dict(module, name, rt, shape)
                print(".", end="", flush=True)
            numpy.testing.assert_allclose(res_np, res_vp)
            numpy.testing.assert_allclose(res_np, res_sp)
        write_runtime(name, str(fix_num))
        fix_num += 1
        cleanup_dict()
    print("done", flush=True)

def bench_ifft():
    name = 'ifft'
    print("computing {}".format(name), end="", flush=True)
    fix_num = 1
    for dim in shapes :
        for shape in dim :
            for module in modules:
                m = importlib.import_module(module)
                data = gen_data(numpy, numpy.random.random, shape)
                data = m.asarray(data)
                if module is 'scipy':
                    res, rt = run_benchmark(m, m.fft.ifft, data)
                else:
                    res, rt = run_benchmark(m, m.fft.ifft, data)

                if module is 'numpy':
                    res_np = res
                elif module is 'nlcpy':
                    res_vp = res
                elif module is 'scipy':
                    res_sp = res
                set_dict(module, name, rt, shape)
                print(".", end="", flush=True)
            numpy.testing.assert_allclose(res_np, res_vp)
            numpy.testing.assert_allclose(res_np, res_sp)
        write_runtime(name, str(fix_num))
        fix_num += 1
        cleanup_dict()
    print("done", flush=True)

def bench_fft2():
    name = 'fft2'
    print("computing {}".format(name), end="", flush=True)
    fix_num = 2 
    for dim in shapes[1:] :
        for shape in dim :
            for module in modules:
                m = importlib.import_module(module)
                data = gen_data(numpy, numpy.random.random, shape)
                data = m.asarray(data)
                if module is 'scipy':
                    res, rt = run_benchmark(m, m.fft.fft2, data)
                else:
                    res, rt = run_benchmark(m, m.fft.fft2, data)

                if module is 'numpy':
                    res_np = res
                elif module is 'nlcpy':
                    res_vp = res
                elif module is 'scipy':
                    res_sp = res
                set_dict(module, name, rt, shape)
                print(".", end="", flush=True)
            numpy.testing.assert_allclose(res_np, res_vp)
            numpy.testing.assert_allclose(res_np, res_sp)
        write_runtime(name, str(fix_num))
        fix_num += 1
        cleanup_dict()
    print("done", flush=True)

def bench_ifft2():
    name = 'ifft2'
    print("computing {}".format(name), end="", flush=True)
    fix_num = 2
    for dim in shapes[1:] :
        for shape in dim :
            for module in modules:
                m = importlib.import_module(module)
                data = gen_data(numpy, numpy.random.random, shape)
                data = m.asarray(data)
                if module is 'scipy':
                    res, rt = run_benchmark(m, m.fft.ifft2, data)
                else:
                    res, rt = run_benchmark(m, m.fft.ifft2, data)

                if module is 'numpy':
                    res_np = res
                elif module is 'nlcpy':
                    res_vp = res
                elif module is 'scipy':
                    res_sp = res
                set_dict(module, name, rt, shape)
                print(".", end="", flush=True)
            numpy.testing.assert_allclose(res_np, res_vp)
            numpy.testing.assert_allclose(res_np, res_sp)
        write_runtime(name, str(fix_num))
        fix_num += 1
        cleanup_dict()
    print("done", flush=True)

def bench_fftn():
    name = 'fftn'
    print("computing {}".format(name), end="", flush=True)
    fix_num = 1
    for dim in shapes :
        for shape in dim :
            for module in modules:
                m = importlib.import_module(module)
                data = gen_data(numpy, numpy.random.random, shape)
                data = m.asarray(data)
                if module is 'scipy':
                    res, rt = run_benchmark(m, m.fft.fftn, data)
                else:
                    res, rt = run_benchmark(m, m.fft.fftn, data)

                if module is 'numpy':
                    res_np = res
                elif module is 'nlcpy':
                    res_vp = res
                elif module is 'scipy':
                    res_sp = res
                set_dict(module, name, rt, shape)
                print(".", end="", flush=True)
            numpy.testing.assert_allclose(res_np, res_vp)
            numpy.testing.assert_allclose(res_np, res_sp)
        write_runtime(name, str(fix_num))
        fix_num += 1
        cleanup_dict()
    print("done", flush=True)

def bench_ifftn():
    name = 'ifftn'
    print("computing {}".format(name), end="", flush=True)
    fix_num = 1
    for dim in shapes :
        for shape in dim :
            for module in modules:
                m = importlib.import_module(module)
                data = gen_data(numpy, numpy.random.random, shape)
                data = m.asarray(data)
                if module is 'scipy':
                    res, rt = run_benchmark(m, m.fft.ifftn, data)
                else:
                    res, rt = run_benchmark(m, m.fft.ifftn, data)

                if module is 'numpy':
                    res_np = res
                elif module is 'nlcpy':
                    res_vp = res
                elif module is 'scipy':
                    res_sp = res
                set_dict(module, name, rt, shape)
                print(".", end="", flush=True)
            numpy.testing.assert_allclose(res_np, res_vp)
            numpy.testing.assert_allclose(res_np, res_sp)
        write_runtime(name, str(fix_num))
        fix_num += 1
        cleanup_dict()
    print("done", flush=True)

def bench_rfft():
    name = 'rfft'
    print("computing {}".format(name), end="", flush=True)
    fix_num = 1
    for dim in shapes :
        for shape in dim :
            for module in modules:
                m = importlib.import_module(module)
                data = gen_data(numpy, numpy.random.random, shape)
                data = m.asarray(data)
                if module is 'scipy':
                    res, rt = run_benchmark(m, m.fft.rfft, data)
                else:
                    res, rt = run_benchmark(m, m.fft.rfft, data)

                if module is 'numpy':
                    res_np = res
                elif module is 'nlcpy':
                    res_vp = res
                elif module is 'scipy':
                    res_sp = res
                set_dict(module, name, rt, shape)
                print(".", end="", flush=True)
            numpy.testing.assert_allclose(res_np, res_vp)
            numpy.testing.assert_allclose(res_np, res_sp)
        write_runtime(name, str(fix_num))
        fix_num += 1
        cleanup_dict()
    print("done", flush=True)

def bench_irfft():
    name = 'irfft'
    print("computing {}".format(name), end="", flush=True)
    fix_num = 1
    for dim in shapes :
        for shape in dim :
            for module in modules:
                m = importlib.import_module(module)
                data = gen_data(numpy, numpy.random.random, shape)
                data = m.asarray(data)
                if module is 'scipy':
                    res, rt = run_benchmark(m, m.fft.irfft, data)
                else:
                    res, rt = run_benchmark(m, m.fft.irfft, data)

                if module is 'numpy':
                    res_np = res
                elif module is 'nlcpy':
                    res_vp = res
                elif module is 'scipy':
                    res_sp = res
                set_dict(module, name, rt, shape)
                print(".", end="", flush=True)
            numpy.testing.assert_allclose(res_np, res_vp, rtol=1e-1)
            numpy.testing.assert_allclose(res_np, res_sp)
        write_runtime(name, str(fix_num))
        fix_num += 1
        cleanup_dict()
    print("done", flush=True)

def bench_rfft2():
    name = 'rfft2'
    print("computing {}".format(name), end="", flush=True)
    fix_num = 2
    for dim in shapes[1:] :
        for shape in dim :
            for module in modules:
                m = importlib.import_module(module)
                data = gen_data(numpy, numpy.random.random, shape)
                data = m.asarray(data)
                if module is 'scipy':
                    res, rt = run_benchmark(m, m.fft.rfft2, data)
                else:
                    res, rt = run_benchmark(m, m.fft.rfft2, data)

                if module is 'numpy':
                    res_np = res
                elif module is 'nlcpy':
                    res_vp = res
                elif module is 'scipy':
                    res_sp = res
                set_dict(module, name, rt, shape)
                print(".", end="", flush=True)
            numpy.testing.assert_allclose(res_np, res_vp)
            numpy.testing.assert_allclose(res_np, res_sp)
        write_runtime(name, str(fix_num))
        fix_num += 1
        cleanup_dict()
    print("done", flush=True)

def bench_irfft2():
    name = 'irfft2'
    print("computing {}".format(name), end="", flush=True)
    fix_num = 2
    for dim in shapes[1:] :
        for shape in dim :
            for module in modules:
                m = importlib.import_module(module)
                data = gen_data(numpy, numpy.random.random, shape)
                data = m.asarray(data)
                if module is 'scipy':
                    res, rt = run_benchmark(m, m.fft.irfft2, data)
                else:
                    res, rt = run_benchmark(m, m.fft.irfft2, data)

                if module is 'numpy':
                    res_np = res
                elif module is 'nlcpy':
                    res_vp = res
                elif module is 'scipy':
                    res_sp = res
                set_dict(module, name, rt, shape)
                print(".", end="", flush=True)
            numpy.testing.assert_allclose(res_np, res_vp)
            numpy.testing.assert_allclose(res_np, res_sp)
        write_runtime(name, str(fix_num))
        fix_num += 1
        cleanup_dict()
    print("done", flush=True)

def bench_rfftn():
    name = 'rfftn'
    print("computing {}".format(name), end="", flush=True)
    fix_num = 1
    for dim in shapes :
        for shape in dim :
            for module in modules:
                m = importlib.import_module(module)
                data = gen_data(numpy, numpy.random.random, shape)
                data = m.asarray(data)
                if module is 'scipy':
                    res, rt = run_benchmark(m, m.fft.rfftn, data)
                else:
                    res, rt = run_benchmark(m, m.fft.rfftn, data)

                if module is 'numpy':
                    res_np = res
                elif module is 'nlcpy':
                    res_vp = res
                elif module is 'scipy':
                    res_sp = res
                set_dict(module, name, rt, shape)
                print(".", end="", flush=True)
            numpy.testing.assert_allclose(res_np, res_vp)
            numpy.testing.assert_allclose(res_np, res_sp)
        write_runtime(name, str(fix_num))
        fix_num += 1
        cleanup_dict()
    print("done", flush=True)

def bench_irfftn():
    name = 'irfftn'
    print("computing {}".format(name), end="", flush=True)
    fix_num = 1
    for dim in shapes :
        for shape in dim :
            for module in modules:
                m = importlib.import_module(module)
                data = gen_data(numpy, numpy.random.random, shape)
                data = m.asarray(data)
                if module is 'scipy':
                    res, rt = run_benchmark(m, m.fft.irfftn, data)
                else:
                    res, rt = run_benchmark(m, m.fft.irfftn, data)

                if module is 'numpy':
                    res_np = res
                elif module is 'nlcpy':
                    res_vp = res
                elif module is 'scipy':
                    res_sp = res
                set_dict(module, name, rt, shape)
                print(".", end="", flush=True)
            numpy.testing.assert_allclose(res_np, res_vp, atol=1e-2)
            numpy.testing.assert_allclose(res_np, res_sp, atol=1e-2)
        write_runtime(name, str(fix_num))
        fix_num += 1
        cleanup_dict()
    print("done", flush=True)

def bench_hfft():
    name = 'hfft'
    print("computing {}".format(name), end="", flush=True)
    fix_num = 1
    for dim in shapes :
        for shape in dim :
            for module in modules:
                m = importlib.import_module(module)
                data = gen_data(numpy, numpy.random.random, shape)
                data = m.asarray(data)
                if module is 'scipy':
                    res, rt = run_benchmark(m, m.fft.hfft, data)
                else:
                    res, rt = run_benchmark(m, m.fft.hfft, data)

                if module is 'numpy':
                    res_np = res
                elif module is 'nlcpy':
                    res_vp = res
                elif module is 'scipy':
                    res_sp = res
                set_dict(module, name, rt, shape)
                print(".", end="", flush=True)
            numpy.testing.assert_allclose(res_np, res_vp, atol=1e-1)
            numpy.testing.assert_allclose(res_np, res_sp)
        write_runtime(name, str(fix_num))
        fix_num += 1
        cleanup_dict()
    print("done", flush=True)

def bench_ihfft():
    name = 'ihfft'
    print("computing {}".format(name), end="", flush=True)
    fix_num = 1
    for dim in shapes :
        for shape in dim :
            for module in modules:
                m = importlib.import_module(module)
                data = gen_data(numpy, numpy.random.random, shape)
                data = m.asarray(data)
                if module is 'scipy':
                    res, rt = run_benchmark(m, m.fft.ihfft, data)
                else:
                    res, rt = run_benchmark(m, m.fft.ihfft, data)

                if module is 'numpy':
                    res_np = res
                elif module is 'nlcpy':
                    res_vp = res
                elif module is 'scipy':
                    res_sp = res
                set_dict(module, name, rt, shape)
                print(".", end="", flush=True)
            numpy.testing.assert_allclose(res_np, res_vp, atol=1e-1)
            numpy.testing.assert_allclose(res_np, res_sp)
        write_runtime(name, str(fix_num))
        fix_num += 1
        cleanup_dict()
    print("done", flush=True)

def get_runtime():
    return result

def write_runtime(name, num):
    path= 'result/{}-{}D.pickle'.format(name, num)
    import pickle
    with open(path, mode='wb') as fo:
        pickle.dump(result, fo)
