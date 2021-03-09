import importlib
import numpy
from bench_core import gen_data
from bench_core import run_benchmark
from pprint import pprint

modules = ['numpy', 'nlcpy']
shapes = [(1000, 1000), (10000, 10000)]
#shapes = [(1000, 1000),]

binary_ops = [
    'add',
    'subtract',
    'multiply',
    'divide',
    'logaddexp', 
    'logaddexp2',
    'true_divide',
    'floor_divide',
    'power',
    'remainder',
    'mod',
    'fmod',
    'heaviside',
    'arctan2',
    'hypot',
    'bitwise_and',
    'bitwise_or',
    'bitwise_xor',
    'left_shift',
    'right_shift',
    'greater',
    'greater_equal',
    'less',
    'less_equal',
    'not_equal',
    'equal',
    'logical_and',
    'logical_or',
    'logical_xor',
    'minimum',
    'maximum',
    'fmax',
    'fmin',
    'ldexp',
    'copysign', 
    'nextafter',
      ]   

unary_ops = [
    'negative',
    'positive',
    'absolute',
    'fabs',
    'sign',
    'conj',
    'conjugate',
    'exp',
    #'log',
    'sqrt',
    'square',
    'cbrt',
    'reciprocal',
    'sin',
    'cos',
    'tan',
    #'arcsin', 
    'arccos', 
    'arctan',
    'sinh',
    'cosh',
    'tanh',
    'arcsinh',
    #'arccosh',
    #'arctanh',
    'deg2rad',
    'rad2deg',
    'degrees',
    'radians',
    'invert',
    'logical_not',
    'isfinite',
    'isinf',
    'isnan',
    'signbit',
    'spacing',
    'floor',
    'ceil',
      ]   


not_float_ops = [
    'bitwise_and',
    'bitwise_or',
    'bitwise_xor',
    'left_shift',
    'right_shift',
    'ldexp',
    'invert',
]

result = {
    'module': [],
    'operation': [],
    'array_size': [],
    'runtime': [],
    'speedup': [],
}

def set_dict(module, op, nbytes, runtime):
    result['module'].append(module)
    result['operation'].append(op)
    result['array_size'].append(str(int(nbytes/1e6))+'MB')
    result['runtime'].append(runtime)
    if module is 'numpy':
        result['speedup'].append(1)
    else:
        result['speedup'].append(result['runtime'][-2] / runtime)



def bench_binary():
    for op in binary_ops:
        print("computing {}".format(op), end="", flush=True)
        for shape in shapes:
            for module in modules:
                m = importlib.import_module(module)
                compute_func = getattr(m, op)
                data = gen_data(numpy, numpy.random.random, shape)
                data = m.asarray(data)
                if op in not_float_ops:
                    data = data.astype(dtype=int)
                res, rt = run_benchmark(m, compute_func, data)
                if module is 'numpy':
                    res_np = res
                elif module is 'nlcpy':
                    res_vp = res
                set_dict(module, op, data.nbytes, rt)
                print(".", end="", flush=True)
            numpy.testing.assert_allclose(res_np, res_vp)
        print("done", flush=True)
    #pprint(result)


def bench_unary():
    for op in unary_ops:
        print("computing {}".format(op), end="", flush=True)
        for shape in shapes:
            for module in modules:
                m = importlib.import_module(module)
                compute_func = getattr(m, op)
                data = gen_data(numpy, numpy.random.random, shape)
                data = m.asarray(data)
                if op in not_float_ops:
                    data = data.astype(dtype=int)
                res, rt = run_benchmark(m, compute_func, data, is_binary=False)
                if module is 'numpy':
                    res_np = res
                elif module is 'nlcpy':
                    res_vp = res
                set_dict(module, op, data.nbytes, rt)
                print(".", end="", flush=True)
            numpy.testing.assert_allclose(res_np, res_vp)
        print("done", flush=True)
    #pprint(result)




def get_runtime():
    return result

def write_runtime(path='result/ufunc_result.pickle'):
    import pickle
    with open(path, mode='wb') as fo:
        pickle.dump(result, fo)

