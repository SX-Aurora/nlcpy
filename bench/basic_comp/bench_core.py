import importlib
import time


def run_flush(m, func, *args):
    res = func(*args)
    if m.__name__ == 'nlcpy':
        m.request.flush()
    elif m.__name__ == 'cupy':
        m.cuda.Stream().null.synchronize()
    return res

def gen_data(m, data_func, shape):
    m.random.seed(0)
    data = run_flush(m, data_func, shape)
    return data

def warmup(m, compute_func, data):
    if type(data) is tuple and len(data) > 1:
        run_flush(m, compute_func, *data)
    else:
        run_flush(m, compute_func, data)

def run_benchmark(m, compute_func, data, rounds=5):
    warmup(m, compute_func, data)
    s = time.time()
    for i in range(rounds):
        if type(data) is tuple and len(data) > 1:
            res = run_flush(m, compute_func, *data)
        else:
            res = run_flush(m, compute_func, data)
    e = time.time() - s
    return res, e

