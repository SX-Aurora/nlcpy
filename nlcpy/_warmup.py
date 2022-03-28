import nlcpy


def _warmup():
    xx = []
    MB = 1024 ** 2
    # allocate VE memory from sbrk (not mmap)
    for i in range(1000):
        xx.append(nlcpy.zeros(10 * MB // 8, dtype='f8'))
    nlcpy.request.flush()
    xx = []
