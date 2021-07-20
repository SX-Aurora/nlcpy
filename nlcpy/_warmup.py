import nlcpy


def _warmup():
    xx = []
    # allocate VE memory as heap
    for i in range(130):
        xx.append(nlcpy.zeros(int((1 * 1e8) / 8), dtype='f8'))
    nlcpy.request.flush()
    xx = []
