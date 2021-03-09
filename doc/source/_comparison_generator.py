import importlib

exclude_list = (
    'abs',
    'add_docstring',
    'add_newdoc',
    'add_newdoc_ufunc',
    'alen',
    'alltrue',
    'bitwise_not',
    'compare_chararrays',
    'cumproduct',
    'fastCopyAndTranspose',
    'get_array_wrap',
    'int_asbuffer',
    'iterable',
    'loads',
    'mafromtxt',
    'max',
    'min',
    'ndfromtxt',
    'ndim',
    'product',
    'rank',
    'recfromcsv',
    'recfromtxt',
    'round',
    'row_stack',
    'safe_eval',
    'set_numeric_ops',
    'size',
    'sometrue',
    'register_func',
    'restore_all',
    'restore_func',
    'fv',
    'ipmt',
    'irr',
    'mirr',
    'nper',
    'npv',
    'pmt',
    'ppmt',
    'pv',
    'rate',
)


def _get_functions(obj):
    return set([
        n for n in dir(obj)
        if (n not in ['test']  # not in blacklist
            and callable(getattr(obj, n))  # callable
            and not isinstance(getattr(obj, n), type)  # not class
            and n[0].islower()  # starts with lower char
            and not n.startswith('__')  # not special methods
            and n not in exclude_list  # not inactive methods
            )
    ])


def _import(mod, klass):
    obj = importlib.import_module(mod)
    if klass:
        obj = getattr(obj, klass)
        return obj, ':meth:`{}.{}.{{}}`'.format(mod, klass)
    else:
        # ufunc is not a function
        return obj, ':obj:`{}.{{}}`'.format(mod)


def _generate_comparison_rst(
        base_mod, nlcpy_mod, base_type, klass, exclude_mod):
    base_obj, base_fmt = _import(base_mod, klass)
    vp_obj, vp_fmt = _import(nlcpy_mod, klass)
    base_funcs = _get_functions(base_obj)
    vp_funcs = _get_functions(vp_obj)

    if exclude_mod:
        exclude_obj, _ = _import(exclude_mod, klass)
        exclude_funcs = _get_functions(exclude_obj)
        base_funcs -= exclude_funcs
        vp_funcs -= exclude_funcs

    buf = []
    buf += [
        '.. csv-table::',
        '   :header: {}, NLCPy'.format(base_type),
        '',
    ]
    for f in sorted(base_funcs):
        base_cell = base_fmt.format(f)
        vp_cell = r'\-'
        if f in vp_funcs:
            vp_cell = vp_fmt.format(f)
            if getattr(base_obj, f) is getattr(vp_obj, f):
                vp_cell = '{} (*alias of* {})'.format(vp_cell, base_cell)
        line = '   {}, {}'.format(base_cell, vp_cell)
        buf.append(line)

    buf += [
        '',
        '.. Summary:',
        '   Number of NumPy functions: {}'.format(len(base_funcs)),
        '   Number of functions covered by NLCPy: {}'.format(
            len(vp_funcs & base_funcs)),
        '   NLCPy specific functions:',
    ] + [
        '   - {}'.format(f) for f in (vp_funcs - base_funcs)
    ]
    return buf


def _section(
        header, base_mod, nlcpy_mod,
        base_type='NumPy', klass=None, exclude=None):
    return [
        header,
        '^' * len(header),
        '',
    ] + _generate_comparison_rst(
        base_mod, nlcpy_mod, base_type, klass, exclude
    ) + [
        '',
    ]


def generate():
    buf = []

    buf += [
        'NumPy / NLCPy APIs',
        '------------------',
        '',
    ]
    buf += _section(
        'Module-Level',
        'numpy', 'nlcpy')
    buf += _section(
        'Multi-Dimensional Array',
        'numpy', 'nlcpy', klass='ndarray')
    buf += _section(
        'Linear Algebra',
        'numpy.linalg', 'nlcpy.linalg')
    buf += _section(
        'Discrete Fourier Transform',
        'numpy.fft', 'nlcpy.fft')
    buf += _section(
        'Random Sampling',
        'numpy.random', 'nlcpy.random', klass='Generator')

    return '\n'.join(buf)
