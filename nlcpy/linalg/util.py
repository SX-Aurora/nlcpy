from nlcpy.linalg import LinAlgError


def _assertRank2(*arrays):
    for a in arrays:
        if a.ndim != 2:
            raise LinAlgError('%d-dimensional array given. '
                              'Array must be two-dimensional' % a.ndim)


def _assertRankAtLeast2(*arrays):
    for a in arrays:
        if a.ndim < 2:
            raise LinAlgError('%d-dimensional array given. '
                              'Array must be at least two-dimensional' % a.ndim)


def _assertNdSquareness(*arrays):
    for a in arrays:
        if a.shape[-1] != a.shape[-2]:
            raise LinAlgError('Last 2 dimensions of the array must be square')


def _assertNotSingular(info):
    if info > 0:
        raise LinAlgError('Singular matrix')


def _assertPositiveDefinite(info):
    if info > 0:
        raise LinAlgError('Matrix is not positive definite')
