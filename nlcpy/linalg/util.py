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


# closure for callback
def _assertNotSingular(info):
    def _info_check(*args):
        if info > 0:
            raise LinAlgError('Singular matrix')
    return _info_check


# closure for callback
def _assertPositiveDefinite(info):
    def _info_check(*args):
        if info > 0:
            raise LinAlgError('Matrix is not positive definite')
    return _info_check
