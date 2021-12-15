import nlcpy as vp
import time


############
#   2D     #
############

def nlcpy_2d_sca_impl(x, y, coef, N, I=1):
    kerns = []
    dx, dy = vp.sca.create_descriptor((x, y))

    def create_sca_kernel(dx, dy):
        loc_x = [(0, i) for i in range(-N, N+1)]
        loc_y = [(i, 0) for i in range(-N, N+1)]
        d = vp.sca.empty_description()
        for loc in set(loc_x + loc_y):
            d += dx[..., loc[0], loc[1]]
        d *= coef
        return vp.sca.create_kernel(d, dy[...])

    kerns.append(create_sca_kernel(dx, dy))
    kerns.append(create_sca_kernel(dy, dx))

    vp.request.flush()
    s = time.time()
    for i in range(I):
        res = kerns[i % 2].execute()
    vp.request.flush()
    e = time.time()
    return e - s, res.get()


############
#   3D     #
############

def nlcpy_3d_sca_impl(x, y, coef, N, I=1):
    kerns = []
    dx, dy = vp.sca.create_descriptor((x, y))

    def create_sca_kernel(dx, dy):
        loc_x = [(0, 0, i) for i in range(-N, N+1)]
        loc_y = [(0, i, 0) for i in range(-N, N+1)]
        loc_z = [(i, 0, 0) for i in range(-N, N+1)]
        d = vp.sca.empty_description()
        for loc in set(loc_x + loc_y + loc_z):
            d += dx[..., loc[0], loc[1], loc[2]]
        d *= coef
        return vp.sca.create_kernel(d, dy[...])

    kerns.append(create_sca_kernel(dx, dy))
    kerns.append(create_sca_kernel(dy, dx))

    vp.request.flush()
    s = time.time()
    for i in range(I):
        res = kerns[i % 2].execute()
    vp.request.flush()
    e = time.time()
    return e - s, res
