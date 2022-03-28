import nlcpy
from nlcpy import ve_types

f_src = r"""
integer(kind=4) function ve_add(px, py, pz, n)
    integer(kind=4), value :: n
    double precision :: px(n), py(n), pz(n)
    !$omp parallel do
    do i=1, n
        pz(i) = px(i) + py(i)
    end do
    ve_add = 0
end
"""

ve_lib = nlcpy.jit.CustomVELibrary(code=f_src, compiler='nfort')

ve_add = ve_lib.get_function(
    've_add_',
    args_type=(ve_types.uint64, ve_types.uint64, ve_types.uint64, ve_types.int32),
    ret_type=ve_types.int32
)


x = nlcpy.arange(10., dtype='f8')
y = nlcpy.arange(10., dtype='f8')
z = nlcpy.empty(10, dtype='f8')
ret = ve_add(x.ve_adr, y.ve_adr, z.ve_adr, z.size, sync=True)

print(z)
print(ret)
