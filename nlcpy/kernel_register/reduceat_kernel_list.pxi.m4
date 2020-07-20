include(macros.m4)dnl
define(<--@ufunc_kernel_list@-->,<--@
    "nlcpy_$1_reduceat": {
        "ret": "uint64_t",
        "args":
            [
                b"void *",
                b"void *",
                b"void *",
                b"void *",
                "int32_t",
                "int32_t",
                "int32_t",
            ],
    },
@-->)dnl
_reduceat_kernel_list = {

    # math_operations
ufunc_kernel_list(add)dnl
ufunc_kernel_list(subtract)dnl
ufunc_kernel_list(multiply)dnl
ufunc_kernel_list(divide)dnl
ufunc_kernel_list(logaddexp)dnl
ufunc_kernel_list(logaddexp2)dnl
ufunc_kernel_list(true_divide)dnl
ufunc_kernel_list(floor_divide)dnl
ufunc_kernel_list(power)dnl
ufunc_kernel_list(remainder)dnl
ufunc_kernel_list(mod)dnl
ufunc_kernel_list(fmod)dnl
ufunc_kernel_list(heaviside)dnl
    # bit-twiddling functions
ufunc_kernel_list(bitwise_and)dnl
ufunc_kernel_list(bitwise_or)dnl
ufunc_kernel_list(bitwise_xor)dnl
ufunc_kernel_list(left_shift)dnl
ufunc_kernel_list(right_shift)dnl
    # comparison functions
ufunc_kernel_list(greater)dnl
ufunc_kernel_list(greater_equal)dnl
ufunc_kernel_list(less)dnl
ufunc_kernel_list(less_equal)dnl
ufunc_kernel_list(not_equal)dnl
ufunc_kernel_list(equal)dnl
ufunc_kernel_list(logical_and)dnl
ufunc_kernel_list(logical_or)dnl
ufunc_kernel_list(logical_xor)dnl
ufunc_kernel_list(maximum)dnl
ufunc_kernel_list(minimum)dnl
ufunc_kernel_list(fmax)dnl
ufunc_kernel_list(fmin)dnl
    # trigonometric functions
ufunc_kernel_list(arctan2)dnl
ufunc_kernel_list(hypot)dnl
    # floating functions
ufunc_kernel_list(copysign)dnl
ufunc_kernel_list(nextafter)dnl

}
