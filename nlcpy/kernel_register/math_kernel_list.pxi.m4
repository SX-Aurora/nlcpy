include(macros.m4)dnl
define(<--@math_kernel_list@-->,<--@
    "nlcpy_$1": {
        "ret": "uint64_t",
        "args":
            [
                b"void *",
                b"void *",
ifelse($2,binary,<--@dnl
                b"void *",
@-->)dnl
                b"int32_t *",
            ],
    },
@-->)dnl
_math_kernel_list = {
    # math_functions
math_kernel_list(angle)dnl
}
