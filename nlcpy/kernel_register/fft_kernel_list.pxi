_fft_kernel_list = {
    "nlcpy_fft_destroy_handle": {
        "ret": "uint64_t",
        "args": ["void"],
    },
    "nlcpy_fft_1d_c128_c128": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", "int64_t", "int64_t", "int",
                 b"int32_t *"],
    },
    "nlcpy_ifft_1d_c128_c128": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", "int64_t", "int64_t", "int",
                 b"int32_t *"],
    },
    "nlcpy_fft_1d_c64_c64": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", "int64_t", "int64_t", "int",
                 b"int32_t *"],
    },
    "nlcpy_ifft_1d_c64_c64": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", "int64_t", "int64_t", "int",
                 b"int32_t *"],
    },

    "nlcpy_rfft_1d_f64_c128": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", "int64_t", "int64_t", "int",
                 b"int32_t *"],
    },
    "nlcpy_rfft_1d_f32_c64": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", "int64_t", "int64_t", "int",
                 b"int32_t *"],
    },
    "nlcpy_irfft_1d_c128_f64": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", "int64_t", "int64_t", "int",
                 b"int32_t *"],
    },
    "nlcpy_irfft_1d_c64_f32": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", "int64_t", "int64_t", "int",
                 b"int32_t *"],
    },

    "nlcpy_fft_2d_c128_c128": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", b"ve_array *", b"ve_array *", "int",
                 b"int32_t *"],
    },
    "nlcpy_ifft_2d_c128_c128": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", b"ve_array *", b"ve_array *", "int",
                 b"int32_t *"],
    },
    "nlcpy_fft_2d_c64_c64": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", b"ve_array *", b"ve_array *", "int",
                 b"int32_t *"],
    },
    "nlcpy_ifft_2d_c64_c64": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", b"ve_array *", b"ve_array *", "int",
                 b"int32_t *"],
    },

    "nlcpy_rfft_2d_f64_c128": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", "int64_t", "int64_t", "int",
                 b"int32_t *"],
    },
    "nlcpy_rfft_2d_f32_c64": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", "int64_t", "int64_t", "int",
                 b"int32_t *"],
    },
    "nlcpy_irfft_2d_c128_f64": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", "int64_t", "int64_t", "int",
                 b"int32_t *"],
    },
    "nlcpy_irfft_2d_c64_f32": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", "int64_t", "int64_t", "int",
                 b"int32_t *"],
    },

    "nlcpy_fft_3d_c128_c128": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", b"ve_array *", b"ve_array *", "int",
                 b"int32_t *"],
    },
    "nlcpy_ifft_3d_c128_c128": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", b"ve_array *", b"ve_array *", "int",
                 b"int32_t *"],
    },
    "nlcpy_fft_3d_c64_c64": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", b"ve_array *", b"ve_array *", "int",
                 b"int32_t *"],
    },
    "nlcpy_ifft_3d_c64_c64": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", b"ve_array *", b"ve_array *", "int",
                 b"int32_t *"],
    },

    "nlcpy_rfft_3d_f64_c128": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", "int64_t", "int64_t", "int",
                 b"int32_t *"],
    },
    "nlcpy_rfft_3d_f32_c64": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", "int64_t", "int64_t", "int",
                 b"int32_t *"],
    },
    "nlcpy_irfft_3d_c128_f64": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", "int64_t", "int64_t", "int",
                 b"int32_t *"],
    },
    "nlcpy_irfft_3d_c64_f32": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", "int64_t", "int64_t", "int",
                 b"int32_t *"],
    },

    "nlcpy_fft_nd_c128_c128": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", "int64_t", "int64_t", "int",
                 b"int32_t *"],
    },
    "nlcpy_ifft_nd_c128_c128": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", "int64_t", "int64_t", "int",
                 b"int32_t *"],
    },
    "nlcpy_fft_nd_c64_c64": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", "int64_t", "int64_t", "int",
                 b"int32_t *"],
    },
    "nlcpy_ifft_nd_c64_c64": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", "int64_t", "int64_t", "int",
                 b"int32_t *"],
    },

    "nlcpy_rfft_nd_f64_c128": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", "int64_t", "int64_t", "int",
                 b"int32_t *"],
    },
    "nlcpy_rfft_nd_f32_c64": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", "int64_t", "int64_t", "int",
                 b"int32_t *"],
    },
    "nlcpy_irfft_nd_c128_f64": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", "int64_t", "int64_t", "int",
                 b"int32_t *"],
    },
    "nlcpy_irfft_nd_c64_f32": {
        "ret": "uint64_t",
        "args": [b"ve_array *", b"ve_array *", "int64_t", "int64_t", "int",
                 b"int32_t *"],
    },
}
