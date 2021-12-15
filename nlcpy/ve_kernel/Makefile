.SUFFIXES: .o .c .h .c.master .h.master .master.m4 .master .c.m4

BASEDIR = ../..

include $(BASEDIR)/make.inc

NLCPATH=$(shell ls -d /opt/nec/ve/nlc/*.*.*|sort -rV |head -1)
LIBPATH=-L$(NLCPATH)/lib
INCPATH=-I$(NLCPATH)/include -I.
LIB_NLC=-lasl_openmp_i64 -llapack_i64 -lcblas_i64 -lblas_openmp_i64 -lsca_openmp_i64

LIB_COMMON=$(LIBDIR)/libnlcpy_ve_kernel_common.so
ifeq ($(FAST_MATH),yes)
LIB=$(LIBDIR)/libnlcpy_ve_kernel_fast_math.so
else
LIB=$(LIBDIR)/libnlcpy_ve_kernel_no_fast_math.so
endif
LIB_PROF=$(LIBDIR)/libnlcpy_profiling.so

CC=ncc
CFLAGS=-fpic -O2 -report-all -fopenmp
LDFLAGS=-shared -L$(LIBDIR) -l$(patsubst lib%.so,%,$(notdir $(LIB_COMMON))) $(LIBPATH) $(LIB_NLC) -fopenmp
LDFLAGS_COMMON=-shared $(LIBPATH) $(LIB_NLC) -fopenmp
ifeq ($(FTRACE),yes)
CFLAGS+=-ftrace
LDFLAGS+=-ftrace
LIB_NLC+=-lveftrace_p
endif
LDFLAGS_PROF=-shared -fpic -fopenmp -ftrace -lveftrace_p

CPP = $(TOOLDIR)/cppnlcpy.pl
ifeq ($(NO_OPERATOR),yes)
CFLAGS+=-DNO_OPERATOR
CPPFLAGS+=-DNO_OPERATOR
else ifeq ($(ADD_ONLY),yes)
CFLAGS+=-DADD_ONLY
CPPFLAGS+=-DADD_ONLY
endif

ifeq ($(FAST_MATH),yes)
CFLAGS+=\
	-ffast-math \
	-mno-vector-intrinsic-check \
	-freciprocal-math \
	-mvector-power-to-explog \
	-mvector-low-precise-divide-function
else
CFLAGS+=\
	-fno-fast-math \
	-mvector-intrinsic-check \
	-fno-reciprocal-math \
	-mno-vector-power-to-explog \
	-mno-vector-low-precise-divide-function
endif

# Sorting
VE_SORTING = \
	nlcpy_sort.c\
	nlcpy_argsort.c\
	nlcpy_sort_multi.c

# UNARY FUNCTION
ifneq ($(NO_OPERATOR),yes)
ifneq ($(ADD_ONLY),yes)
VE_UNARY_FUNCTION =\
  nlcpy_positive.c\
  nlcpy_negative.c\
  nlcpy_invert.c\
  nlcpy_logical_not.c\
  nlcpy_sin.c\
  nlcpy_cos.c\
  nlcpy_tan.c\
  nlcpy_arcsin.c\
  nlcpy_arccos.c\
  nlcpy_arctan.c\
  nlcpy_sinh.c\
  nlcpy_cosh.c\
  nlcpy_tanh.c\
  nlcpy_arcsinh.c\
  nlcpy_arccosh.c\
  nlcpy_arctanh.c\
  nlcpy_exp.c\
  nlcpy_expm1.c\
  nlcpy_exp2.c\
  nlcpy_log.c\
  nlcpy_log10.c\
  nlcpy_log2.c\
  nlcpy_log1p.c\
  nlcpy_rad2deg.c\
  nlcpy_deg2rad.c\
  nlcpy_radians.c\
  nlcpy_degrees.c\
  nlcpy_absolute.c\
  nlcpy_fabs.c\
  nlcpy_sqrt.c\
  nlcpy_cbrt.c\
  nlcpy_square.c\
  nlcpy_reciprocal.c\
  nlcpy_sign.c\
  nlcpy_rint.c\
  nlcpy_conj.c\
  nlcpy_conjugate.c\
  nlcpy_ceil.c\
  nlcpy_trunc.c\
  nlcpy_floor.c\
  nlcpy_isfinite.c\
  nlcpy_isinf.c\
  nlcpy_isnan.c\
  nlcpy_signbit.c\
  nlcpy_spacing.c\
  nlcpy_erf.c\
  nlcpy_erfc.c
endif
endif
VE_COPY_FUNCTION =\
  nlcpy_copy.c\
  nlcpy_angle.c

VE_COPY_MASKED_FUNCTION =\
  nlcpy_copy_masked.c\

# BINARY OPERATOR
ifneq ($(NO_OPERATOR),yes)
ifneq ($(ADD_ONLY),yes)
VE_BINARY_FUNCTION_COMMON = \
  nlcpy_add.c\
  nlcpy_subtract.c\
  nlcpy_multiply.c\
  nlcpy_bitwise_and.c\
  nlcpy_bitwise_xor.c\
  nlcpy_bitwise_or.c\
  nlcpy_logical_and.c\
  nlcpy_logical_xor.c\
  nlcpy_logical_or.c\
  nlcpy_right_shift.c\
  nlcpy_left_shift.c\
  nlcpy_less.c\
  nlcpy_greater.c\
  nlcpy_less_equal.c\
  nlcpy_greater_equal.c\
  nlcpy_equal.c\
  nlcpy_not_equal.c\
  nlcpy_maximum.c\
  nlcpy_minimum.c\
  nlcpy_hypot.c\
  nlcpy_heaviside.c\
  nlcpy_ldexp.c\
  nlcpy_copysign.c\
  nlcpy_fmax.c\
  nlcpy_fmin.c\
  nlcpy_nextafter.c
VE_BINARY_FUNCTION = \
  nlcpy_floor_divide.c\
  nlcpy_true_divide.c\
  nlcpy_divide.c\
  nlcpy_mod.c\
  nlcpy_remainder.c\
  nlcpy_power.c\
  nlcpy_arctan2.c\
  nlcpy_logaddexp.c\
  nlcpy_logaddexp2.c\
  nlcpy_fmod.c
else ifneq ($(NO_OPERATOR),yes)
VE_BINARY_FUNCTION = nlcpy_add.c
endif
endif
#VE_BINARY_FUNCTION +=\
#  nlcpy_arange.c

### MATMUL OPERATOR
VE_MATMUL_OPERATOR = nlcpy_matmul.c
# nlcpy_imatmul.c  # not support yet

VE_CAST = \
  nlcpy_cast.c

# REDUCTION FUNCTION
ifneq ($(NO_OPERATOR),yes)
ifneq ($(ADD_ONLY),yes)
VE_REDUCE_COMMON = \
  nlcpy_add_reduce.c\
  nlcpy_subtract_reduce.c\
  nlcpy_multiply_reduce.c\
  nlcpy_bitwise_and_reduce.c\
  nlcpy_bitwise_xor_reduce.c\
  nlcpy_bitwise_or_reduce.c\
  nlcpy_logical_and_reduce.c\
  nlcpy_logical_xor_reduce.c\
  nlcpy_logical_or_reduce.c\
  nlcpy_right_shift_reduce.c\
  nlcpy_left_shift_reduce.c\
  nlcpy_less_reduce.c\
  nlcpy_greater_reduce.c\
  nlcpy_less_equal_reduce.c\
  nlcpy_greater_equal_reduce.c\
  nlcpy_equal_reduce.c\
  nlcpy_not_equal_reduce.c\
  nlcpy_maximum_reduce.c\
  nlcpy_minimum_reduce.c\
  nlcpy_hypot_reduce.c\
  nlcpy_heaviside_reduce.c\
  nlcpy_copysign_reduce.c\
  nlcpy_fmax_reduce.c\
  nlcpy_fmin_reduce.c\
  nlcpy_nextafter_reduce.c
VE_REDUCE = \
  nlcpy_floor_divide_reduce.c\
  nlcpy_true_divide_reduce.c\
  nlcpy_divide_reduce.c\
  nlcpy_mod_reduce.c\
  nlcpy_remainder_reduce.c\
  nlcpy_power_reduce.c\
  nlcpy_arctan2_reduce.c\
  nlcpy_logaddexp_reduce.c\
  nlcpy_logaddexp2_reduce.c\
  nlcpy_fmod_reduce.c
else
VE_REDUCE = nlcpy_add_reduce.c
endif
endif

ifneq ($(NO_OPERATOR),yes)
ifneq ($(ADD_ONLY),yes)
# ACCUMULATION FUNCTION
VE_ACCUMULATE_COMMON = \
  nlcpy_add_accumulate.c\
  nlcpy_subtract_accumulate.c\
  nlcpy_multiply_accumulate.c\
  nlcpy_bitwise_and_accumulate.c\
  nlcpy_bitwise_or_accumulate.c\
  nlcpy_bitwise_xor_accumulate.c\
  nlcpy_left_shift_accumulate.c\
  nlcpy_right_shift_accumulate.c\
  nlcpy_greater_accumulate.c\
  nlcpy_greater_equal_accumulate.c\
  nlcpy_less_accumulate.c\
  nlcpy_less_equal_accumulate.c\
  nlcpy_not_equal_accumulate.c\
  nlcpy_equal_accumulate.c\
  nlcpy_logical_and_accumulate.c\
  nlcpy_logical_or_accumulate.c\
  nlcpy_logical_xor_accumulate.c\
  nlcpy_maximum_accumulate.c\
  nlcpy_minimum_accumulate.c\
  nlcpy_hypot_accumulate.c\
  nlcpy_heaviside_accumulate.c\
  nlcpy_copysign_accumulate.c\
  nlcpy_fmax_accumulate.c\
  nlcpy_fmin_accumulate.c\
  nlcpy_nextafter_accumulate.c
VE_ACCUMULATE = \
  nlcpy_divide_accumulate.c\
  nlcpy_logaddexp_accumulate.c\
  nlcpy_logaddexp2_accumulate.c\
  nlcpy_true_divide_accumulate.c\
  nlcpy_floor_divide_accumulate.c\
  nlcpy_power_accumulate.c\
  nlcpy_remainder_accumulate.c\
  nlcpy_mod_accumulate.c\
  nlcpy_fmod_accumulate.c\
  nlcpy_arctan2_accumulate.c
else
VE_ACCUMULATE = nlcpy_add_accumulate.c
endif
endif

# OUTER FUNCTION
ifneq ($(NO_OPERATOR),yes)
ifneq ($(ADD_ONLY),yes)
VE_OUTER_COMMON = \
  nlcpy_add_outer.c\
  nlcpy_subtract_outer.c\
  nlcpy_multiply_outer.c\
  nlcpy_bitwise_and_outer.c\
  nlcpy_bitwise_xor_outer.c\
  nlcpy_bitwise_or_outer.c\
  nlcpy_logical_and_outer.c\
  nlcpy_logical_xor_outer.c\
  nlcpy_logical_or_outer.c\
  nlcpy_right_shift_outer.c\
  nlcpy_left_shift_outer.c\
  nlcpy_less_outer.c\
  nlcpy_greater_outer.c\
  nlcpy_less_equal_outer.c\
  nlcpy_greater_equal_outer.c\
  nlcpy_equal_outer.c\
  nlcpy_not_equal_outer.c\
  nlcpy_maximum_outer.c\
  nlcpy_minimum_outer.c\
  nlcpy_hypot_outer.c\
  nlcpy_heaviside_outer.c\
  nlcpy_ldexp_outer.c\
  nlcpy_copysign_outer.c\
  nlcpy_fmax_outer.c\
  nlcpy_fmin_outer.c\
  nlcpy_nextafter_outer.c
VE_OUTER = \
  nlcpy_floor_divide_outer.c\
  nlcpy_true_divide_outer.c\
  nlcpy_divide_outer.c\
  nlcpy_power_outer.c\
  nlcpy_mod_outer.c\
  nlcpy_remainder_outer.c\
  nlcpy_arctan2_outer.c\
  nlcpy_logaddexp_outer.c\
  nlcpy_logaddexp2_outer.c\
  nlcpy_fmod_outer.c
else
VE_OUTER = nlcpy_add_outer.c
endif
endif

ifneq ($(NO_OPERATOR),yes)
ifneq ($(ADD_ONLY),yes)
# REDUCEAT FUNCTION
VE_REDUCEAT_COMMON = \
  nlcpy_add_reduceat.c\
  nlcpy_subtract_reduceat.c\
  nlcpy_multiply_reduceat.c\
  nlcpy_bitwise_and_reduceat.c\
  nlcpy_bitwise_xor_reduceat.c\
  nlcpy_bitwise_or_reduceat.c\
  nlcpy_logical_and_reduceat.c\
  nlcpy_logical_xor_reduceat.c\
  nlcpy_logical_or_reduceat.c\
  nlcpy_right_shift_reduceat.c\
  nlcpy_left_shift_reduceat.c\
  nlcpy_less_reduceat.c\
  nlcpy_greater_reduceat.c\
  nlcpy_less_equal_reduceat.c\
  nlcpy_greater_equal_reduceat.c\
  nlcpy_equal_reduceat.c\
  nlcpy_not_equal_reduceat.c\
  nlcpy_maximum_reduceat.c\
  nlcpy_minimum_reduceat.c\
  nlcpy_hypot_reduceat.c\
  nlcpy_heaviside_reduceat.c\
  nlcpy_copysign_reduceat.c\
  nlcpy_fmax_reduceat.c\
  nlcpy_fmin_reduceat.c\
  nlcpy_nextafter_reduceat.c
VE_REDUCEAT = \
  nlcpy_divide_reduceat.c\
  nlcpy_floor_divide_reduceat.c\
  nlcpy_true_divide_reduceat.c\
  nlcpy_mod_reduceat.c\
  nlcpy_remainder_reduceat.c\
  nlcpy_power_reduceat.c\
  nlcpy_arctan2_reduceat.c\
  nlcpy_logaddexp_reduceat.c\
  nlcpy_logaddexp2_reduceat.c\
  nlcpy_fmod_reduceat.c
else
VE_REDUCEAT = nlcpy_add_reduceat.c
endif
endif

# ARGFUNC
VE_ARGMAX = nlcpy_argmax.c
VE_ARGMIN = nlcpy_argmin.c
VE_ARGFUNC = $(VE_ARGMAX) $(VE_ARGMIN)


# Linear Algebra
VE_DOT = nlcpy_dot.c
VE_LINALG = \
  nlcpy_solve.c\
  nlcpy_inv.c\
  nlcpy_lstsq.c\
  nlcpy_svd.c\
  nlcpy_eig.c\
  nlcpy_eigh.c\
  nlcpy_norm.c\
  nlcpy_cholesky.c\
  nlcpy_qr.c

VE_MATH = nlcpy_clip.c

VE_BASIC_MASTER =\
  nlcpy_arange.c.master\
  nlcpy_boolean_mask.c.master\
  nlcpy_take.c.master\
  nlcpy_prepare_indexing.c.master\
  nlcpy_scatter.c.master\
  nlcpy_eye.c.master\
  nlcpy_linspace.c.master\
  nlcpy_tri.c.master\
  nlcpy_nonzero.c.master\
  nlcpy_where.c.master\
  nlcpy_argwhere.c.master\
  nlcpy_tile.c.master\
  nlcpy_repeat.c.master\
  nlcpy_diff.c.master\
  nlcpy_delete.c.master\
  nlcpy_insert.c.master\
  nlcpy_roll.c.master\
  nlcpy_fill_diagonal.c.master\
  nlcpy_block.c.master\
  nlcpy_domain_mask.c.master


VE_CONTROLLER =\
  ve_controller.c\
  ve_selector.c


VE_BASIC_KERNEL = $(VE_BASIC_MASTER:.c.master=.c)

VE_RANDOM_KERNEL =\
  nlcpy_random.c\
  nlcpy_random_shuffle.c

VE_FFT_KERNEL =\
  nlcpy_fft.c

VE_SCA_KERNEL =\
  nlcpy_sca.c

VE_CBLAS_WRAPPER = cblas_wrapper.c

VE_UTILITY = array_utility.c

VE_MEMPOOL = nlcpy_mempool_ve.c

OBJS = \
  $(VE_UNARY_FUNCTION:.c=.o)\
  $(VE_BINARY_FUNCTION:.c=.o)\
  $(VE_REDUCE:.c=.o)\
  $(VE_REDUCEAT:.c=.o)\
  $(VE_ACCUMULATE:.c=.o)\
  $(VE_OUTER:.c=.o)

OBJS_COMMON = \
  $(VE_UTILITY:.c=.o)\
  $(VE_MEMPOOL:.c=.o)\
  $(VE_COPY_FUNCTION:.c=.o)\
  $(VE_COPY_MASKED_FUNCTION:.c=.o)\
  $(VE_MATMUL_OPERATOR:.c=.o)\
  $(VE_RANDOM_KERNEL:.c=.o)\
  $(VE_FFT_KERNEL:.c=.o)\
  $(VE_SCA_KERNEL:.c=.o)\
  $(VE_CBLAS_WRAPPER:.c=.o)\
  $(VE_SORTING:.c=.o)\
  $(VE_ARGFUNC:.c=.o)\
  $(VE_DOT:.c=.o)\
  $(VE_LINALG:.c=.o)\
  $(VE_MATH:.c=.o)\
  $(VE_CAST:.c=.o)\
  $(VE_BASIC_KERNEL:.c=.o)\
  $(VE_CONTROLLER:.c=.o)\
  $(VE_BINARY_FUNCTION_COMMON:.c=.o)\
  $(VE_REDUCE_COMMON:.c=.o)\
  $(VE_REDUCEAT_COMMON:.c=.o)\
  $(VE_ACCUMULATE_COMMON:.c=.o)\
  $(VE_OUTER_COMMON:.c=.o)\
  xerbla.o


ifeq ($(FAST_MATH),yes)
all: make.dep $(LIB)
else ifeq ($(COMMON),yes)
all: make.dep $(LIB_COMMON) $(LIB_PROF)
else
all: make.dep $(LIB)
endif

make.dep:
	sh $(TOOLDIR)/make_dep.sh

$(LIB): $(OBJS)
	mkdir -p $(LIBDIR) && $(CC) -o $@ $(OBJS) ${LDFLAGS}

$(LIB_COMMON): $(OBJS_COMMON)
	mkdir -p $(LIBDIR) && $(CC) -o $@ $^ ${LDFLAGS_COMMON}

$(LIB_PROF): nlcpy_profiling.c
	mkdir -p $(LIBDIR) && $(CC) -o $@ $^ ${LDFLAGS_PROF}

include make.dep

perl: \
	$(VE_UNARY_FUNCTION)\
	$(VE_COPY_FUNCTION)\
	$(VE_COPY_MASKED_FUNCTION)\
	$(VE_BINARY_FUNCTION)\
	$(VE_BINARY_FUNCTION_COMMON)\
	$(VE_MATMUL_OPERATOR)\
	$(VE_REDUCE)\
	$(VE_REDUCE_COMMON)\
	$(VE_ACCUMULATE)\
	$(VE_ACCUMULATE_COMMON)\
	$(VE_REDUCEAT)\
	$(VE_REDUCEAT_COMMON)\
	$(VE_CAST)\
	$(VE_OUTER)\
	$(VE_OUTER_COMMON)\
	$(VE_ARGFUNC)\
	$(VE_DOT)\
	$(VE_LINALG)\
	$(VE_MATH)

$(VE_UNARY_FUNCTION): unary_operator.c.master unary_operator.c.master2
	rm -f $@
	$(CPP) $(CPPFLAGS) unary_operator.c.master $@

$(VE_COPY_FUNCTION): copy.c.master
	rm -f $@
	$(CPP) $(CPPFLAGS) $? $@

$(VE_COPY_MASKED_FUNCTION): copy_masked.c.master
	rm -f $@
	$(CPP) $(CPPFLAGS) $? $@

$(VE_BINARY_FUNCTION) $(VE_BINARY_FUNCTION_COMMON): binary_operator.c.master binary_operator.c.master2
	rm -f $@
	$(CPP) $(CPPFLAGS) binary_operator.c.master $@

$(VE_MATMUL_OPERATOR): matmul_operator.c.master
	rm -f $@
	$(CPP) $(CPPFLAGS) $? $@

$(VE_REDUCE) $(VE_REDUCE_COMMON): reduce.c.master
	rm -f $@
	$(CPP) $(CPPFLAGS) $? $@

$(VE_ACCUMULATE) $(VE_ACCUMULATE_COMMON): accumulate.c.master
	rm -f $@
	$(CPP) $(CPPFLAGS) $? $@

$(VE_REDUCEAT) $(VE_REDUCEAT_COMMON): reduceat.c.master
	rm -f $@
	$(CPP) $(CPPFLAGS) $? $@

$(VE_CAST): cast.c.master
	rm -f $@
	$(CPP) $(CPPFLAGS) $? $@

$(VE_OUTER) $(VE_OUTER_COMMON): outer.c.master
	rm -f $@
	$(CPP) $(CPPFLAGS) $? $@

$(VE_ARGFUNC): argfunc.c.master
	rm -f $@
	$(CPP) $(CPPFLAGS) $? $@

$(VE_DOT): $(VE_DOT:.c=.c.master)
	rm -f $@
	$(CPP) $(CPPFLAGS) $? $@

%.c: %.c.master
	rm -f $@
	$(CPP) $(CPPFLAGS) $< $@

%.o: %.c
	$(CC) -c ${CFLAGS} -o $@ $(@F:%.o=%.c) ${INCPATH}

ifneq ($(FAST_MATH),yes)
nlcpy_remainder.o:CFLAGS+=-mno-vector-fma
nlcpy_mod.o:CFLAGS+=-mno-vector-fma
nlcpy_floor_divide.o:CFLAGS+=-mno-vector-fma
nlcpy_absolute.o:CFLAGS+=-mno-vector-intrinsic-check
endif

clean:
	rm -rf $(LIB)
	cd $(OBJDIR) && rm -rf $(OBJS) $(OBJS_COMMON)
	cd $(OBJDIR) && rm -rf $(VE_UNARY_FUNCTION)
	cd $(OBJDIR) && rm -rf $(VE_BINARY_FUNCTION)
	cd $(OBJDIR) && rm -rf $(VE_MATMUL_OPERATOR)
	cd $(OBJDIR) && rm -rf $(VE_ARGFUNC)
	cd $(OBJDIR) && rm -rf $(VE_LINALG)
	cd $(OBJDIR) && rm -rf $(VE_DOT)
	cd $(OBJDIR) && rm -rf $(VE_MATH)
