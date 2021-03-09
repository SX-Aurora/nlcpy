#!/bin/sh

BASEDIR=$1

cd ${BASEDIR}

rm -rf obj/
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
(cd nlcpy; rm -rf *.c *.so *.cpp *.o __pycache__)
(cd nlcpy/core; rm -rf *.c *.so *.cpp *.o __pycache__)
(cd nlcpy/veo; rm -rf *.c *.so *.cpp *.o __pycache__)
(cd nlcpy/kernel_register; rm -rf *.c *.so *.cpp *.o __pycache__)
(cd nlcpy/creation; rm -rf  __pycache__)
(cd nlcpy/indexing; rm -rf  __pycache__)
(cd nlcpy/error_handler; rm -rf  __pycache__)
(cd nlcpy/manipulation; rm -rf  __pycache__)
(cd nlcpy/ufuncs; rm -rf *.c *.so *.cpp *.o __pycache__)
(cd nlcpy/math; rm -rf *.c *.so *.cpp *.o __pycache__)
(cd nlcpy/error_handler; rm -rf *.c *.so *.cpp *.o __pycache__)
(cd nlcpy/statistics; rm -rf *.c *.so *.cpp *.o __pycache__)
(cd nlcpy/linalg; rm -rf *.c *.so *.cpp *.o __pycache__)
(cd nlcpy/request; rm -rf *.c *.so *.cpp *.o __pycache__)
(cd nlcpy/sorting; rm -rf *.c *.so *.cpp *.o __pycache__)
(cd nlcpy/mempool; rm -rf mempoo*.c *.so *.cpp *.o __pycache__)
(cd nlcpy/random; rm -rf *.c *.so *.cpp *.o __pycache__)
(cd nlcpy/mempool; rm -rf mempoo*.c *.so *.cpp *.o __pycache__)
(cd nlcpy/prof; rm -rf __pycache__)
(cd nlcpy/datatype; rm -rf __pycache__)
(cd nlcpy/logic; rm -rf __pycache__)
(cd nlcpy; rm -rf __pycache__)
(cd nlcpy/testing; rm -rf __pycache__)
(cd tests/pytest; rm -rf __pycache__)
(cd tests/pytest/creation_tests; rm -rf __pycache__)
(cd tests/pytest/manipulation_tests; rm -rf __pycache__)
(cd tests/pytest/math_tests; rm -rf __pycache__)
(cd tests/pytest/indexing_tests; rm -rf __pycache__)
(cd tests/pytest/ufunc_tests; rm -rf __pycache__)
(cd tests/pytest/statistics_tests; rm -rf __pycache__)
(cd tests/pytest/random_tests; rm -rf __pycache__)
(cd tests/pytest/logic_tests; rm -rf __pycache__)
(cd tests/pytest/sorting_tests; rm -rf __pycache__)
rm -rf .tox
rm -rf nlcpy/*.so nlcpy/lib/*.so __pycache__
