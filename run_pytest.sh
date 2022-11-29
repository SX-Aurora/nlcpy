#!/bin/sh

COV_PATH=nlcpy
COV_OPT="--cov-config=.coveragerc --cov-report=html --cov-append"
NOCOV_OPT=""
PYTEST_DIR=tests/pytest
PYTEST_CMD=pytest

##### fast math tests #####
#
function run_fast_math_tests () {
    local COV=$1
    if [[ $COV = true ]]; then
        _COV_OPT="--cov=$COV_PATH $COV_OPT"
    else
        _COV_OPT=$NOCOV_OPT
        cd ${PYTEST_DIR}
    fi
    echo ""
    echo "---------- Start fast_math tests ----------"
    echo ""
    set -x
    VE_NLCPY_FAST_MATH=yes $PYTEST_CMD $_COV_OPT --fast_math -x
    set +x
    echo ""
    echo "---------- End fast_math tests ------------"
    echo ""
}
#
##### fast math tests #####

##### venode tests #####
#
function run_venode_tests () {
    local COV=$1
    local SCH=$2
    local LPYTEST_DIR=${PYTEST_DIR}
    if [[ $COV = true ]]; then
        _COV_OPT="--cov=$COV_PATH $COV_OPT"
    else
        _COV_OPT=$NOCOV_OPT
        cd ${PYTEST_DIR}
        LPYTEST_DIR=.
    fi
    echo ""
    echo "---------- Start venode tests ----------"
    echo ""
    set -x
    $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests
    VE_NLCPY_NODELIST=1,0 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests
    VE_NLCPY_NODELIST=0 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests
    if [[ $SCH = false ]]; then
        VE_NODE_NUMBER=0 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests
    fi
    $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_use.py
    $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_apply.py
    $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_enter.py
    $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py
    VE_NLCPY_MEMPOOL_SIZE=1024 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py
    VE_NLCPY_MEMPOOL_SIZE=1024B $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py
    VE_NLCPY_MEMPOOL_SIZE=2K $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py
    VE_NLCPY_MEMPOOL_SIZE=2M $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py
    VE_NLCPY_MEMPOOL_SIZE=2G $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py
    VE_NLCPY_MEMPOOL_SIZE=1024b VE_NLCPY_NODELIST=0,1 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py
    VE_NLCPY_MEMPOOL_SIZE=2m VE_NLCPY_NODELIST=1,0 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py
    if [[ $SCH = false ]]; then
        VE_NLCPY_MEMPOOL_SIZE=2G VE_NODE_NUMBER=0 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py
        VE_NLCPY_MEMPOOL_SIZE=2k _VENODELIST="0 1" $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py
        VE_NLCPY_MEMPOOL_SIZE=2g _VENODELIST="1 0" $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py
        VE_NODE_NUMBER=1 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py
        _VENODELIST="0 1 2" VE_NLCPY_NODELIST=2,1 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py
    fi
    VE_NLCPY_NODELIST=0,100 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err
    VE_NLCPY_NODELIST=0,1,0 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err
    if [[ $SCH = false ]]; then
        VE_NODE_NUMBER=100 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err
        _VENODELIST="0 100" $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err
        _VENODELIST="0 1 0" $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err
        _VENODELIST="0 1" VE_NLCPY_NODELIST=1,2 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err
        _VENODELIST="0 1" VE_NLCPY_NODELIST=1,-2 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err
        _VENODELIST="0 1" VE_NLCPY_NODELIST=0,1,1 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err
    fi
    MPIRANK=0 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_runtime_error_at_import.py --import_err
    MPIRANK=0 _MPI4PYVE_MPI_INITIALIZED=0 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_runtime_error_at_import.py --import_err
    if [[ $SCH = false ]]; then
        MPIRANK=0 _MPI4PYVE_MPI_INITIALIZED=1 _VENODELIST="100 0" $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err
        MPIRANK=0 _MPI4PYVE_MPI_INITIALIZED=1 _VENODELIST="1 0 1" $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err
        MPIRANK=0 _MPI4PYVE_MPI_INITIALIZED=1 _VENODELIST="0 1" VE_NLCPY_NODELIST=0,1,2 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err
        MPIRANK=0 _MPI4PYVE_MPI_INITIALIZED=1 _VENODELIST="0 1" VE_NLCPY_NODELIST=-1,0 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err
        MPIRANK=0 _MPI4PYVE_MPI_INITIALIZED=1 _VENODELIST="0 1" VE_NLCPY_NODELIST=0,0 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err
        MPIRANK=0 _MPI4PYVE_MPI_INITIALIZED=1 _MPI4PYVE_MPI_LOCAL_SIZE=1 _VENODELIST="0 1" $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py
        MPIRANK=0 _MPI4PYVE_MPI_INITIALIZED=1 _MPI4PYVE_MPI_LOCAL_SIZE=1 _VENODELIST="0 1" VE_NLCPY_NODELIST=0,1 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py
    fi
    MPIRANK=0 _MPI4PYVE_MPI_INITIALIZED=1 VE_NLCPY_NODELIST=100,0 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err
    MPIRANK=0 _MPI4PYVE_MPI_INITIALIZED=1 VE_NLCPY_NODELIST=1,0,1 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err
    MPIRANK=0 _MPI4PYVE_MPI_INITIALIZED=1 _MPI4PYVE_MPI_LOCAL_SIZE=1000 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err
    MPIRANK=0 _MPI4PYVE_MPI_INITIALIZED=1 _MPI4PYVE_MPI_LOCAL_SIZE=1 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py
    MPIRANK=0 _MPI4PYVE_MPI_INITIALIZED=1 _MPI4PYVE_MPI_LOCAL_SIZE=1 VE_NLCPY_NODELIST=0,1 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py
    # TODO: enable following tests when using 11E-ftrace
    # $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_ftrace_ves.py --ftrace_gen
    # $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_ftrace_ves.py --ftrace_chk
    set +x
    echo ""
    echo "---------- End venode tests ------------"
    echo ""
}
#
##### venode tests #####


##### full tests #####
#
function run_full_ufunc_tests () {
    local COV=$1
    if [[ $COV = true ]]; then
        _COV_OPT="--cov=$COV_PATH $COV_OPT"
    else
        _COV_OPT=$NOCOV_OPT
        cd ${PYTEST_DIR}
    fi
    echo ""
    echo "---------- Start full ufunc tests ----------"
    echo ""
    set -x
    FILES=`find . -path "*ufunc_tests/*" -type f -name "test*.py"`
    for f in $FILES; do
        VE_NLCPY_FAST_MATH=no $PYTEST_CMD $_COV_OPT --test=full $f
    done
    set +x
    echo ""
    echo "---------- End full ufunc tests ------------"
    echo ""
}
function run_full_tests () {
    local COV=$1
    if [[ $COV = true ]]; then
        _COV_OPT="--cov=$COV_PATH $COV_OPT"
    else
        _COV_OPT=$NOCOV_OPT
        cd ${PYTEST_DIR}
    fi
    echo ""
    echo "---------- Start full tests ----------"
    echo ""
    set -x
    VE_NLCPY_FAST_MATH=no $PYTEST_CMD $_COV_OPT --test=full  -k "not ufunc_tests"
    set +x
    echo ""
    echo "---------- End full tests ------------"
    echo ""
}
#
##### full tests #####


##### small tests #####
#
function run_small_tests () {
    local COV=$1
    if [[ $COV = true ]]; then
        _COV_OPT="--cov=$COV_PATH $COV_OPT"
    else
        _COV_OPT=$NOCOV_OPT
        cd ${PYTEST_DIR}
    fi
    echo ""
    echo "---------- Start small tests ----------"
    echo ""
    set -x
    VE_NLCPY_FAST_MATH=no $PYTEST_CMD $_COV_OPT -x
    set +x
    echo ""
    echo "---------- End small tests ------------"
    echo ""
}
#
##### small tests #####

##### clean coverage #####
#
function clean_coverage () {
    rm -rf .coverage .coverage.* htmlcov
}
#
##### clean coverage #####


##### usage #####
#
function usage () {
    echo "Usage: run_pytest.sh [ARGUMENT]..."
    echo ""
    echo "  ARGUMENT:"
    echo "  --test MODE1,MODE2,... or --test=MODE1,MODE2,...: specify the test MODE"
    echo "         available MODEs are [full|small|fast_math|venode]"
    echo "  --cov,-c: specify whether report coverage or not"
    echo "         (default: no coverage)"
    echo "  --cov-path PATH or --cov-path=PATH: specify coverage path"
    echo "         (default: nlcpy)"
    echo "  --scheduler,-s: specify whether job is launched by scheduler or not"
    echo "         (default: not scheduler)"
    echo "  --pytest-cmd CMD or --pytest-cmd=CMD: specify pytest command"
    echo "         (default: pytest)"
    echo ""
}
#
##### usage #####


##### main #####
#
COV=false
SCH=false

while (( $# > 0 ))
do
    case $1 in
        --test | --test=*)
            if [[ "$1" =~ ^--test= ]]; then
                TEST=$(echo $1 | sed -e 's/^--test=//')
            elif [[ -z "$2" ]] || [[ "$2" =~ ^-+ ]]; then
                usage
                exit 1
            else
                TEST="$2"
                shift
            fi
            TESTS=(${TEST//,/ })
            for t in ${TESTS[@]}
            do
                if [[ $t != "full" ]] && [[ $t != "small" ]] && [[ $t != "fast_math" ]] && [[ $t != "venode" ]]; then
                    usage
                    exit 1
                fi
            done
        ;;
        --cov | -c)
            COV=true
        ;;

        --cov-path | --cov-path=*)
            if [[ "$1" =~ ^--cov-path= ]]; then
                COV_PATH=$(echo $1 | sed -e 's/^--cov-path=//')
            elif [[ -z "$2" ]] || [[ "$2" =~ ^-+ ]]; then
                usage
                exit 1
            else
                COV_PATH="$2"
                shift
            fi
            if [[ -z ${COV_PATH} ]]; then
                usage
                exit 1
            fi
        ;;


        --scheduler | -s)
            SCH=true
        ;;
        --pytest-cmd | --pytest-cmd=*)
            if [[ "$1" =~ ^--pytest-cmd= ]]; then
                PYTEST_CMD=$(echo $1 | sed -e 's/^--pytest-cmd=//')
            elif [[ -z "$2" ]] || [[ "$2" =~ ^-+ ]]; then
                usage
                exit 1
            else
                PYTEST_CMD="$2"
                shift
            fi
            if [[ -z ${PYTEST_CMD} ]]; then
                usage
                exit 1
            fi
        ;;
        -h | --help)
            usage
            exit 1
        ;;
    esac
    shift
done

if [[ ${#TESTS[@]} -eq 0 ]]; then
    usage
    exit 1
fi

echo "test: ${TESTS[@]}"
echo "cov: ${COV}"
echo "cov-path: ${COV_PATH}"
echo "scheduler: ${SCH}"
echo "pytest-cmd: ${PYTEST_CMD}"

clean_coverage

for t in ${TESTS[@]}
do
    if [[ $t = "full" ]]; then
        (run_full_tests $COV)
        (run_full_ufunc_tests $COV)
    elif [[ $t = "fast_math" ]]; then
        (run_fast_math_tests $COV)
    elif [[ $t = "venode" ]]; then
        (run_venode_tests $COV $SCH)
    else
        (run_small_tests $COV)
    fi
done

#
##### main #####
