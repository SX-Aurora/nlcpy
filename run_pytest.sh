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
    local LPYTEST_DIR=${PYTEST_DIR}
    local retcode=0
    if [[ $COV = true ]]; then
        _COV_OPT="--cov=$COV_PATH $COV_OPT"
    else
        _COV_OPT=$NOCOV_OPT
        cd ${PYTEST_DIR}
        LPYTEST_DIR=.
    fi
    echo ""
    echo "---------- Start fast_math tests ----------"
    echo ""
    set -x
    VE_NLCPY_FAST_MATH=yes $PYTEST_CMD $LPYTEST_DIR $_COV_OPT --fast_math -x || retcode=$(($? | retcode))
    set +x
    echo ""
    echo "---------- End fast_math tests ------------"
    echo ""
    return $retcode
}
#
##### fast math tests #####

##### venode tests #####
#
function run_venode_tests () {
    local COV=$1
    local SCH=$2
    local LPYTEST_DIR=${PYTEST_DIR}
    local retcode=0
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
    $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests || retcode=$(($? | retcode))
    VE_NLCPY_NODELIST=1,0 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests || retcode=$(($? | retcode))
    VE_NLCPY_NODELIST=0 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests || retcode=$(($? | retcode))
    if [[ $SCH = false ]]; then
        VE_NODE_NUMBER=0 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests || retcode=$(($? | retcode))
    fi
    $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_use.py || retcode=$(($? | retcode))
    $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_apply.py || retcode=$(($? | retcode))
    $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_enter.py || retcode=$(($? | retcode))
    $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py || retcode=$(($? | retcode))
    VE_NLCPY_MEMPOOL_SIZE=1024 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py || retcode=$(($? | retcode))
    VE_NLCPY_MEMPOOL_SIZE=1024B $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py || retcode=$(($? | retcode))
    VE_NLCPY_MEMPOOL_SIZE=2K $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py || retcode=$(($? | retcode))
    VE_NLCPY_MEMPOOL_SIZE=2M $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py || retcode=$(($? | retcode))
    VE_NLCPY_MEMPOOL_SIZE=2G $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py || retcode=$(($? | retcode))
    VE_NLCPY_MEMPOOL_SIZE=1024b VE_NLCPY_NODELIST=0,1 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py || retcode=$(($? | retcode))
    VE_NLCPY_MEMPOOL_SIZE=2m VE_NLCPY_NODELIST=1,0 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py || retcode=$(($? | retcode))
    if [[ $SCH = false ]]; then
        VE_NLCPY_MEMPOOL_SIZE=2G VE_NODE_NUMBER=0 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py || retcode=$(($? | retcode))
        VE_NLCPY_MEMPOOL_SIZE=2k _VENODELIST="0 1" $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py || retcode=$(($? | retcode))
        VE_NLCPY_MEMPOOL_SIZE=2g _VENODELIST="1 0" $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py || retcode=$(($? | retcode))
        VE_NODE_NUMBER=1 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py || retcode=$(($? | retcode))
        _VENODELIST="0 1 2" VE_NLCPY_NODELIST=2,1 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py || retcode=$(($? | retcode))
    fi
    VE_NLCPY_NODELIST=0,100 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err || retcode=$(($? | retcode))
    VE_NLCPY_NODELIST=0,1,0 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err || retcode=$(($? | retcode))
    if [[ $SCH = false ]]; then
        VE_NODE_NUMBER=100 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err || retcode=$(($? | retcode))
        _VENODELIST="0 100" $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err || retcode=$(($? | retcode))
        _VENODELIST="0 1 0" $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err || retcode=$(($? | retcode))
        _VENODELIST="0 1" VE_NLCPY_NODELIST=1,2 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err || retcode=$(($? | retcode))
        _VENODELIST="0 1" VE_NLCPY_NODELIST=1,-2 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err || retcode=$(($? | retcode))
        _VENODELIST="0 1" VE_NLCPY_NODELIST=0,1,1 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err || retcode=$(($? | retcode))
    fi
    MPIRANK=0 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_runtime_error_at_import.py --import_err || retcode=$(($? | retcode))
    MPIRANK=0 _MPI4PYVE_MPI_INITIALIZED=0 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_runtime_error_at_import.py --import_err || retcode=$(($? | retcode))
    if [[ $SCH = false ]]; then
        MPIRANK=0 _MPI4PYVE_MPI_INITIALIZED=1 _VENODELIST="100 0" $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err || retcode=$(($? | retcode))
        MPIRANK=0 _MPI4PYVE_MPI_INITIALIZED=1 _VENODELIST="1 0 1" $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err || retcode=$(($? | retcode))
        MPIRANK=0 _MPI4PYVE_MPI_INITIALIZED=1 _VENODELIST="0 1" VE_NLCPY_NODELIST=0,1,2 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err || retcode=$(($? | retcode))
        MPIRANK=0 _MPI4PYVE_MPI_INITIALIZED=1 _VENODELIST="0 1" VE_NLCPY_NODELIST=-1,0 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err || retcode=$(($? | retcode))
        MPIRANK=0 _MPI4PYVE_MPI_INITIALIZED=1 _VENODELIST="0 1" VE_NLCPY_NODELIST=0,0 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err || retcode=$(($? | retcode))
        MPIRANK=0 _MPI4PYVE_MPI_INITIALIZED=1 _MPI4PYVE_MPI_LOCAL_SIZE=1 _VENODELIST="0 1" $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py || retcode=$(($? | retcode))
        MPIRANK=0 _MPI4PYVE_MPI_INITIALIZED=1 _MPI4PYVE_MPI_LOCAL_SIZE=1 _VENODELIST="0 1" VE_NLCPY_NODELIST=0,1 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py || retcode=$(($? | retcode))
    fi
    MPIRANK=0 _MPI4PYVE_MPI_INITIALIZED=1 VE_NLCPY_NODELIST=100,0 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err || retcode=$(($? | retcode))
    MPIRANK=0 _MPI4PYVE_MPI_INITIALIZED=1 VE_NLCPY_NODELIST=1,0,1 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err || retcode=$(($? | retcode))
    MPIRANK=0 _MPI4PYVE_MPI_INITIALIZED=1 _MPI4PYVE_MPI_LOCAL_SIZE=1000 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_value_error_at_import.py --import_err || retcode=$(($? | retcode))
    MPIRANK=0 _MPI4PYVE_MPI_INITIALIZED=1 _MPI4PYVE_MPI_LOCAL_SIZE=1 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py || retcode=$(($? | retcode))
    MPIRANK=0 _MPI4PYVE_MPI_INITIALIZED=1 _MPI4PYVE_MPI_LOCAL_SIZE=1 VE_NLCPY_NODELIST=0,1 $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_status.py || retcode=$(($? | retcode))
    $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_ftrace_ves.py --ftrace_gen || retcode=$(($? | retcode))
    $PYTEST_CMD $_COV_OPT -x ${LPYTEST_DIR}/venode_tests/test_ftrace_ves.py --ftrace_chk || retcode=$(($? | retcode))
    set +x
    echo ""
    echo "---------- End venode tests ------------"
    echo ""
    return $retcode
}
#
##### venode tests #####


##### full tests #####
#
function run_full_ufunc_tests () {
    local COV=$1
    local retcode=0
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
        VE_NLCPY_FAST_MATH=no $PYTEST_CMD $_COV_OPT --test=full $f || retcode=$(($? | retcode))
    done
    set +x
    echo ""
    echo "---------- End full ufunc tests ------------"
    echo ""
    return $retcode
}
function run_full_tests () {
    local COV=$1
    local LPYTEST_DIR=${PYTEST_DIR}
    local retcode=0
    if [[ $COV = true ]]; then
        _COV_OPT="--cov=$COV_PATH $COV_OPT"
    else
        _COV_OPT=$NOCOV_OPT
        cd ${PYTEST_DIR}
        LPYTEST_DIR=.
    fi
    echo ""
    echo "---------- Start full tests ----------"
    echo ""
    set -x
    VE_NLCPY_FAST_MATH=no $PYTEST_CMD $LPYTEST_DIR $_COV_OPT --test=full  -k "not ufunc_tests" || retcode=$(($? | retcode))
    set +x
    echo ""
    echo "---------- End full tests ------------"
    echo ""
    return $retcode
}
#
##### full tests #####


##### small tests #####
#
function run_small_tests () {
    local COV=$1
    local LPYTEST_DIR=${PYTEST_DIR}
    local retcode=0
    if [[ $COV = true ]]; then
        _COV_OPT="--cov=$COV_PATH $COV_OPT"
    else
        _COV_OPT=$NOCOV_OPT
        cd ${PYTEST_DIR}
        LPYTEST_DIR=.
    fi
    echo ""
    echo "---------- Start small tests ----------"
    echo ""
    set -x
    VE_NLCPY_FAST_MATH=no $PYTEST_CMD $LPYTEST_DIR $_COV_OPT -x || retcode=$(($? | retcode))
    set +x
    echo ""
    echo "---------- End small tests ------------"
    echo ""
    return $retcode
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

err_state=0

for t in ${TESTS[@]}
do
    if [[ $t = "full" ]]; then
        (run_full_tests $COV)
        err_state=$(($? | err_state))
        (run_full_ufunc_tests $COV)
        err_state=$(($? | err_state))
    elif [[ $t = "fast_math" ]]; then
        (run_fast_math_tests $COV)
        err_state=$(($? | err_state))
    elif [[ $t = "venode" ]]; then
        (run_venode_tests $COV $SCH)
        err_state=$(($? | err_state))
    else
        (run_small_tests $COV)
        err_state=$(($? | err_state))
    fi
done

if [[ $err_state -eq 0 ]]; then
    echo "All tests passed"
else
    echo "!!!!!!!!!!!!!!!!!!!!"
    echo "!!! Tests failed !!!"
    echo "!!!!!!!!!!!!!!!!!!!!"
fi

exit $err_state

#
##### main #####
