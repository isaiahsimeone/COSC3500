#!/bin/bash
ITERATIONS=1000
TESTCOUNT=$ITERATIONS-4

RED='\033[0;31m'
BLUE='\033[1;34m'
PURP='\033[0;35m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color
clr_eol=$(tput el)

# Clean
echo -en "[${CYAN}INFO${NC}] Cleaning..."
make clean 1>/dev/null
rm serial_results.txt avx_results.txt openmp_results.txt >/dev/null 2>&1
rm -rf test_output 1>/dev/null
echo -e " ${GREEN}Done${NC}"

# Build
echo -en "[${CYAN}INFO${NC}] Building all targets..."
make 1>/dev/null
echo -e " ${GREEN}Done${NC}"

# Test directory
echo -en "[${CYAN}INFO${NC}] Creating directory ./test_output"
mkdir -p test_output;
echo -e " ${GREEN}Done${NC}"

echo -e "[${CYAN}INFO${NC}] Running tests with matrix sizes 4 to ${ITERATIONS}..."
echo ""

# Run Serial tests
echo -e "------------------------"
echo -e "                 ${YELLOW}Serial${NC} "
echo -e "------------------------"

for i in $(seq 4 $ITERATIONS);
do
    ./Assignment_serial $i 1>/dev/null
    cp serial_results.txt test_output/serial_${i}.txt
    if [[ $i -le $ITERATIONS ]]; then
        echo -en "${clr_eol}$i/${ITERATIONS} complete\r"
    fi;
done;
echo ""
echo ""


# Run AVX tests
echo -e "------------------------"
echo -e "                    ${YELLOW}AVX${NC} "
echo -e "------------------------"

for i in $(seq 4 $ITERATIONS);
do
    ./Assignment_avx $i 1>/dev/null
    cp avx_results.txt test_output/avx_${i}.txt
    if [[ $i -le $ITERATIONS ]]; then
        echo -en "${clr_eol}$i/${ITERATIONS} complete\r"
    fi;
done;
echo ""
echo ""


# Run OpenMP tests
echo -e "------------------------"
echo -e "                 ${YELLOW}OpenMP${NC} "
echo -e "------------------------"

for i in $(seq 4 $ITERATIONS);
do
    ./Assignment_openmp $i 1>/dev/null
    cp openmp_results.txt test_output/openmp_${i}.txt
    if [[ $i -le $ITERATIONS ]]; then
        echo -en "${clr_eol}$i/${ITERATIONS} complete\r"
    fi;
done;
echo ""

# Compare the results
TOTALFAILURES=0

# AVX and serial
echo -e ""
echo -e "[${CYAN}INFO${NC}] Results generated. Comparing..."
echo -e ""

echo -e "------------------------"
echo -e "           ${YELLOW}Serial & AVX${NC} "
echo -e "------------------------"
echo -e ""

FAILURES=0
for i in $(seq 4 $ITERATIONS);
do
    if [[ $i -le $ITERATIONS ]]; then
        echo -en "${clr_eol}$FAILURES Failures so far. Comparing avx_${i}.txt and serial_${i}.txt\r"
    fi;
    cmp  test_output/serial_${i}.txt test_output/avx_${i}.txt
    if [[ $? -ne 0 ]]; then
        FAILURES=$((FAILURES+1))
    fi;
done;
echo ""
echo ""

TOTALFAILURES=$FAILURES

# Compare Open MP and serial
echo -e "------------------------"
echo -e "        ${YELLOW}Serial & OpenMP${NC} "
echo -e "------------------------"
echo -e ""

FAILURES=0
for i in $(seq 4 $ITERATIONS);
do
    if [[ $i -le $ITERATIONS ]]; then
        echo -en "${clr_eol}$FAILURES Failures so far. Comparing openmp_${i}.txt and serial_${i}.txt\r"
    fi;
    cmp  test_output/serial_${i}.txt test_output/openmp_${i}.txt
    if [[ $? -ne 0 ]]; then
        FAILURES=$((FAILURES+1))
    fi;
done;
echo ""
echo ""

TOTALFAILURES=$((TOTALFAILURES+FAILURES))

if [[ $TOTALFAILURES -eq 0 ]]; then
    echo -e "${GREEN}ALL TESTS SUCCESSFUL${NC}"
else
    echo -e "${RED}$TOTALFAILURES total failures encountered.${NC}"
fi

echo ""
