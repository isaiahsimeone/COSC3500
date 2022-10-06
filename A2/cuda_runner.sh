CANDIDATE_SIZES=(10 20 40 80 160 320 640 1280 2000 2500 5000 10000 20000 30000)

rm cuda_timings.txt

for size in ${CANDIDATE_SIZES[@]}; do
	(echo "----- size = $size -----" &&
		./Assignment_cuda $size | grep "Time per matrix-vector multiplication") >> cuda_timings.txt
done
