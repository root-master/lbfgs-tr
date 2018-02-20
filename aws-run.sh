declare -a m_vec=(5 10 15 20)
declare -a n_vec=(3 6 12 36 54)
declare -a methods=("line-search" "trust-region")

for m in ${m_vec[@]}
do
	for n in ${n_vec[@]}
		do
			for method in ${methods[@]}
				do
					echo $m - $n - $method
					#python LBFGS_TR.py -m=$m -num-batch=$n -method=$method >> output.txt
					#sleep 5
				done
		done
done