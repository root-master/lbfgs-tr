declare -a m_vec=(5 10 15 20)
declare -a methods=("line-search" "trust-region")

for m in ${m_vec[@]}
do
	for method in ${methods[@]}
		do
			echo $m - $method
			python LBFGS_TR.py -m=$m -use-whole-data -method=$method
			sleep 5
		done
done