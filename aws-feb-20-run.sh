declare -a m_vec=(15 20)
declare -a methods=("line-search" "trust-region")

for m in ${m_vec[@]}
do
	for method in ${methods[@]}
		do
			python LBFGS_TR.py -m=$m -num-batch=4 -method=$method -minibatch=500
			sleep 5
		done
done

for m in ${m_vec[@]}
do
	for method in ${methods[@]}
		do
			python LBFGS_TR.py -m=$m -use-whole-data -method=$method -minibatch=500
			sleep 5
		done
done