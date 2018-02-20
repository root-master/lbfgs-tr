declare -a m_vec=(5 10 15 20)

for m in ${m_vec[@]}
do
	python LBFGS_TR.py -m=$m -use-whole-data -method="trust-region"
	sleep 5
done