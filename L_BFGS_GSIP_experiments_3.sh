#! /bin/bash
#$ -S /bin/bash
#$ -q gpu.q
#$ -cwd
#$ -N myjob
#$ -j y
#$ -o GlobalSIP.qlog
#$ -l mem_free=64G
#$ -pe smp 20
#$ -V

export CUDA_HOME=/usr/local/cuda:/usr/local/cuda-8.0:/home/jrafatiheravi/src/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

module load anaconda3
source activate jacobenv
cd ~
cd lbfgs-tr

declare -a m_vec=(10 20)
declare -a n_vec=(2 5 10 25 50 100)

for m in ${m_vec[@]}
do
	for n in ${n_vec[@]}
	do
		echo $m - $n - "preconditioning"
		python L_BFGS_TR_modified.py -m=$m -num-batch=$n -use-overlap -preconditioning -pre-cond-mode=3
	done
done

