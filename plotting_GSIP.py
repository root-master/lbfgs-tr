import pickle
import matplotlib.pyplot as plt
import numpy as np

results_folder = './results/'

m = 20
num_batch_in_data = 100
ov = 1
pr = 1
pre_cond_mode = 1

result_file_path = './results/results_GSP_' + \
				'_m_' + str(m) + \
				'_n_' + str(num_batch_in_data) + \
				'_useoverlap_' + str(ov) + \
				'_precond_' + str(pr) + \
				'_mode_' + str(pre_cond_mode) + '.pkl'


with open(result_file_path,'rb') as f:  
	loss_train_results, loss_test_results = pickle.load(f)
	accuracy_train_results, accuracy_test_results = pickle.load(f)
	loop_time, each_iteration_avg_time = pickle.load(f)
