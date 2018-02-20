import pickle
import matplotlib.pyplot as plt
import numpy as np

results_dict = {}

def load_pickle(file='file.pkl'):
	with open(result_file_path,'rb') as f:
		loss_train_results, loss_validation_results, loss_test_results = \
																pickle.load(f)
		accuracy_train_results,accuracy_validation_results, \
										accuracy_test_results = pickle.load(f)
		loop_time, each_iteration_avg_time = pickle.load(f)

	return loss_train_results, loss_validation_results, loss_test_results, \
			accuracy_train_results,accuracy_validation_results, \
			accuracy_test_results, loop_time, each_iteration_avg_time


# pickle files folder
results_folder = './results/'

# folder to save plots
plots_folder = './report/plots/'

m_vec = [5,10,15,20]
n_vec = [3,6,12,36,54]
method_vec = ['line-search','trust-region']
for m in m_vec:
	for n in n_vec:
		for method in method_vec:
			result_file_path = results_folder + 'results_experiment_' + \
				str(method) + '_m_'+ str(m) + '_n_' + str(n) + '.pkl'
			results = load_pickle(file=result_file_path)
			key = 'loss_train_' + str(method) + '_m_' + str(m) + '_n_' + str(n) 
			results_dict[key] = results[0]
