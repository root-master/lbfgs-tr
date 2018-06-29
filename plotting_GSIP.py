import pickle
import matplotlib.pyplot as plt
import numpy as np

results_folder = './results/'
plots_folder = './GSIP-plots/plots/'

# loss_train_results			| 0
# loss_test_results				| 1
# accuracy_train_results		| 2
# accuracy_test_results			| 3
# loop_time 					| 4
# each_iteration_avg_time		| 5


m_vec = [10,20]
n_vec = [2,5,10,25,50,100,200,500,1000]
ov = 1
pr_vec = [0,1]
mode_vec = [1,2,3]

results_dict = {}


def make_result_file_name(m,n,ov,pr,mode):
	result_file_path = './results/results_GSP_' + \
	'_m_' + str(m) + \
	'_n_' + str(n) + \
	'_useoverlap_' + str(ov) + \
	'_precond_' + str(pr) + \
	'_mode_' + str(mode) + '.pkl'
	return result_file_path

def load_pickle(file='file.pkl'):
	with open(file,'rb') as f:  
		loss_train_results, loss_test_results = pickle.load(f)
		accuracy_train_results, accuracy_test_results = pickle.load(f)
		loop_time, each_iteration_avg_time = pickle.load(f)
		return ( loss_train_results, loss_test_results, accuracy_train_results ,\
		accuracy_test_results, loop_time, each_iteration_avg_time )


for m in m_vec:
	for n in n_vec:
		for pr in pr_vec:
			if pr == 0:
				mode = 1
				result_file_path = make_result_file_name(m,n,ov,pr,mode)
				key = 'm_' + str(m) + '_n_' + str(n) + '_pr_' + str(pr) + '_mode_' + str(mode) 
				results = load_pickle(file=result_file_path)
				results_dict[key] = results
			else:
				for mode in mode_vec:
					result_file_path = make_result_file_name(m,n,ov,pr,mode)
					results = load_pickle(file=result_file_path)
					key = 'm_' + str(m) + '_n_' + str(n) + '_pr_' + str(pr) + '_mode_' + str(mode) 
					results_dict[key] = results

# loop time for 300 iterations - fixed on m
loop_time = {}
# plt.figure()
# plt.title('loop time for 300 iterations')
# plt.xlabel('size of multi-batch sample')
# plt.ylabel('time(s)')

for m in m_vec:
	for pr in pr_vec:
		if pr == 0:
			mode = 1
			plot_key = 'm_' + str(m) + '_pr_' + str(pr) + '_mode_' + str(mode)
			loop_time[plot_key] = []
			for n in n_vec:
				key = 'm_' + str(m) + '_n_' + str(n) + '_pr_' + str(pr) + '_mode_' + str(mode) 
				loop_time[plot_key].append(results_dict[key][4])
		else:
			for mode in mode_vec:
				plot_key = 'm_' + str(m) + '_pr_' + str(pr) + '_mode_' + str(mode)
				loop_time[plot_key] = []
				for n in n_vec:
					key = 'm_' + str(m) + '_n_' + str(n) + '_pr_' + str(pr) + '_mode_' + str(mode) 
					loop_time[plot_key].append(results_dict[key][4])
					



# best test accuracy -- fixed on m
max_test_accuracy = {}
# plt.figure()
# plt.title('test accuracy for 300 iterations')
# plt.xlabel('size of multi-batch sample')
# plt.ylabel('accuracy')

for m in m_vec:
	for pr in pr_vec:
		if pr == 0:
			mode = 1
			plot_key = 'm_' + str(m) + '_pr_' + str(pr) + '_mode_' + str(mode)
			max_test_accuracy[plot_key] = []
			for n in n_vec:
				key = 'm_' + str(m) + '_n_' + str(n) + '_pr_' + str(pr) + '_mode_' + str(mode) 
				max_test_accuracy[plot_key].append(max(results_dict[key][3]))
		else:
			for mode in mode_vec:
				plot_key = 'm_' + str(m) + '_pr_' + str(pr) + '_mode_' + str(mode)
				max_test_accuracy[plot_key] = []
				for n in n_vec:
					key = 'm_' + str(m) + '_n_' + str(n) + '_pr_' + str(pr) + '_mode_' + str(mode) 
					max_test_accuracy[plot_key].append(max(results_dict[key][3]))

pr = 1

for m in m_vec:
	plot_key_best = 'best_mode_m_' + str(m)
	mode = 1
	plot_key = 'm_' + str(m) + '_pr_' + str(pr) + '_mode_' + str(mode)
	l1 = max_test_accuracy[plot_key]
	mode = 2
	plot_key = 'm_' + str(m) + '_pr_' + str(pr) + '_mode_' + str(mode)
	l2 = max_test_accuracy[plot_key]
	mode = 3
	plot_key = 'm_' + str(m) + '_pr_' + str(pr) + '_mode_' + str(mode)
	l3 = max_test_accuracy[plot_key]
	max_test_accuracy[plot_key_best] = max(l1,l2,l3)



