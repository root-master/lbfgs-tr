import pickle
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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


# loss_train_results			| 0
# loss_validation_results		| 1
# loss_test_results				| 2
# accuracy_train_results		| 3
# accuracy_validation_results	| 4
# accuracy_test_results			| 5
# loop_time 					| 6
# each_iteration_avg_time		| 7

# pickle files folder
results_folder = './results/'

# folder to save plots
plots_folder = './report/plots/'

m_vec = [5,10,15,20]
n_vec = [2,3,6,12,36,54] # 2 stands for using whole data
method_vec = ['line-search','trust-region']
for m in m_vec:
	for n in n_vec:
		for method in method_vec:
			result_file_path = results_folder + 'results_experiment_' + \
				str(method) + '_m_'+ str(m) + '_n_' + str(n) + '.pkl'
			results = load_pickle(file=result_file_path)
			key = 'results_' + str(method) + '_m_' + str(m) + '_n_' + str(n) 
			results_dict[key] = results


# loop time for 200 iterations - fixed on m
loop_time = {}
plt.figure()
plt.title('loop time for 200 iterations')
plt.xlabel('number of multi batch')
plt.ylabel('time(s)')
for m in m_vec:
	for method in method_vec:
		key = 'loop_time_' + str(method) + '_m_' + str(m) 
		loop_time[key] = []
		for n in n_vec:
			key_result = 'results_' + str(method) + '_m_' + str(m) + '_n_' + str(n)
			loop_time[key].append(results_dict[key_result][6])
		legend = str(method) + ' $m =$' + str(m)
		plt.plot(n_vec, loop_time[key],'-*',label=legend,markersize=10)
		plt.legend(loc=1)
	

# max train accuracy in 200 iterations - fixed on m
max_accuracy_train = {}
plt.figure()
plt.title('maximum training accuracy for 200 iterations')
plt.xlabel('number of multi batch')
plt.ylabel('accuracy')
for m in m_vec:
	for method in method_vec:
		key = 'max_accuracy_train_' + str(method) + '_m_' + str(m) 
		max_accuracy_train[key] = []
		for n in n_vec:
			key_result = 'results_' + str(method) + '_m_' + str(m) + '_n_' + str(n)
			max_accuracy_train[key].append(max(results_dict[key_result][3]))
		legend = str(method) + ' $m =$' + str(m)
		plt.plot(n_vec, max_accuracy_train[key],'-*',label=legend,markersize=10)
		plt.legend(loc=1)


# max test accuracy in 200 iterations - fixed on m
max_accuracy_test = {}
plt.figure()
plt.title('maximum test accuracy for 200 iterations')
plt.xlabel('number of multi batch')
plt.ylabel('accuracy')
for m in m_vec:
	for method in method_vec:
		key = 'max_accuracy_train_' + str(method) + '_m_' + str(m) 
		max_accuracy_test[key] = []
		for n in n_vec:
			key_result = 'results_' + str(method) + '_m_' + str(m) + '_n_' + str(n)
			max_accuracy_test[key].append(max(results_dict[key_result][5]))
		legend = str(method) + ' $m =$' + str(m)
		plt.plot(n_vec, max_accuracy_test[key],'-*',label=legend,markersize=10)
		plt.legend(loc=1)


# min training loss in 200 iterations - fixed on m
min_train_loss = {}
plt.figure()
plt.title('minimum training loss for 200 iterations')
plt.xlabel('number of multi batch')
plt.ylabel('loss')

for m in m_vec:
	for method in method_vec:
		key = 'max_accuracy_train_' + str(method) + '_m_' + str(m) 
		min_train_loss[key] = []
		for n in n_vec:
			key_result = 'results_' + str(method) + '_m_' + str(m) + '_n_' + str(n)
			min_train_loss[key].append(min(results_dict[key_result][0]))
		legend = str(method) + ' $m =$' + str(m)
		plt.plot(n_vec, min_train_loss[key],'-*',label=legend,markersize=10)
		plt.legend(loc=1)


# min test loss in 200 iterations - fixed on m
min_test_loss = {}
plt.figure()
plt.title('minimum test loss for 200 iterations')
plt.xlabel('number of multi batch')
plt.ylabel('loss')

for m in m_vec:
	for method in method_vec:
		key = 'max_accuracy_train_' + str(method) + '_m_' + str(m) 
		min_test_loss[key] = []
		for n in n_vec:
			key_result = 'results_' + str(method) + '_m_' + str(m) + '_n_' + str(n)
			min_test_loss[key].append(min(results_dict[key_result][2]))
		legend = str(method) + ' $m =$' + str(m)
		plt.plot(n_vec, min_test_loss[key],'-*',label=legend,markersize=10)
		plt.legend(loc=1)


performance_results_dict = {}
m_vec = [15,20]
n_vec = [2,4] # 2 stands for using whole data
method_vec = ['line-search','trust-region']
for m in m_vec:
	for n in n_vec:
		for method in method_vec:
			result_file_path = results_folder + 'results_experiment_FEB_20_' + \
				str(method) + '_m_'+ str(m) + '_n_' + str(n) + '.pkl'
			results = load_pickle(file=result_file_path)
			key = 'results_' + str(method) + '_m_' + str(m) + '_n_' + str(n) 
			performance_results_dict[key] = results



for m in m_vec:	
	for n in n_vec:
		plt.figure()		
		for method in method_vec:
			key_result = 'results_' + str(method) + '_m_' + str(m) + '_n_' + str(n)
			loss_train = performance_results_dict[key_result][0]
			loss_test  = performance_results_dict[key_result][2]
			x_vec = range(len(loss_train)) 
			legend_1 = str(method) + ' $m =$' + str(m) + ' -- '+ '$n =$' \
					+ str(n) + ' -- (train loss)'
			legend_2 = str(method) + ' $m =$' + str(m) + ' -- '+ '$n =$' \
					+ str(n) + ' -- (test loss) '
			plt.plot(x_vec,loss_train,label=legend_1)
			plt.plot(x_vec,loss_test,label=legend_2)
			plt.xlabel('iterations')
			plt.ylabel('loss')
			plt.legend(loc=1)
			plot_path = plots_folder + 'performance_' + '_loss_' + '_m_' \
											+ str(m) + '_n_' + str(n) + '.eps'
			plt.savefig(plot_path, format='eps', dpi=1000)


for m in m_vec:	
	for n in n_vec:
		plt.figure()		
		for method in method_vec:
			key_result = 'results_' + str(method) + '_m_' + str(m) + '_n_' + str(n)
			accuracy_train = performance_results_dict[key_result][3]
			accuracy_test  = performance_results_dict[key_result][5]
			x_vec = range(len(accuracy_train)) 
			legend_1 = str(method) + ' $m =$' + str(m) + ' -- '+ '$n =$' \
					+ str(n) + ' -- (train accuracy)'
			legend_2 = str(method) + ' $m =$' + str(m) + ' -- '+ '$n =$' \
					+ str(n) + ' -- (test accuracy) '
			plt.plot(x_vec,accuracy_train,label=legend_1)
			plt.plot(x_vec,accuracy_test,label=legend_2)
			plt.xlabel('iterations')
			plt.ylabel('accuracy')
			plt.legend(loc=4)
			plot_path = plots_folder + 'performance_' + '_accuracy_' + '_m_' \
											+ str(m) + '_n_' + str(n) + '.eps'
			plt.savefig(plot_path, format='eps', dpi=1000)
















