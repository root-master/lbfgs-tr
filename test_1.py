import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--storage', '-m', default=10, help='The Memory Storage')
parser.add_argument('--mini_batch','-batch', default=1000,help='minibatch size')
parser.add_argument('--num_batch_in_data', '-num-batch',default=5,
        							help='number of batches with overlap')
parser.add_argument('--method', '-method',default='trust-region',
        	help="""Method of optimization ['line-search','trust-region']""")
parser.add_argument(
        '--whole_gradient','-use-whole-data', action='store_true',default=False,
        help='Compute the gradient using all data')

args = parser.parse_args()

minibatch = int(args.mini_batch)
m = int(args.storage)
num_batch_in_data = int(args.num_batch_in_data)
use_whole_data = args.whole_gradient
method = str(args.method)

result_file_path = './results/results_experiment_' + str(method) + '_m_' \
							+ str(m) + '_n_' + str(num_batch_in_data) + '.pkl'
if use_whole_data:
	result_file_path = './results/results_experiment_' + str(method) + '_m_' \
							+ str(m) + '_whole_data' + '.pkl'

print(result_file_path)
