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
# if minibatch==500: ==> num_batch_in_data in [3, 6, 9, 12, 18, 36, 54, 108]
# if minibatch==1000 ==> num_batch_in_data in [3, 6, 9, 18, 54]
# if minibatch ==540 ==> num_batch_in_data in [5, 10, 20, 25, 50, 100]
# if minibatch ==1080 ==> num_batch_in_data in [5, 10, 25, 50]
method = str(args.method)
