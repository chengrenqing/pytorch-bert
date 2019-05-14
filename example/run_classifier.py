from __future__ import absolute_import
import argparse
import logging
import os
import random
import sys

import numpy as np
import torch

# print(sys.path)
sys.path.append(os.path.abspath('../model/'))
# print(sys.path)
import BertForSequenceClassification
from util import BertConfig,BertTokenizer
from util import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from datasets import MrpcProcessor

logger = logging.getLogger(__name__)

processors = {
	# "cola":ColaProcessor,
	# "mnli":MnliProcessor,
	# "mnli-mm":MnliMismatchedProcessor,
	"mrpc":MrpcProcessor,
	# "sst-2":Sst2Processor,
	# "sts-b":StsProcessor,
	# "qqp":QqpProcessor,
	# "qnli":QnliProcessor,
	# "rte":RteProcessor,
	# "wnli":WnliProcessor,
}

output_modes = {
	"cola":"classification",
	"mnli":"classification",
	"mrpc":"classification",
	"sst-2":"classification",
	"sts-b":"regression",
	"qqp":"classification",
	"qnli":"classification",
	"rte":"classification",
	"wnli":"classification",
}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir",default=None,type=str,required=True,help="The input data dir")
	parser.add_argument("--bert_model",default=None,type=str,required=True,help="Bert ptr-trained model")
	parser.add_argument("--task_name",default=None,type=str,required=True,help="The name of the task to train")
	parser.add_argument("--output_dir",default=None,type=str,required=True,help="The output directory")

	#other parameters
	parser.add_argument("--cache_dir",default="",type=str,help="Where do you want to store the pre-trained models downloaded from s3")
	parser.add_argument("--max_seq_length",default=128,type=int,help="The maximum total input sequence length after WordPiece tokenization.\n")
	parser.add_argument("--do_train",action='store_true',help="Where to run training.")
	parser.add_argument("--do_eval",action='store_true',help="Where to run eval on the dev set")
	parser.add_argument("--do_lower_case",action='store_true',help="Set this flag if you are using an uncased model")
	parser.add_argument("--train_batch_size",default=32,type=int,help="Total batch size for training")
	parser.add_argument("--eval_batch_size",default=8,type=int,help="Total batch size for eval")
	parser.add_argument("--learning_rate",default=5e-5,type=float,help="The initial learning rate for Adam.")
	parser.add_argument("--num_train_epochs",default=3.0,type=float,help="Total number of training epochs to perform.")
	parser.add_argument("--warmup_proportion",default=0.1,type=float,help="Proportion of traning to perform linear learning rate warmup for.")
	parser.add_argument("--no_cuda",action='store_true',help="Whether not to use CUDA when available.")
	parser.add_argument("--local_rank",type=int,default=-1,help="local_rank for distributed training on gpus")
	parser.add_argument("--seed",type=int,default=42,help="random seed for initialization")
	parser.add_argument("--gradient_accumulation_steps",type=int,default=1,help="Number of updates steps to accumulate before performing a backward/update pass")
	parser.add_argument("--fp16",action='store_true',help="Whether to use 16-bit float precision instead of 32-bit")
	parser.add_argument("--loss_scale",type=float,default=0,help="Loss scaling to improve fp16 numeric stability.Only used when fp16 set to True.\n")
	parser.add_argument("--server_ip",type=str,default='',help="Can be used for distant debugging.")
	parser.add_argument("--server_port",type=str,default='',help="Can be used for distant debugging.")

	args = parser.parse_args()

	if args.server_ip and args.server_port:
		import ptvsd
		print("Waiting for debugger attach")
		ptvsd.enable_attach(address=(args.server_ip,args.server_port),redirect_output=True)
		ptvsd.wait_for_attach()

	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		n_gpu = torch.cuda.device_count()
	else:
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda",args.local_rank)
		n_gpu = 1
		torch.distributed.init_process_group(backend='nccl')

	logging.basicConfig(format = '%(asctime)s - %(levelname)s -%(name)s - %(message)s',datefmt = '%m/%d/%Y %H:%M:%S',level=logging.INFO if args.local_rank in [-1,0] else logging.WARN)
	logger.info("device:{} n_gpu:{},distributed training:{},16-bits training:{}".format(device,n_gpu,bool(args.local_rank != -1),args.fp16))
	if args.gradient_accumulation_steps < 1:
		raise ValueError("Invalid gradient_accumulation_steps parameter:{},should be >=1".format(args.gradient_accumulation_steps))
	args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed) #The initialization of the parameters is random, set seed for multi gpu in order to make the results consistent each time.

	if not args.do_train and not args.do_eval:
		raise ValueError("At least one of 'do_train' or 'do_eval' must be True.")

	if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
		raise ValueError("Output directory({}) already exists and is not empty.".format(args.output_dir))
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	task_name = args.task_name.lower()

	if task_name not in processors:
		raise ValueError("Task not found:%s"%(task_name))

	processor = processors[task_name]()
	output_mode = output_modes[task_name]

	label_list = processor.get_labels()
	num_labels = len(label_list)

	tokenization = BertTokenizer.from_pretrained(args.bert_model,do_lower_case=args.do_lower_case) #call

	train_examples = None
	num_train_optimization_steps = None
	if args.do_train:
		train_examples = processor.get_train_examples(args.data_dir)
		num_train_optimization_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

		if args.local_rank != -1:
			num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

	#prepare model
	cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),'distributed_{}'.format(args.local_rank))
	model = BertForSequenceClassification.from_pretrained(args.bert_model,cache_dir=cache_dir,num_labels=num_labels)#call

	if args.fp16:
		model.half()
	model.to(device)
	if args.local_rank != -1:
		try:
			from apex.parallel import DistributedDataParallel as DDP
		except ImportError:
			raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

		model = DDP(model)

	elif n_gpu > 1:
		model = torch.nn.DataParallel(model)

	# Prepare optimizer
	if args.do_train:
		param_optimizer = list(model.named_parameters)





if __name__ == "__main__":
	main()
