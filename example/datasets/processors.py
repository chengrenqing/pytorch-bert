import os
import logging
import csv
import sys

logger = logging.getLogger(__name__)

class InputExample(object):
	"""A single training/test example for simple sequence classification."""
	def __init__(self, guid,text_a,text_b,label=None):
		super(InputExample, self).__init__()
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.label = label
		
class DataProcessor(object):
	"""Base class for data converters for sequence classification data sets."""
	def get_train_examples(self,data_dir):
		"""Gets a collection of `InputExample`s for the train set."""
		raise NotImplementedError()
	def get_dev_examples(self,data_dir):
		"""Gets a collection of `InputExample`s for the dev set."""
		raise NotImplementedError()
	def get_labels(self):
		"""Gets the list of labels for this data set."""
		raise NotImplementedError()
	@classmethod
	def _read_tsv(cls,input_file,quotechar=None):
		"""Reads a tab separated value file."""
		with open(input_file,"r",encoding="utf-8") as f:
			reader = csv.reader(f,delimiter="\t",quotechar=quotechar)
			lines = []
			for line in reader:
				if sys.version_info[0] ==2:
					line = list(unicode(cell,'utf-8') for cell in line)
				lines.append(line)
			return lines

class MrpcProcessor(DataProcessor):
	"""Processor for the MRPC data set (GLUE version)."""
	def get_train_examples(self,data_dir):
		logger.info("LOOKING AT {}".format(os.path.join(data_dir,"train.tsv")))
		return self._create_examples(self._read_tsv(os.path.join(data_dir,"train.tsv")),"train")

	def get_dev_examples(self,data_dir):
		return self._create_examples(self._read_tsv(os.path.join(data_dir,"dev.tsv")),"dev")

	def get_labels(self):
		return ["0","1"]

	def _create_examples(self,lines,set_type):
		 """Creates examples for the training and dev sets."""
		 examples =  []
		 for (i,line) in enumerate(lines):
		 	if i == 0:
		 		continue
		 	guid = "%s-%s"%(set_type,i)
		 	text_a = line[3]
		 	text_b = line[4]
		 	label = line[0]
		 	examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
		 return examples
		 
		


		