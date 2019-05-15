import logging
import torch
from torch import nn
from util import BertConfig,BertLayerNorm


logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_AMP = {
	'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
	'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
	'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
	'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
	'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
	'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
	'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz"
}
class BertPreTrainedModel(nn.Module):
	def __init__(self,config,*input,**kwargs):
		super(BertPreTrainedModel,self).__init__()
		if not isinstance(config,BertConfig):
			raise ValueError('BertPreTrainedModel config')
		self.config = config
	def init_bert_weights(self,module):
		'''init the weights
		'''
		if isinstance(module,(nn.Linear,nn.Embedding)):
			module.weight.data.normal_(mean=0.0,std=self.config.initializer_range)
		elif isinstance(module,BertLayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)
		if isinstance(module,nn.Linear) and module.bias is not None:
			module.bias.data.zero_()

	@classmethod
	def from_pretrained(cls,pretrained_model_name_or_path,*input,**kwargs):
		state_dict = kwargs.get('state_dict',None)
		kwargs.pop('state_dict',None)
		cache_dir = kwargs.get('cache_dir',None)
		kwargs.pop('cache_dir',None)
		from_tf = kwargs.get('from_tf',None)
		kwargs.pop('from_tf',None)

		if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_AMP:
			archive_file = PRETRAINED_MODEL_ARCHIVE_AMP[pretrained_model_name_or_path]
		else:
			archive_file = pretrained_model_name_or_path

		try:
			resolved_archive_file = cached_path(archive_file,cache_dir=cache_dir) #call
		except EnvironmentError:
			logger.error("Model name is invaid")
			return None

		if resolved_archive_file == archive_file:
			logger.info("loading archive file {}".format(archive_file))
		else:
			logger.info("loading archive file {} from cache at {}".format(archive_file,resolved_archive_file))

		tempdir = None
		if os.path.isdir(resolved_archive_file) or from_tf:
			serialization_dir = resolved_archive_file
		else:
			tempdir = tempfile.mkdtemp()
			logger.info("extracting archive file {} to temp dir {}".format(resolved_archive_file,tempdir))
			with tarfile.open(resolved_archive_file,'r:gz') as archive:
				archive.extractall(tempdir)

		#load config
		config_file = os.path.join(serialization_dir,CONFIG_NAME)
		if not os.path.exists(config_file):
			config_file = os.path.join(serialization_dir,BERT_CONFIG_NAME)

		config = BertConfig.from_json_file(config_file) #call
		logger.info("Model config {}".format(config))

		#init model

		model = cls(config,*inputs,**kwargs)
		if state_dict is None and not from_tf:
			weights_path = os.path.join(serialization_dir,WEIGHTS_NAME)
			state_dict = torch.load(weights_path,map_location='cpu')
		return model



