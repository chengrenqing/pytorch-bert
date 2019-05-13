import os
from BertPreTrainedModel import BertPreTrainedModel
from BertConfig import BertConfig
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
def main():

	try:
		from pathlib import Path
		PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',Path.home() / '.pytorch_pretrained_bert'))
		print('1',PYTORCH_PRETRAINED_BERT_CACHE)
	except (AttributeError,ImportError):
		PYTORCH_PRETRAINED_BERT_CACHE = os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',os.path.join(os.path.expanduser("~"),'.pytorch_pretrained_bert'))
		print('2',PYTORCH_PRETRAINED_BERT_CACHE)
	print('finished')
	config = BertConfig(vocab_size_or_config_json_file=-1)
	a = BertPreTrainedModel(config)
	bert_model = 'bert-base-uncased'
	cache_dir = ''
	local_rank = -1
	cache_dir = cache_dir if cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(local_rank))
	print(cache_dir)
	model = BertPreTrainedModel.from_pretrained(pretrained_model_name_or_path=bert_model,cache_dir = cache_dir)
if __name__ == "__main__":
	main()