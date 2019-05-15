import os
import collections
import logging

from file_utils import cached_path

logger = logging.getLogger(__name__)


PRETRAINED_VOCAB_ARCHIVE_MAP = {
	'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
	'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
	'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
	'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
	'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
	'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
	'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
}
PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP = {
	'bert-base-uncased': 512,
	'bert-large-uncased': 512,
	'bert-base-cased': 512,
	'bert-large-cased': 512,
	'bert-base-multilingual-uncased': 512,
	'bert-base-multilingual-cased': 512,
	'bert-base-chinese': 512,
}
VOCAB_NAME = 'vocab.txt'
def load_vocab(vocab_file):
	"""Loads a vocabulary file into a dictionary."""
	vocab = collections.OrderedDict()
	index = 0
	with open(vocab_file,"r",encoding="utf-8") as reader:
		while True:
			token = reader.readline()
			if not token:
				break
			token = token.strip()
			vocab[token] = index
			index += 1
	return vocab
class BertTokenizer(object):
	"""Runs end-to-end tokenization: punctuation splitting + wordpiece"""
	def __init__(self, vocab_file,do_lower_case=True,max_len=None,do_basic_tokenize=True,never_split=("[UNK]","[SEP]","[PAD]","[CLS]","[MASK]")):
		if not os.path.isfile(vocab_file):
			raise ValueError("can't find a vocabulary file at path '{}'.".format(vocab_file))
		self.vocab = load_vocab(vocab_file)
		self.ids_to_tokens = collections.OrderedDict([(ids,tok) for tok,ids in self.vocab.items()])
		self.do_basic_tokenize = do_basic_tokenize
		if do_basic_tokenize:
			self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,never_split=never_split)
		self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
		self.max_len = max_len if max_len is not None else int(1e12)

	@classmethod
	def from_pretrained(cls,pretrained_model_name_or_path,cache_dir=None,*inputs,**kwargs):
		if pretrained_model_name_or_path in PRETRAINED_VOCAB_ARCHIVE_MAP:
			vocab_file = PRETRAINED_VOCAB_ARCHIVE_MAP[pretrained_model_name_or_path]
			if '-cased' in pretrained_model_name_or_path and kwargs.get('do_lower_case',True):
				logger.warning("The pre_trained model you are loading is a cased model but you have not set do_lower_case to False.")
				kwargs['do_lower_case'] = False
			elif '-cased' not in pretrained_model_name_or_path and not kwargs.get('do_lower_case',True):
				logger.warning("The pre-trained model you are loading is an uncased model but you set do_lower_case to False.")
				kwargs['do_lower_case'] = True
		else:
			vocab_file = pretrained_model_name_or_path
		if os.path.isdir(vocab_file):
			vocab_file = os.path.join(vocab_file,VOCAB_NAME)

		# redirect to the cache, if necessary
		try:
			resolved_vocab_file = cached_path(vocab_file,cache_dir=cache_dir) #call
		except EnvironmentError:
			logger.error("Model name '{}' was not found".format(pretrained_model_name_or_path))

		if resolved_vocab_file ==vocab_file:
			logger.info("loading vocabulary file {}".format(vocab_file))
		else:
			logger.info("loading vocabulary file {} from cache at {}".format(vocab_file,resolved_vocab_file))

		if pretrained_model_name_or_path in PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP:
			max_len = PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP[pretrained_model_name_or_path]
			kwargs['max_len'] = min(kwargs.get('max_len',int(1e12)),max_len)

		tokenizer = cls(resolved_vocab_file,*inputs,**kwargs)
		return tokenizer


class BasicTokenizer(object):
	"""Runs basic tokenization (punctuation splitting, lower casing, etc.)."""
	def __init__(self, do_lower_case=True,never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
		self.do_lower_case = do_lower_case
		self.never_split = never_split

		
class WordpieceTokenizer(object):
	"""Runs WordPiece tokenization."""
	def __init__(self, vocab,unk_token="[UNK]",max_input_chars_per_word=100):
		self.vocab = vocab
		self.unk_token = unk_token
		self.max_input_chars_per_word = max_input_chars_per_word

		