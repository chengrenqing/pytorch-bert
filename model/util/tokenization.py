import os
import collections
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
	
		