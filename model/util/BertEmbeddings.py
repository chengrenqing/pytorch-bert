from util.Bert_utils import BertLayerNorm
import torch
from torch import nn

class BertEmbeddings(nn.Module):
	def __init__(self,config):
		super(BertEmbeddings,self).__init__()
		self.word_embeddings = nn.Embedding(config.vocab_size,config.hidden_size,padding_idx=0)
		self.position_embeddings = nn.Embedding(config.max_position_embeddings,config.hidden_size)
		self.token_type_embeddings = nn.Embedding(config.type_vocab_size,config.hidden_size)

		self.LayerNorm = BertLayerNorm(config.hidden_size,eps=1e-12)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self,input_ids,token_type_ids=None):
		seq_length = input_ids.size(1)
		position_ids = torch.arange(seq_length,dtype=torch.long,device=input_ids.device)
		position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
		if token_type_ids is None:
			token_type_ids = torch.zeros_like(input_ids)

		words_embeddings = self.word_embeddings(input_ids)
		position_ids = self.position_embeddings(position_ids)
		token_type_embeddings = self.token_type_embeddings(token_type_ids)

		embeddings = words_embeddings + position_embeddings + token_type_embeddings
		embeddings = self.LayerNorm(embeddings)
		embeddings = self.dropout(embeddings)
		return embeddings