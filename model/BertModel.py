import torch
from util import BertEmbeddings,BertEncoder,BertPooler,BertConfig
from BertPreTrainedModel import BertPreTrainedModel

class BertModel(BertPreTrainedModel):
	def __init__(self,config):
		super(BertModel,self).__init__(config)
		self.embeddings = BertEmbeddings(config)
		self.encoder = BertEncoder(config)
		self.pooler = BertPooler(config)
		self.apply(self.init_bert_weights)

	def forward(self,input_ids,token_type_ids=None,attention_mask=None,output_all_encoded_layers=True):
		if attention_mask is None:
			attention_mask = torch.one_like(input_ids)
		if token_type_ids is None:
			token_type_ids = torch.zeros_like(input_ids)

		extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) #[batch_size, num_heads, from_seq_length, to_seq_length]

		extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
		extended_attention_mask = (1.0 - extended_attention_mask)* -10000.0

		print("extended_attention_mask",extended_attention_mask.shape)

		embedding_output = self.embeddings(input_ids,token_type_ids)
		encoded_layers = self.encoder(embedding_output,extended_attention_mask,output_all_encoded_layers=output_all_encoded_layers)

		sequence_output = encoded_layers[-1]
		pooled_output = self.pooler(sequence_output)
		if not output_all_encoded_layers:
			encoded_layers = encoded_layers[-1]
		return encoded_layers,pooled_output

if __name__ == "__main__":
	input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
	input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
	token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
	config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
		num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
	model = BertModel(config=config)
	all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)


