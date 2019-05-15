import sys
import os
from BertPreTrainedModel import BertPreTrainedModel
from torch.nn import CrossEntropyLoss
from BertModel import BertModel
from torch import nn
from util import BertConfig

class BertForSequenceClassification(BertPreTrainedModel):
	def __init__(self,config,num_labels):
		super(BertForSequenceClassification,self).__init__(config)
		self.num_labels = num_labels
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config,hidden_size,num_labels)
		self.apply(self.init_bert_weights)

	def forward(self,input_ids,token_type_ids = None,attention_mask=None,labels=None):
		_,pooled_output = self.bert(input_ids,token_type_ids,attention_mask,output_all_encoded_layers=False)
		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)

		if labels is not None:
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(logits.view(-1,self.num_labels),labels.view(-1))
			return loss
		else:
			return logits

if __name__ == '__main__':
	input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
	input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
	token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
	config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
	num_labels = 2
	model = BertForSequenceClassification(config, num_labels)
	logits = model(input_ids, token_type_ids, input_mask)
	print('BertForSequenceClassification test: logits=',logits)
