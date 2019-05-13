import math
import copy
import torch
from torch import nn
from util.Bert_utils import BertLayerNorm

def gelu(x):
	return x*0.5*(1.0 + torch.erf(x / math.sqrt(2.0)))
def swish(x):
	return x*torch.sigmoid(x)
ACT2FN = {"gelu":gelu, "relu":torch.nn.functional.relu,"swish":swish}

class BertSelfAttention(nn.Module):
	"""docstring for BertSelfAttention"""
	def __init__(self, config):
		super(BertSelfAttention, self).__init__()
		if config.hidden_size % config.num_attention_heads != 0:
			raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)"%(config.hidden_size,config.num_attention_heads))
		self.num_attention_heads = config.num_attention_heads
		self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size

		self.query = nn.Linear(config.hidden_size,self.all_head_size) #[batch_size,seq_length,all_head_size]
		self.key = nn.Linear(config.hidden_size,self.all_head_size)
		self.value = nn.Linear(config.hidden_size,self.all_head_size)

		self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
	def transpose_for_scores(self,x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads,self.attention_head_size) #[batch_size,seq_length,num_attention_heads,attention_head_size]
		x = x.view(*new_x_shape)
		return x.permute(0,2,1,3) #[batch_size,num_attention_heads,seq_length,attention_head_size]

	def forward(self,hidden_states,attention_mask):
		mixed_query_layer = self.query(hidden_states)
		mixed_key_layer = self.key(hidden_states)
		mixed_value_layer = self.value(hidden_states)
		
		query_layer = self.transpose_for_scores(mixed_query_layer)
		key_layer = self.transpose_for_scores(mixed_key_layer)
		value_layer = self.transpose_for_scores(mixed_value_layer)

		attention_scores = torch.matmul(query_layer,key_layer.transpose(-1,-2))
		attention_scores = attention_scores / math.sqrt(self.attention_head_size)
		attention_scores = attention_scores + attention_mask

		attention_probs = nn.Softmax(dim=-1)(attention_scores)

		attention_probs = self.dropout(attention_probs)

		context_layer = torch.matmul(attention_probs,value_layer)
		context_layer = context_layer.permute(0,2,1,3).contiguous()
		new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
		context_layer = context_layer.view(*new_context_layer_shape)
		return context_layer

class BertSelfOutput(nn.Module):
	"""docstring for BertSelfOutput"""
	def __init__(self, config):
		super(BertSelfOutput, self).__init__()
		self.dense = nn.Linear(config.hidden_size,config.hidden_size)
		self.LayerNorm = BertLayerNorm(config.hidden_size,eps=1e-12)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self,hidden_states,input_tensor):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.LayerNorm(hidden_states+input_tensor)
		return hidden_states
		

class BertAttention(nn.Module):
	"""docstring for BertAttention"""
	def __init__(self, config):
		super(BertAttention, self).__init__()
		self.self = BertSelfAttention(config)
		self.output = BertSelfOutput(config)
	def forward(self,input_tensor,attention_mask):
		self_output = self.self(input_tensor,attention_mask)
		attention_output = self.output(self_output,input_tensor)
		return attention_output

class BertIntermediate(nn.Module):
	"""docstring for BertIntermediate"""
	def __init__(self, config):
		super(BertIntermediate, self).__init__()
		self.dense = nn.Linear(config.hidden_size,config.intermediate_size)
		if isinstance(config.hidden_act,str) or (sys.version_info[0] == 2  and isinstance(config.hidden_act,unicode)):
			self.intermediate_act_fn = ACT2FN[config.hidden_act]
		else:
			self.intermediate_act_fn = config.hidden_act
	def forward(self,hidden_states):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.intermediate_act_fn(hidden_states)
		return hidden_states

class BertOutput(nn.Module):
	"""docstring for BertOutput"""
	def __init__(self, config):
		super(BertOutput, self).__init__()
		self.dense = nn.Linear(config.intermediate_size,config.hidden_size)
		self.LayerNorm = BertLayerNorm(config.hidden_size,eps=1e-12)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
	def forward(self,hidden_states,input_tensor):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.LayerNorm(hidden_states)
		return hidden_states
		
		

class BertLayer(nn.Module):
	def __init__(self,config):
		super(BertLayer,self).__init__()
		self.attention = BertAttention(config) #call
		self.intermediate = BertIntermediate(config) #call
		self.output = BertOutput(config) #call
	def forward(self,hidden_states,attention_mask):
		attention_output = self.attention(hidden_states,attention_mask)
		intermediate_output = self.intermediate(attention_output)
		layer_output = self.output(intermediate_output,attention_output)
		return layer_output




class BertEncoder(nn.Module):
	def __init__(self,config):
		super(BertEncoder,self).__init__()
		layer = BertLayer(config) #call
		self.layer =nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers) ])

	def forward(self,hidden_states,attention_mask,output_all_encoded_layers=True):
		all_encoder_layers = []
		for layer_module in self.layer:
			hidden_states = layer_module(hidden_states,attention_mask) #call
			if output_all_encoded_layers:
				all_encoder_layers.append(hidden_states)
		if not output_all_encoded_layers:
			all_encoder_layers.append(hidden_states)
		return all_encoder_layers
