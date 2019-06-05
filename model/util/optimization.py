import sys
import abc
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_
import logging


logger = logging.getLogger(__name__)
if sys.version_info >= (3,4):
	ABC = abc.ABC
else:
	ABC = abc.ABCMeta('ABC',(),{})

class _LRSchedule(ABC):
	"""Parent of all LRSchedules here."""
	warn_t_total = False
	def __init__(self, warmup=0.002, t_total=-1, **kw):
		super(_LRSchedule, self).__init__(**kw)
		if t_total < 0:
			logger.warning("t_total value of {} results in schedule not being applied".format(t_total))
		if not 0.0 <= warmup < 1.0 and not warmup == -1:
			raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
		warmup = max(warmup,0.)
		self.warmup, self.t_total = float(warmup), float(t_total)
		self.warned_for_t_total_at_progress = -1

	def get_lr(self,step,nowarn=False):
		if self.t_total < 0:
			return 1.
		progress = float(step) / self.t_total
		ret = self.get_lr_(progress)
		if not nowarn and self.warn_t_total and progress > 1. and progress > self.warned_for_t_total_at_progress:
			logger.warning("Training beyond specified 't_total'. Learning rate multiplier set to {}. Please set 't_total' of {} correctly." .format(ret, self.__class__.__name__))
			self.warned_for_t_total_at_progress = progress

		return ret
	@abc.abstractmethod
	def get_lr_(self, progress):
		return 1.

class ConstantLR(_LRSchedule):
	def get_lr_(self,progress):
		return 1.

class WarmupLinearSchedule(_LRSchedule):
	warn_t_total = True
	def get_lr_(self,progress):
		if progress < self.warmup:
			return progress/self.warmup
		return max( (progress-1.) / (self.warmup-1.),0. )

SCHEDULES = {
	None: ConstantLR,
	"None": ConstantLR,
	"warmup_linear": WarmupLinearSchedule
}
class BertAdam(Optimizer):
	"""Implements BERT version of Adam algorithm with weight decay fix"""
	def __init__(self, params,lr=required, warmup=-1, t_total=-1, schedule='warmup_linear',b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01, max_grad_norm=1.0, **kwargs):
		if lr is not required and lr < 0.0:
			raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
		if not isinstance(schedule,_LRSchedule) and schedule not in SCHEDULES:
			raise ValueError("Invalid schedule parameter: {}".format(schedule))
		if not 0.0 <= b1 < 1.0:
			raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
		if not 0.0 <= b2 < 1.0:
			raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
		if not e >= 0.0:
			raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))

		if not isinstance(schedule,_LRSchedule):
			schedule_type = SCHEDULES[schedule]
			schedule = schedule_type(warmup=warmup,t_total=t_total)
		else:
			if warmup != -1 or t_total != -1:
				logger.warning("warmup and t_total on the optimizer are ineffective when _LRSchedule object is provided as schedule.")
		defaults = dict(lr=lr,schedule=schedule,b1=b1,b2=b2,e=e,weight_decay=weight_decay,max_grad_norm=max_grad_norm)
		super(BertAdam,self).__init__(params,defaults)
	def get_lr(self):
		lr = []
		for group in self.param_groups:
			for p in group['params']:
				state = self.state[p]
				if len(state) == 0:
					return [0]
				lr_scheduled = group['lr']
				lr_scheduled *= group['schedule'].get_lr(state['step'])
				lr.append(lr_scheduled)
		return lr
	def step(self,closure=None):
		'''Performs a single optimization step.'''
		loss = None
		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			for p in group['params']:
				if p.grad is None:
					continue
				grad = p.grad.data
				if grad.is_sparse:
					raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

				state = self.state[p]

				if len(state) == 0:
					state['step'] = 0:

					state['next_m'] = torch.zeros_like(p.data)
					state['next_v'] = torch.zeros_like(p.data)

				next_m,next_v = state['next_m'],state['next_v']
				beta1, beta2 = group['b1'], group['b2']

				if group['max_grad_norm'] > 0:
					clip_grad_norm_(p,group['max_grad_norm'])

				next_m.mul_(beta1).add_(1-beta1,grad)
				next_v.mul_(beta2).addcmul_(1-beta2,grad,grad)
				update = next_m / (next_v.sqrt()+group['e'])

				if group['weight_decay'] > 0.0:
					update += group['weight_decay']*p.data

				lr_scheduled = group['lr']
				lr_scheduled *= group['schedule'].get_lr(state['step'])

				update_with_lr = lr_scheduled*update
				p.data.add_(-update_with_lr)

				state['step'] += 1
		return loss
		
