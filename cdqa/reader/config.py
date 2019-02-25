import os
import sys
import logging
import random
import numpy as np
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

bert_model=''
output_dir=''
train_file=''
predict_file=''
max_seq_length=384
doc_stride=128
max_query_length=64
train_batch_size=32
predict_batch_size=8
learning_rate=5e-5
num_train_epochs=3.0
warmup_proportion=0.1
n_best_size=20
max_answer_length=30
verbose_logging=False
no_cuda=False
seed=42
gradient_accumulation_steps=1
do_lower_case=True
local_rank=-1
fp16=True
loss_scale=0
version_2_with_negative=False
null_score_diff_threshold=0.0

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

if local_rank == -1 or no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    device, n_gpu, bool(local_rank != -1), fp16))

if gradient_accumulation_steps < 1:
    raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                        gradient_accumulation_steps))

train_batch_size = train_batch_size // gradient_accumulation_steps

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

if os.path.exists(output_dir) and os.listdir(output_dir) and do_train:
    raise ValueError("Output directory () already exists and is not empty.")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

train_examples = None
num_train_optimization_steps = None

# Prepare model
model = BertForQuestionAnswering.from_pretrained(bert_model,
            cache_dir=os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(local_rank)))

if fp16:
    model.half()
model.to(device)
if local_rank != -1:
    try:
        from apex.parallel import DistributedDataParallel as DDP
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    model = DDP(model)
elif n_gpu > 1:
    model = torch.nn.DataParallel(model)

# Prepare optimizer
param_optimizer = list(model.named_parameters())

# hack to remove pooler, which is not used
# thus it produce None grad that break apex
param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

if fp16:
    try:
        from apex.optimizers import FP16_Optimizer
        from apex.optimizers import FusedAdam
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    optimizer = FusedAdam(optimizer_grouped_parameters,
                            lr=learning_rate,
                            bias_correction=False,
                            max_grad_norm=1.0)
    if loss_scale == 0:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    else:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=loss_scale)
else:
    optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=learning_rate,
                            warmup=warmup_proportion,
                            t_total=num_train_optimization_steps)