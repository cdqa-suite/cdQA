from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering, BertConfig
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME, CONFIG_NAME
from config import *
from processing import read_squad_examples, convert_examples_to_features

if do_train:
    train_examples = read_squad_examples(
        input_file=train_file, is_training=True, version_2_with_negative=version_2_with_negative)
    num_train_optimization_steps = int(
        len(train_examples) / train_batch_size / gradient_accumulation_steps) * num_train_epochs
    if local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

global_step = 0
if do_train:
    cached_train_features_file = train_file+'_{0}_{1}_{2}_{3}'.format(
        list(filter(None, bert_model.split('/'))).pop(), str(max_seq_length), str(doc_stride), str(max_query_length))
    train_features = None
    try:
        with open(cached_train_features_file, "rb") as reader:
            train_features = pickle.load(reader)
    except:
        train_features = convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=True)
        if local_rank == -1 or torch.distributed.get_rank() == 0:
            logger.info("  Saving train features into cached file %s", cached_train_features_file)
            with open(cached_train_features_file, "wb") as writer:
                pickle.dump(train_features, writer)
    logger.info("***** Running training *****")
    logger.info("  Num orig examples = %d", len(train_examples))
    logger.info("  Num split examples = %d", len(train_features))
    logger.info("  Batch size = %d", train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions)
    if local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

    model.train()
    for _ in trange(int(num_train_epochs), desc="Epoch"):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            if n_gpu == 1:
                batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
            input_ids, input_mask, segment_ids, start_positions, end_positions = batch
            loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            if fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                if fp16:
                    # modify learning rate with special warm up BERT uses
                    # if fp16 is False, BertAdam is used and handles this automatically
                    lr_this_step = learning_rate * warmup_linear(global_step/num_train_optimization_steps, warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

if do_train:
    # Save a trained model and the associated configuration
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    with open(output_config_file, 'w') as f:
        f.write(model_to_save.config.to_json_string())

    # Load a trained model and config that you have fine-tuned
    config = BertConfig(output_config_file)
    model = BertForQuestionAnswering(config)
    model.load_state_dict(torch.load(output_model_file))
else:
    model = BertForQuestionAnswering.from_pretrained(bert_model)