import sys
import torch
import random
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, PretrainedConfig

from modules import TransformerForMaskedLanguageModeling, MLP, MoE, MoDE

PATH_TO_TRAIN_DATA = ''

def seed_all(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_gpu_device() -> torch.device:
    if torch.cuda.is_available():
        device_str = "cuda"     # GPU
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_str = "mps"      # Apple silicon
    else:
        print("Warning: No GPU found, using CPU instead.")
        device_str = "cpu"      # CPU
    return torch.device(device_str)

seed_all()

base_config = dict(
    vocab_size=30522, # bert-base-uncased vocab size
    max_position_embeddings=256,
    num_attention_heads=12,
    num_hidden_layers=12,
    hidden_dropout_prob=0.1,
    hidden_size=768,
    intermediate_size=1024,
    bias=True
)

standard_config = PretrainedConfig(
    **base_config,
    ff_cls=MLP,
    mh_moe=False
)

moe_config = PretrainedConfig(
    **base_config,
    num_experts=8,
    capacity_factor=2.0,
    num_experts_per_token=1,
    ff_cls=MoE,
    mh_moe=False
)

mode_config = PretrainedConfig(
    **base_config,
    num_experts=9, # add 1 no-op expert
    capacity_factor=2.0,
    num_experts_per_token=1,
    ff_cls=MoDE,
    mh_moe=False
)

mh_moe_config = PretrainedConfig(
    **base_config,
    num_experts=8,
    capacity_factor=2.0,
    num_experts_per_token=1,
    ff_cls=MoE,
    mh_moe=True
)

mh_mode_config = PretrainedConfig(
    **base_config,
    num_experts=9, # add 1 no-op expert
    capacity_factor=2.0,
    num_experts_per_token=1,
    ff_cls=MoDE,
    mh_moe=True
)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

datasets = load_dataset('text', data_files={'train': PATH_TO_TRAIN_DATA})

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=128, padding='max_length')

tokenized_datasets = datasets.map(tokenize_function, batched=True)

modes = ['standard', 'moe', 'mode', 'mhmoe', 'mhmode']
configs = [standard_config, moe_config, mode_config, mh_moe_config, mh_mode_config]

mod = int(sys.argv[1])
print(modes[mod])

model = TransformerForMaskedLanguageModeling(configs[mod])

training_args = TrainingArguments(
    output_dir=f'./results_{modes[mod]}', 
    evaluation_strategy='epoch',
    learning_rate=5e-5,
    weight_decay=0.01,
    per_device_train_batch_size=128,
    num_train_epochs=1,
    warmup_steps=500,
    logging_dir=f'./logs_{modes[mod]}',
    logging_steps=10,
    save_total_limit=2,
    save_steps=500,
    use_cpu=False,
    fp16=True,
    dataloader_num_workers=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    data_collator=data_collator,
)

trainer.train()

pd.DataFrame(trainer.state.log_history).to_csv(f'loss_{modes[mod]}', index=False, sep='\t')
