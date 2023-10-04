#!/usr/bin/env python
# coding: utf-8


# accelerate launch --config_file  deepspeed_zs3.json Finetune_t5_large_peft-practice.py
# accelerate launch --config_file deepspeed_zs3.json --mixed_precision bf16 --gradient_accumulation_steps 1 --gradient_clipping 1.0 Finetune_t5_large_peft-practice.py 
from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"gpu memory: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

print_gpu_utilization()


############### 
import numpy as np

def levenshtein_distance(str1, str2):
    # TC: O(N^2)
    # SC: O(N^2)
    if str1 == str2:
        return 0
    num_rows = len(str1) + 1
    num_cols = len(str2) + 1
    dp_matrix = np.empty((num_rows, num_cols))
    dp_matrix[0, :] = range(num_cols)
    dp_matrix[:, 0] = range(num_rows)

    for i in range(1, num_rows):
        for j in range(1, num_cols):
            if str1[i - 1] == str2[j - 1]:
                dp_matrix[i, j] = dp_matrix[i - 1, j - 1]
            else:
                dp_matrix[i, j] = min(dp_matrix[i - 1, j - 1], dp_matrix[i - 1, j], dp_matrix[i, j - 1]) + 1

    return dp_matrix[num_rows - 1, num_cols - 1]


def get_closest_label(eval_pred, classes):
    min_id = sys.maxsize
    min_edit_distance = sys.maxsize
    for i, class_label in enumerate(classes):
        edit_distance = levenshtein_distance(eval_pred.strip(), class_label)
        if edit_distance < min_edit_distance:
            min_id = i
            min_edit_distance = edit_distance
    return classes[min_id]

###############

# In[2]:


import sys
import importlib
import os
# for module in sys.modules.values():
#     print(module)
#     importlib.reload(module)
# importlib.reload(transformers

from accelerate.utils import write_basic_config

# write_basic_config()  # Write a config file
# os._exit(00)  # Restart the notebook


# In[3]:


TRACKING_URI=os.getenv('TRACKING_URI', 'http://192.168.10.38:5000')
# num_epochs = 3
training_set_file = 'training_set.xlsx'
# training_set_file = '1.xlsx'


# # Model

# In[4]:


from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
import torch
from datasets import load_dataset
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup, get_scheduler
from tqdm import tqdm
from datasets import load_dataset

device = "cuda"
model_name_or_path = 'agemagician/mlong-t5-tglobal-large'  #"bigscience/mt0-large"
tokenizer_name_or_path = 'agemagician/mlong-t5-tglobal-large' # "bigscience/mt0-large"

checkpoint_name = "test_analysis_lora_v1.pt"
text_column = "text"
label_column = "path"
max_length = 4096 #8192 #16384

lr = 1e-3
weight_decay = 0.0

num_epochs = 3
batch_size = 2 #8


# In[5]:


# creating model
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, 
    target_modules=["q", "v"],
    inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# model = model.half()


# # Dataset

# In[6]:


import pandas as pd

training_columns=['path', 'text']
df = pd.read_excel(training_set_file, index_col=0)

# fix historical naming
df.rename(columns={'clear_text':'text'}, inplace=True)
df.head()


# In[7]:


t_len = df['text'].map(len).max()
print(f'{t_len=}')
print(f'upper bound estimate in word pieces: {int(t_len/3)}')


# In[8]:


cc = 0
for row in df.itertuples():
    if not isinstance(row.original_text, str):
        df.at[row.Index, 'original_text'] = row.text
        cc+=1
cc


# In[9]:


from pathlib import Path


def fix_path_for_type(x):
    pp = Path(x.replace('\\','/'))
    if len(pp.parts) > 1:
        return pp.parent.parts[2]
    return x

fix_path_for_type('ee')

x = df['path'].transform(lambda x : fix_path_for_type(x))
sorted(list(x.unique()))


# In[10]:


df = df[training_columns]
df.head()


# In[11]:


# df['path'] = df['path'].transform(lambda x: Path(x.replace('\\','/')).parent.parts[-1])
df['path'] = df['path'].transform(lambda x: fix_path_for_type(x))

# df.head()


# In[12]:


df = df.replace({
    'path': { 
        'Практика правовой поддежрки брендов' : 'Практика правовой поддержки брендов',
        'Практика прововой поддержки брендов' : 'Практика правовой поддержки брендов',
        'Практика экспорта, оптовых продаж и сбыта бизнес-едициниц' : 'Практика экспорта, оптовых продаж и сбыта бизнес-единиц',
        'Практика правового сопровожденияия операционной деятельности БРД' : 'Практика правового сопровождения операционной деятельности БРД',
        'Практика экспорта, оптовых продаж и сбыта бизнес-едениц' : 'Практика экспорта, оптовых продаж и сбыта бизнес-единиц',
        
        'Практика судедбной защиты' : 'Практика судебной защиты',
    }
})


# In[13]:


x = df['path']
sorted(list(x.unique()))


# In[14]:


df = df[df['path'] != 'Практика судебной защиты']


# In[15]:

# experimental:
df['path'] = df['path'].astype(str) + '<\s>'


categories = df['path'].unique()
print(f'sorted categories: {sorted(categories)}')




# In[16]:


from datasets import Dataset, ClassLabel, Features
categories_feature=ClassLabel(names=categories.tolist())
features = Features({'path': categories_feature})

ds = Dataset.from_pandas(df)
ds


# In[17]:


ds = ds.train_test_split(test_size=0.09)


# In[18]:


ds


# In[19]:


# loading dataset
# dataset = load_dataset("financial_phrasebank", "sentences_allagree")
# dataset = dataset["train"].train_test_split(test_size=0.1)
dataset = ds
dataset["validation"] = dataset["test"]
# del dataset["test"]

# classes = dataset["train"].features["label"].names
# dataset = dataset.map(
#     lambda x: {"text_label": [classes[label] for label in x["label"]]},
#     batched=True,
#     num_proc=1,
# )

# dataset["train"][0]


# # Data preprocessing

# In[20]:


# data preprocessing
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[label_column]
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length=20, padding="max_length", truncation=True, return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["validation"]

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)


# # Accelerate config


from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.state import AcceleratorState
from accelerate.logging import get_logger
from accelerate.utils import DummyOptim, DummyScheduler, set_seed

import datasets
import transformers
import os, math

os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
print_gpu_utilization()

def training_function(model):
    global train_dataloader, optimizer, eval_dataloader
    
    # deepspeed_plugin = DeepSpeedPlugin(
    #     # hf_ds_config='/root/.cache/huggingface/accelerate/default_config.yaml'
    #   gradient_accumulation_steps=1,
    #   gradient_clipping=1.0,
    #   offload_optimizer_device='cpu',
    #   offload_param_device='cpu',
    #   zero3_init_flag=True,
    #   zero3_save_16bit_model=True,
    #   zero_stage=3,    
    # )
    # accelerator = Accelerator(mixed_precision='fp16', deepspeed_plugin=deepspeed_plugin,
    #                           # distributed_type='DEEPSPEED'
    #                          )

    # accelerator.distributed_type = 'DEEPSPEED'
  
    accelerator = Accelerator(
        # mixed_precision='fp16'
    )
    accelerator.print(f"{accelerator.state}")
    accelerator.print(accelerator.distributed_type)
    accelerator.print(accelerator.state.deepspeed_plugin)

    device = accelerator.device
    
    # To have only one message (and not 8) per logs of Transformers or Datasets, we set the logging verbosity
    # to INFO for the main process only.
    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()   

    # optimizer and lr scheduler
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Optimizerf
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # New Code #
    # Creates Dummy Optimizer if `optimizer` was spcified in the config file else creates Adam Optimizer
    optimizer_cls = (
        torch.optim.Adam
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(optimizer_grouped_parameters, lr=lr)

    
    # parameters 
    gradient_accumulation_steps = 1
    num_warmup_steps = 0
    lr_scheduler_type = 'linear'
    output_dir = 'models'

    
    # lr_scheduler = get_linear_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=(len(train_dataloader) * num_epochs),
    # )
    # New Code
    # Get gradient accumulation steps from deepspeed config if available
    if accelerator.state.deepspeed_plugin is not None:
        gradient_accumulation_steps = accelerator.state.deepspeed_plugin.deepspeed_config[
            "gradient_accumulation_steps"
        ]

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_epochs * num_update_steps_per_epoch
    

    # New Code #
    # Creates Dummy Scheduler if `scheduler` was spcified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=max_train_steps, warmup_num_steps=num_warmup_steps
        )
    
    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    
    # training and evaluation
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            
            # batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            
            # loss.backward()
            accelerator.backward(loss)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        print_gpu_utilization()
        
        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
            )
    
        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

        # print accuracy
        correct = 0
        total = 0
        for pred, true in zip(eval_preds, dataset["validation"]["path"]):
            pred_name = get_closest_label(pred.strip(), categories)
            # if pred.strip() == true.strip():
            if pred_name == true.strip():
                correct += 1
            total += 1
        accuracy = correct / total * 100
        print(f"{accuracy=} % on the evaluation dataset")
        print(f"{eval_preds[:10]=}")
        print(f"{dataset['validation']['path'][:10]=}")


    unwrapped_model = accelerator.unwrap_model(model)

    # New Code #
    # Saves the whole/unpartitioned fp16 model when in ZeRO Stage-3 to the output directory if
    # `stage3_gather_16bit_weights_on_model_save` is True in DeepSpeed Config file or
    # `zero3_save_16bit_model` is True in DeepSpeed Plugin.
    # For Zero Stages 1 and 2, models are saved as usual in the output directory.
    # The model name saved is `pytorch_model.bin`
    unwrapped_model.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )


# # Train


# from accelerate import notebook_launcher

# print_gpu_utilization()
# notebook_launcher(training_function, (model,), 
#                   num_processes=1, mixed_precision='fp16',
#                  )

training_function(model)



