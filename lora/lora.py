import settings as settings
import os, torch, logging
from datasets import Dataset, load_dataset, load_from_disk
from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

from tinystories_preprocess import load_stories 

#model_name = settings.OUTPUT_DIR
model_name = "results"
#dataset_name = settings.dataset_name
#dataset_path = "./botany-conv-formatted.txt"
#dataset_path = "../tinyteacher/tinyteach-blank/tinyteach-blank1.txt"
dataset_path = "./unknown-conv.txt"
new_model = "lora-model"

lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

output_dir = "./lora-results"
num_train_epochs = 1

fp16 = False
bf16 = False

per_device_train_batch_size = 4
per_device_eval_batch_size = 1

gradient_accumulation_steps = 1
gradient_checkpointing = True

max_grad_norm = 0.3
learning_rate = 2e-4

weight_decay = 0.001

optim = "paged_adamw_32bit"

lr_scheduler_type = "cosine"

max_steps = -1

warmup_ratio = 0.03

group_by_length = True

save_steps = 0

logging_steps = 25

max_seq_length = settings.MAX_LENGTH
packing = False
device_map = {"": 0}

#dataset = load_dataset(dataset_name, split="train")
#dataset = load_from_disk(dataset_name)["train"]
#dataset = load_dataset(dataset_name, split="train")
stories = list(load_stories(dataset_path))
training = [stories[i] + stories[(i+1)%len(stories)] for i in range(0, len(stories), 2)]
valid_size = -(-len(training)) // settings.VALIDATION_FRACTION
training = training[valid_size:]
dataset = Dataset.from_list([{'text': text } for text in stories ])
print(dataset)
#print(dataset['input_ids'])

#dataset_name = "mlabonne/guanaco-llama2-1k"
#dataset = load_dataset(dataset_name, split="train")
#print(dataset)
#print(dataset['text'])

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant
)

if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

model = LlamaForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map='auto'
)
print(model)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(settings.tokenizer, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)

training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing
)

trainer.train()
trainer.model.save_pretrained(new_model)

prompt = "= What is the Big Bang?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"{prompt}")
print(result[0]['generated_text'])
