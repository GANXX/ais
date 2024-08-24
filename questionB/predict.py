import os
from modelscope import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from datasets import load_dataset
from peft import PeftModel, LoraConfig, TaskType
from transformers import pipeline
import pandas as pd
import json
from config import DATA_SET
cur_dir=os.path.dirname(os.path.realpath(__file__))
import sys

lora_path ="%s/saved_model/%s"%(cur_dir, DATA_SET)
# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(lora_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(lora_path, device_map="auto",torch_dtype=torch.bfloat16)

# 默认参数
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)


# 读取测试数据集
# DATA_PATH = "%s/%s_"%(evaldata_dir, DATA_SET1) #无需修改
# 从指定路径加载JSON格式的数据集，这里使用的是QQP_dev.json文件
# 注意：test文件通常没有标签，因此无法用来评判模型的精度，所以我们一般使用dev.json文件来进行评估
test_data = load_dataset("json", data_files="/home/ganxin/fa/ais/workspace/questionB/pre_data/QQP/QQP_dev.json")
# test文件基本没有标签无法评判模型精度，所以一般使用dev.json

# 开始测试
total_number = len(test_data['train'])
right_number = 0
device = "cuda:0"

# 打开一个文件用于写入结果，文件名包含当前目录和数据集名称
# %s 是字符串格式化的占位符，用于插入变量值
# cur_dir 是当前目录的变量，DATA_SET 是数据集名称的变量
# 'w' 表示以写入模式打开文件，如果文件不存在则创建，如果存在则清空内容
# encoding='utf-8' 指定文件的编码格式为 UTF-8
f_out = open("%s/result_%s.txt" % (cur_dir, DATA_SET), 'w', encoding='utf-8')

for i in range(0, total_number):
    messages = [
        {"role": "user", "content": test_data['train'][i]['instruction']+ " " + test_data['train'][i]['input']} # role不需要调整 
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=64
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print(response)
    if i == total_number-1:
        f_out.write(response[0])
    else:
        f_out.write(response[0]+'\n')
    f_out.flush()

