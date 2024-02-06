# version: 01.27
from transformers import AutoTokenizer
import transformers
import torch
import json
import pandas as pd
from tqdm import tqdm
import logging
import time
from datasets import load_dataset
from datasets import Dataset
from pathlib import Path
import sys
import getopt
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
#%% arg
seed = 42
info = ""
model_short = "llama-7b-chat-hf"

argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "d:s:m:e:i:",  
            [
                "data_file_name=",          # CWA_depth-0_meta-test.jsonl
                "dataset_name=",            # proofwriter/FOLIO
                "model_short=",             # llama2-13b/llama2-7b
                "seed=",                    # default = 42
                "info=",
            ])
    for opt, arg in opts:
        if opt in ['-d', '--data_file_name']:
            data_file_name = arg            # CWA_depth-0_meta-test.jsonl
        elif opt in ['-s', '--dataset_name']:
            dataset_name = arg              # proofwriter/FOLIO
        elif opt in ['--model_short']:
            model_short = arg 
        elif opt in ['--info']:
            info = arg 
        elif opt in ['--seed']:
            seed = int(arg) 

except:
    print("Error")

if model_short == "llama-7b-chat-hf":
    model   = "meta-llama/Llama-2-7b-chat-hf"
elif model_short == "llama-13b-chat-hf":
    model   = "meta-llama/Llama-2-13b-chat-hf"
elif model_short == "llama-13b-hf":
    model   = "meta-llama/Llama-2-13b-hf"
elif model_short == "code-7b":
    model   = "codellama/CodeLlama-7b-hf"

PROMPT_METHOD           = "COT"    

# path
# dataset_file_full_path       = "/dss/dsshome1/0A/di35fer/dataset/proofwriter/CWA_depth-0_meta-test.jsonl"
dataset_file_full_path  = f"/dss/dsshome1/0A/di35fer/dataset/{dataset_name}/{data_file_name}"

output_path_suffix      = time.strftime("%m_%d_%H_%M_%S", time.localtime())

#%% paths
path_output_log         = f"/dss/dsshome1/0A/di35fer/code/result/COT/{dataset_name}/{PROMPT_METHOD}_{model_short}_{seed}_{dataset_name}_{data_file_name}_{output_path_suffix}_log.txt"
output_file_path        = f"/dss/dsshome1/0A/di35fer/code/result/COT/{dataset_name}/{PROMPT_METHOD}_{model_short}_{seed}_{dataset_name}_{data_file_name}_{output_path_suffix}_res.csv"

# hyperparameters
max_length              = 400
torch_dtype             = torch.float16
isFormatTheory          = False


# Define Logging
logging.basicConfig(level=logging.DEBUG)
file_formatter  = logging.Formatter(fmt="%(asctime)s   %(message)s",
                                    datefmt="%m/%d/%Y %H:%M:%S", )
file_handler    = logging.FileHandler(path_output_log)
file_handler.setFormatter(file_formatter)
logging.root.addHandler(file_handler)

logging.debug(f"**info                              [{info}]")
logging.debug(f"seed                                [{seed}]")
logging.debug(f"model                               [{model}]")
logging.debug(f"model_short                         [{model_short}]")
logging.debug(f"PROMPT_METHOD                       [{PROMPT_METHOD}]")
logging.debug(f"arg data_file_name                  [{data_file_name}]")
logging.debug(f"arg dataset_name                    [{dataset_name}]")
logging.debug(f"max_length                          [{max_length}]")
logging.debug(f"torch_dtype                         [{torch_dtype}]")
logging.debug(f"isFormatTheory                      [{isFormatTheory}]")
logging.debug(f"dataset_file_full_path              [{dataset_file_full_path}]")
logging.debug(f"path_output_log                     [{path_output_log}]")
logging.debug(f"output_file_path                    [{output_file_path}]")


#%% Model
tokenizer = AutoTokenizer.from_pretrained(model)
transformers.set_seed(seed)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch_dtype,
    device_map="auto",
)

#%%

def ask(question:str) -> str:
    sequences = pipeline(
        question,
        do_sample=False,
        # top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=max_length,
    )
    for seq in sequences:
        return seq['generated_text']

def formulate_prompt_conclusion(context:str, query:str)->str:
    prompt = f"""Task please answer if the query can be inferred from the given context:
<Example>
Context: Bob is round. If Bob is round, then Bob is cute. 
Query: Bob is Cute. 
Answer: True, Because Bob is round, and If Bob is round then Bob is cute. So Bob is cute. The query is True.
</Example>
<Context>
Context: {context}
</Context>
<Query>
Query: {query}
</Query>
<Answer>
Answer:
"""
    return prompt

def result_extractor_cot(prompt, result):
    if prompt in result:
        result = result.replace(prompt, "")
    if '</Answer>' in result:
        result = result.split('</Answer>')[0].strip()
    if 'true' in result.lower() and 'false' not in result.lower():
        return 'True', result
    elif 'true' not in result.lower() and 'false' in result.lower():
        return 'False', result
    else:
        return 'Error', result
    
#%% Main
# dataset_folder_path = dataset_folder_path
# paths = [str(x) for x in Path(dataset_folder_path).glob("*.json")]
# for dataset_file_path in paths:
Dataset.cleanup_cache_files
dataset = load_dataset('json', data_files=dataset_file_full_path)

n = 0
df_resut = pd.DataFrame()
df_resut.to_csv(output_file_path)
for data in tqdm(dataset['train']):
    n += 1
    theory      = ""
    question    = ""
    prompt      = ""
    theory = data['theory']
    question = data['question']
    prompt = formulate_prompt_conclusion(theory, question)
    answer = ask(prompt)
    answer_label, answer_reason = result_extractor_cot(prompt=prompt, result=answer)
    # evaluate
    correct = 0
    error   = 0
    if answer_label.lower() == data['answer'].lower():
        correct = 1 
    if answer_label.lower() == 'error':
        error = 1
    # save in file
    dic_answer = {}
    dic_meta = {}
    dic_meta.update(data)
    dic_meta.update({
        'output_prompt'        : prompt,
        'output_answer'        : answer,
        'answer_label'  : answer_label,
        'answer_reason' : answer_reason,
        'correct'       : correct,
        'error'         : error,
    })
    dic_meta.update(dic_answer)

    df_resut = pd.concat([df_resut, pd.DataFrame([dic_meta])], ignore_index=True)
    if n % 200 == 0:
        logging.debug(f"nr      [{n}]")
        df_resut.to_csv(output_file_path)
df_resut.to_csv(output_file_path)
logging.debug(f"finished")
print("finished")