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
PROMPT_METHOD           = "Standard-Fewshot"    
# PROMPT_METHOD           = "COT"    

# path
# dataset_file_full_path       = "/dss/dsshome1/0A/di35fer/dataset/proofwriter/CWA_depth-0_meta-test.jsonl"
dataset_file_full_path  = f"/dss/dsshome1/0A/di35fer/dataset/{dataset_name}/{data_file_name}"

output_path_suffix      = time.strftime("%m_%d_%H_%M_%S", time.localtime())

#%% paths
path_output_log         = f"/dss/dsshome1/0A/di35fer/code/result/{dataset_name}/Standard_Fewshot_{model_short}_{seed}_{dataset_name}_{data_file_name}_{output_path_suffix}_log.txt"
output_file_path        = f"/dss/dsshome1/0A/di35fer/code/result/{dataset_name}/Standard_Fewshot_{model_short}_{seed}_{dataset_name}_{data_file_name}_{output_path_suffix}_res.csv"

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

def proofwriter_get_prompt(theory, question) -> str:
        # good standard
    prompt = ""
    if PROMPT_METHOD in ["Standard"]:
        prompt = "Task: all of the facts and rules are ture. Based on the provided Facts and Rules, please answer: is the Statement true or false?"
    elif PROMPT_METHOD in ["COT"]:
    # good cot
        prompt = "Task: all of the facts and rules are ture. Based on the provided Facts and Rules, please answer: is the Statement true or false? Let's think step by step."
    prompt += "\nFacts and Rules: " + theory
    prompt += "\nStatements: " + question
    prompt += "\nYour Answer:"
    return prompt

def proofwriter_formulate_theories(theories:str)->str:
    theories_list = [theory.strip() for theory in theories.split('.') if len(theory) > 0]
    theory_formulated = ""
    for i, theory in enumerate(theories_list):
        theory_formulated += f"{str(i)}. {theory}.\n"
    return theory_formulated


def proofwriter_get_prompt_few_shot(theory, question) -> str:
        # good standard
    prompt = F"""Task: all of the facts and rules are ture. Based on the provided Facts and Rules, please answer: is the Statement true or false?
Example Theory: Dave is big. Dave is blue. Dave is furry. Dave is nice. Dave is rough. Dave is round. Dave is white. If Dave is blue and Dave is not furry then Dave is white. If someone is round and not blue then they are nice.
Example Query: Dave is nice.
Example Answer: True
Facts and Rules: {theory}
Statements: {question}
Your Answer:"""
    return prompt

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
    prompt = proofwriter_get_prompt_few_shot(theory, question)

    # print(prompt_question)
    # print("----------")
    answer = ask(prompt)
    # print(answer)
    # print("==========")
    # print()
    
    # formulate result
    lines_answer:list = answer.split('\n')
    dic_answer = {}
    for index_answer_line in range(1, len(lines_answer)+1):
        dic_answer[index_answer_line] = lines_answer[index_answer_line-1]
    dic_meta = {}
    dic_meta.update(data)
    dic_meta.update({
        'output_prompt':prompt,
        'output_answer':answer,
    })
    dic_meta.update(dic_answer)
    df_resut = pd.concat([df_resut, pd.DataFrame([dic_meta])], ignore_index=True)
    if n % 200 == 0:
        logging.debug(f"nr      [{n}]")
        df_resut.to_csv(output_file_path)
    # logging.debug(f"nr      [{n}]")
    # df_resut.to_csv(output_file_path)
df_resut.to_csv(output_file_path)
logging.debug(f"finished")
print("finished")