# version: 1.8
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
import code
import code_ablation
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
#%% arg
seed = 42
info = ""
num_example = '0'

argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "d:s:m:e:i:n:",  
            [
                "data_file_name=",          # CWA_depth-0_meta-test.jsonl
                "dataset_name=",            # proofwriter/FOLIO
                "model_short=",             # llama2-13b/llama2-7b
                "seed=",                    # default = 42
                "info=",
                "num_example=",
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
        elif opt in ['--num_example']:
            num_example = arg
except:
    print("Error")

#%% parameter
if model_short == "llama2-7b":
    model                   = "meta-llama/Llama-2-7b-chat-hf"
elif model_short == "llama2-13b":
    model = "meta-llama/Llama-2-13b-chat-hf"
PROMPT_METHOD           = "Vanilla"    

#%% path
# dataset_file_full_path       = "/dss/dsshome1/0A/di35fer/dataset/proofwriter/CWA_depth-0_meta-test.jsonl"
# dataset_file_full_path  = f"/dss/dsshome1/0A/di35fer/dataset/{dataset_name}/{data_file_name}"
dataset_file_full_path  = f"/dss/dsshome1/0A/di35fer/dataset/proofwriter_selected_top_1000case/{data_file_name}.jsonl"
# dataset_folder_path     = f"/dss/dsshome1/0A/di35fer/dataset/{dataset_name}/"         # not use yet
# dataset_file_full_path       = "../data/proofwriter_selected_top_1000case/CWA_depth-0_meta-test.jsonl"


output_path_suffix      = time.strftime("%m_%d_%H_%M_%S", time.localtime())

#%% paths
# path_output_log         = f"/dss/dsshome1/0A/di35fer/code/result/{dataset_name}/{model_short}_{PROMPT_METHOD}_{dataset_name}_{output_path_suffix}_log.txt"
# output_file_path        = f"/dss/dsshome1/0A/di35fer/code/result/{dataset_name}/{model_short}_{PROMPT_METHOD}_{dataset_name}_{output_path_suffix}_res.csv"
path_output_log         = f"/dss/dsshome1/0A/di35fer/code/code_vanilla/result/ablation/translation/rule/Single_rule_{model_short}_{seed}_{data_file_name}_{output_path_suffix}_exm{num_example}_{info}_log.txt"
output_file_rule_path   = f"/dss/dsshome1/0A/di35fer/code/code_vanilla/result/ablation/translation/rule/Single_rule_{model_short}_{seed}_{data_file_name}_{output_path_suffix}_exm{num_example}_{info}_rule.csv"



# hyperparameters
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
logging.debug(f"**task                              [run_ablation_translation_rules_single.py]")
logging.debug(f"num_example                         [{num_example}]")
logging.debug(f"seed                                [{seed}]")
logging.debug(f"model                               [{model}]")
logging.debug(f"model_short                         [{model_short}]")
logging.debug(f"PROMPT_METHOD                       [{PROMPT_METHOD}]")
logging.debug(f"arg data_file_name                  [{data_file_name}]")
logging.debug(f"arg dataset_name                    [{dataset_name}]")
logging.debug(f"torch_dtype                         [{torch_dtype}]")
logging.debug(f"isFormatTheory                      [{isFormatTheory}]")
logging.debug(f"dataset_file_full_path              [{dataset_file_full_path}]")
logging.debug(f"path_output_log                     [{path_output_log}]")
logging.debug(f"output_file_rule_path               [{output_file_rule_path}]")


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

def ask(question:str, max_length=500) -> str:
    sequences = pipeline(
        question,
        do_sample=False,
        # do_sample=True,
        # top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=max_length,
    )
    for seq in sequences:
        return seq['generated_text']


#%% Main
# dataset_folder_path = dataset_folder_path
# paths = [str(x) for x in Path(dataset_folder_path).glob("*.json")]
# for dataset_file_path in paths:
Dataset.cleanup_cache_files
dataset = load_dataset('json', data_files=dataset_file_full_path)

n = 0
df_resut_fact = pd.DataFrame()
df_resut_rule = pd.DataFrame()
df_resut_rule.to_csv(output_file_rule_path)
for data in tqdm(dataset['train']):
    n += 1
    theory      = ""
    question    = ""
    prompt      = ""
    theory      = data['theory']
    question    = data['question']

    formulated_rules_list   = []
    rules_raw:list          = code.FR_decomposer_proofwriter_new(theory)['rules']
    for rule in rules_raw:
        if num_example in ['0']:
            prompt_translate_rules  = code_ablation.prompt_formulator_translation_rules_2_ifthen_0_example([rules_raw])
        if num_example in ['1']:
            prompt_translate_rules  = code_ablation.prompt_formulator_translation_rules_2_ifthen_1_example([rules_raw])
        if num_example in ['2']:
            prompt_translate_rules  = code_ablation.prompt_formulator_translation_rules_2_ifthen_2_example([rules_raw])
        if num_example in ['3']:
            prompt_translate_rules  = code_ablation.prompt_formulator_translation_rules_2_ifthen_3_example([rules_raw])
        if num_example in ['4']:
            prompt_translate_rules  = code_ablation.prompt_formulator_translation_rules_2_ifthen_4_example([rules_raw])
        if num_example in ['5']:
            prompt_translate_rules  = code_ablation.prompt_formulator_translation_rules_2_ifthen_5_example([rules_raw])
        if num_example in ['6']:
            prompt_translate_rules  = code_ablation.prompt_formulator_translation_rules_2_ifthen_6_example([rules_raw])

        formulated_rules_raw    = ask(prompt_translate_rules, max_length=700)
        formulated_rules_list.extend([rule for rule in code.result_extractor_translation_rules_2_ifthen(formulated_rules_raw) if rule not in formulated_rules_list])
    
    rules_evaluation_dict   = code_ablation.evaluation_translation_rule(rules_raw, formulated_rules_list)

    # logging.debug(f"theory                    [{theory  }]")
    # logging.debug(f"question                  [{question}]")
    # logging.debug(f"prompt_translate_facts    [{prompt_translate_facts.strip() }]")
    # logging.debug(f"prompt_translate_rules    [{prompt_translate_rules.strip() }]")
    # logging.debug(f"formulated_facts_raw      [{formulated_facts_raw.strip()   }]")
    # logging.debug(f"formulated_rules_raw      [{formulated_rules_raw.strip()   }]")
    # logging.debug(f"formulated_facts_list     [{','.join(formulated_facts_list)  }]")
    # logging.debug(f"formulated_rules_list     [{','.join(formulated_rules_list)  }]")

    # save result: rules
    dic_rule_result = {}
    dic_rule_result.update(data)
    dic_rule_result.update(rules_evaluation_dict)
    dic_rule = {'formulated_rules_list' : formulated_rules_list}

    dic_rule_result.update(dic_rule)

    df_resut_rule = pd.concat([df_resut_rule, pd.DataFrame([dic_rule_result])], ignore_index=True)
    if n % 200 == 0:
        logging.debug(f"nr      [{n}]")
        df_resut_rule.to_csv(output_file_rule_path)
df_resut_rule.to_csv(output_file_rule_path)
logging.debug(f"finished")
print("finished")