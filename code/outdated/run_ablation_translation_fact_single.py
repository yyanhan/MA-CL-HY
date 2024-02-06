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
path_output_log         = f"/dss/dsshome1/0A/di35fer/code/code_vanilla/result/ablation/translation/Single_fact_{model_short}_{seed}_{data_file_name}_{output_path_suffix}_ex{num_example}_{info}log.txt"
output_file_fact_path   = f"/dss/dsshome1/0A/di35fer/code/code_vanilla/result/ablation/translation/Single_fact_{model_short}_{seed}_{data_file_name}_{output_path_suffix}_ex{num_example}_{info}.csv"
# output_file_rule_path   = f"/dss/dsshome1/0A/di35fer/code/code_vanilla/result/ablation/translation/Trans_{model_short}_{seed}_{data_file_name}_{output_path_suffix}_{info}_rule.csv"



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
logging.debug(f"**task                              [run_ablation_translation_fact.py]")
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
logging.debug(f"output_file_fact_path               [{output_file_fact_path}]")
# logging.debug(f"output_file_rule_path               [{output_file_rule_path}]")


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
df_resut_fact.to_csv(output_file_fact_path)
# df_resut_rule.to_csv(output_file_rule_path)
for data in tqdm(dataset['train']):
    n += 1
    theory      = ""
    question    = ""
    prompt      = ""
    theory      = data['theory']
    question    = data['question']
    formulated_facts_list   = []
    facts_raw:list          = code.FR_decomposer_proofwriter_new(theory)['facts']
    # rules_raw:list          = code.FR_decomposer_proofwriter_new(theory)['rules']
    for fact in facts_raw:
        if num_example in ['0']:
            prompt_translate_facts  = code_ablation.prompt_formulator_translation_facts_2_adj_0_example([fact])
        if num_example in ['2']:
            prompt_translate_facts  = code_ablation.prompt_formulator_translation_facts_2_adj_2_example([fact])
        if num_example in ['3']:
            prompt_translate_facts  = code_ablation.prompt_formulator_translation_facts_2_adj_3_example([fact])

        # prompt_translate_rules  = code.prompt_formulator_translation_rules_2_ifthen(rules_raw)
        formulated_facts_raw    = ask(prompt_translate_facts, max_length=400)
        # formulated_rules_raw    = ask(prompt_translate_rules, max_length=400)
        formulated_facts_list.extend([fact for fact in code.result_extractor_translation_fact_2_adj(formulated_facts_raw) if fact not in formulated_facts_list])
        # formulated_rules_list   = code.result_extractor_translation_rules_2_ifthen(formulated_rules_raw)
    
    facts_evaluation_dict   = code_ablation.evaluation_translation_fact(facts_raw, formulated_facts_list)

    # logging.debug(f"theory                    [{theory  }]")
    # logging.debug(f"question                  [{question}]")
    # logging.debug(f"prompt_translate_facts    [{prompt_translate_facts.strip() }]")
    # logging.debug(f"prompt_translate_rules    [{prompt_translate_rules.strip() }]")
    # logging.debug(f"formulated_facts_raw      [{formulated_facts_raw.strip()   }]")
    # logging.debug(f"formulated_rules_raw      [{formulated_rules_raw.strip()   }]")
    # logging.debug(f"formulated_facts_list     [{','.join(formulated_facts_list)  }]")
    # logging.debug(f"formulated_rules_list     [{','.join(formulated_rules_list)  }]")


    # save result: facts
    dic_fact_result = {}
    dic_fact_result.update(data)
    dic_fact_result.update(facts_evaluation_dict)
    dic_fact = {
        'prompt_translate_facts'  :   prompt_translate_facts.strip(),
        'formulated_facts_raw'    :   formulated_facts_raw.strip(),
    }
    for index, fact in enumerate(formulated_facts_list):
        dic_fact[str(index+1)] = fact
    dic_fact_result.update(dic_fact)

    # save result: rules
    # dic_rule_result = {}
    # dic_rule_result.update(data)
    # dic_rule = {
    #     'prompt_translate_rules'  :   prompt_translate_rules.strip(),
    #     'formulated_rules_raw'    :   formulated_rules_raw.strip(),
    # }
    # for index, rule in enumerate(formulated_rules_list):
    #     dic_rule[str(index+1)] = rule
    # dic_rule_result.update(dic_rule)

    df_resut_fact = pd.concat([df_resut_fact, pd.DataFrame([dic_fact_result])], ignore_index=True)
    # df_resut_rule = pd.concat([df_resut_rule, pd.DataFrame([dic_rule_result])], ignore_index=True)
    if n % 200 == 0:
        logging.debug(f"nr      [{n}]")
        df_resut_fact.to_csv(output_file_fact_path)
        # df_resut_rule.to_csv(output_file_rule_path)
df_resut_fact.to_csv(output_file_fact_path)
# df_resut_rule.to_csv(output_file_rule_path)
logging.debug(f"finished")
print("finished")