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
import code
import code_ablation
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
#%% arg
seed = 42
info = ""
mode_fact       = ''   # multi / single
mode_rule       = ''   # multi / single
mode_inference  = ''   # multi / single
num_example_fact= ''       # TODO: str: 0/2/3 
num_example_rule= ''       # TODO: str: 0-6
model_short     = 'llama2-7b'
data_file_name  = ''
dataset_name    = ''
argv = sys.argv[1:]

opts, args = getopt.getopt(argv, "d:s:m:e:i:a:b:c:d:f",  
        [
            "data_file_name=",          # CWA_depth-0_meta-test.jsonl
            "dataset_name=",            # proofwriter/FOLIO
            "model_short=",             # llama2-13b/llama2-7b
            "seed=",                    # default = 42
            "info=",
            "mode_fact=",               # multi / single     
            "mode_rule=",               # multi / single     
            "mode_inference=",          # multi / single    
            "num_example_fact=",        # 
            "num_example_rule=",        # 
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
    elif opt in ['--mode_fact']:
        mode_fact = arg
    elif opt in ['--mode_rule']:
        mode_rule = arg
    elif opt in ['--mode_inference']:
        mode_inference = arg
    elif opt in ['--num_example_fact']:
        num_example_fact = str(arg)
    elif opt in ['--num_example_rule']:
        num_example_rule = str(arg)


# print(mode_fact       )
# print(mode_rule       )
# print(mode_inference  )
# print(num_example_fact)
# print(num_example_rule)
# print(model_short     )
# print(data_file_name  )
# print(dataset_name    )
#%% parameter
if model_short == "llama-7b-chat-hf":
    model   = "meta-llama/Llama-2-7b-chat-hf"
elif model_short == "llama-13b-chat-hf":
    model   = "meta-llama/Llama-2-13b-chat-hf"
elif model_short == "llama-13b-hf":
    model   = "meta-llama/Llama-2-13b-hf"
elif model_short == "code-7b":
    model   = "codellama/CodeLlama-7b-hf"

#%% path
# dataset_file_full_path       = "/dss/dsshome1/0A/di35fer/dataset/proofwriter/CWA_depth-0_meta-test.jsonl"
# dataset_file_full_path  = f"/dss/dsshome1/0A/di35fer/dataset/{dataset_name}/{data_file_name}"
dataset_file_full_path  = f"/dss/dsshome1/0A/di35fer/dataset/{dataset_name}/{data_file_name}.jsonl"
# dataset_folder_path     = f"/dss/dsshome1/0A/di35fer/dataset/{dataset_name}/"         # not use yet
# dataset_file_full_path       = "../data/proofwriter_selected_top_1000case/CWA_depth-0_meta-test.jsonl"


output_path_suffix      = time.strftime("%m_%d_%H_%M_%S", time.localtime())

#%% paths
# path_output_log         = f"/dss/dsshome1/0A/di35fer/code/result/{dataset_name}/{model_short}_{PROMPT_METHOD}_{dataset_name}_{output_path_suffix}_log.txt"
# output_file_path        = f"/dss/dsshome1/0A/di35fer/code/result/{dataset_name}/{model_short}_{PROMPT_METHOD}_{dataset_name}_{output_path_suffix}_res.csv"
path_output_log = f"/dss/dsshome1/0A/di35fer/code/code_vanilla/result/module/{dataset_name}/lg_fact[{mode_fact}{num_example_fact}]_rule[{mode_rule}{num_example_rule}]_inf[{mode_inference}]_[{model_short}]_{seed}_{data_file_name}_{output_path_suffix}_{info}_log.txt"
output_file_path= f"/dss/dsshome1/0A/di35fer/code/code_vanilla/result/module/{dataset_name}/lg_fact[{mode_fact}{num_example_fact}]_rule[{mode_rule}{num_example_rule}]_inf[{mode_inference}]_[{model_short}]_{seed}_{data_file_name}_{output_path_suffix}_{info}_res.csv"

# hyperparameters
torch_dtype             = torch.float16


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

logging.debug(f"mode_fact                           [{mode_fact       }]")
logging.debug(f"mode_rule                           [{mode_rule       }]")
logging.debug(f"mode_inference                      [{mode_inference  }]")
logging.debug(f"num_example_fact                    [{num_example_fact}]")
logging.debug(f"num_example_rule                    [{num_example_rule}]")


logging.debug(f"arg dataset_name                    [{dataset_name}]")
logging.debug(f"arg data_file_name                  [{data_file_name}]")
logging.debug(f"torch_dtype                         [{torch_dtype}]")
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
df_resut = pd.DataFrame()
df_resut.to_csv(output_file_path)
for data in tqdm(dataset['train']):
    n += 1
    theory      = ""
    question    = ""
    prompt      = ""
    theory      = data['theory']
    question    = data['question']
    facts_rules_raw:dict    = code.FR_decomposer_proofwriter_new(theory)
    facts_raw:list          = facts_rules_raw['facts']
    rules_raw:list          = facts_rules_raw['rules']

    formulated_facts_list   = code_ablation.formulate_fact_module(facts_raw, mode_fact=mode_fact, num_example_fact=num_example_fact, pipeline=pipeline, tokenizer=tokenizer)

    formulated_rules_list   = code_ablation.formulate_rule_module(rules_raw, mode_rule=mode_rule, num_example_rule=num_example_rule, pipeline=pipeline, tokenizer=tokenizer)

    resulted_new_facts_list = code_ablation.inference_module(formulated_facts_list, formulated_rules_list, mode_inference=mode_inference, pipeline=pipeline, tokenizer=tokenizer)

    # process query
    prompt_translate_query  = code.prompt_formulator_translation_facts_2_adj([question])
    formulated_query_raw    = ask(prompt_translate_query)
    formulated_query_list   = code.result_extractor_translation_rules_2_ifthen(formulated_query_raw)
    # answer with code
    answer_code             = code.answer_query(conclusions=resulted_new_facts_list, formulated_query=formulated_query_list[0])
    # answer with LLM
    prompt_LLM_answer_query = code.formulate_answer_query_LLM(formulated_facts=resulted_new_facts_list, 
                                    formulated_query=formulated_query_list[0])
    answer_LLM_raw          = ask(prompt_LLM_answer_query, max_length=300)
    answer_LLM              = code.result_extractor_LLM_answer_query(answer_LLM_raw)
    # logging.debug(f"theory                    [{theory  }]")
    # logging.debug(f"question                  [{question}]")
    # logging.debug(f"prompt_translate_facts    [{prompt_translate_facts }]")
    # logging.debug(f"prompt_translate_rules    [{prompt_translate_rules }]")
    # logging.debug(f"formulated_facts_raw      [{formulated_facts_raw.strip()}]")
    # logging.debug(f"formulated_rules_raw      [{formulated_rules_raw.strip()}]")
    # logging.debug(f"facts_raw                 [{','.join(facts_raw)  }]")
    # logging.debug(f"rules_raw                 [{','.join(rules_raw)  }]")
    # logging.debug(f"formulated_facts_list     [{','.join(formulated_facts_list)  }]")
    # logging.debug(f"formulated_rules_list     [{','.join(formulated_rules_list)  }]")
    # logging.debug(f"prompt_facts_rules        [{prompt_facts_rules     }]")
    # logging.debug(f"prompt_query              [{prompt_query           }]")
    # logging.debug(f"result_query_raw          [{result_query_raw       }]")
    # logging.debug(f"resulted_new_facts_list         [{','.join(resulted_new_facts_list)}]")
    # logging.debug(f"prompt_translate_query    [{prompt_translate_query }]")
    # logging.debug(f"formulated_query_raw      [{formulated_query_raw.strip()}]")
    # logging.debug(f"formulated_query_list     [{','.join(formulated_query_list)}]")
    # logging.debug(f"answer_code               [{answer_code            }]")
    # logging.debug(f"prompt_LLM_answer_query   [{prompt_LLM_answer_query}]")
    # logging.debug(f"answer_LLM_raw            [{answer_LLM_raw.strip()}]")
    # logging.debug(f"answer_LLM                [{answer_LLM             }]")

    # formulate result
    dic_answer = {
        # 'prompt_translate_facts'  :   prompt_translate_facts.strip(),
        # 'prompt_translate_rules'  :   prompt_translate_rules.strip(),
        # 'formulated_facts_raw'    :   formulated_facts_raw.strip(),
        # 'formulated_rules_raw'    :   formulated_rules_raw.strip(),
        'formulated_facts_str'      :   code.number_list_to_str(formulated_facts_list),
        'formulated_rules_str'      :   code.number_list_to_str(formulated_rules_list),
        # 'result_query_raw'        :   result_query_raw.strip(),
        # 'formulated_query_raw'    :   formulated_query_raw.strip(),
        'resulted_new_facts_list'   :   code.number_list_to_str(resulted_new_facts_list),
        'formulated_query_list'     :   code.number_list_to_str(formulated_query_list),
        # 'prompt_LLM_answer_query' :   prompt_LLM_answer_query.strip(),
        'answer_code'               :   str(answer_code), # note: 'answer' column has been used already for the ground-truth
        'answer_LLM'                :   answer_LLM, # note: 'answer' column has been used already for the ground-truth
    }
    dic_meta = {}
    dic_meta.update(data)
    dic_meta.update(dic_answer)
    df_resut = pd.concat([df_resut, pd.DataFrame([dic_meta])], ignore_index=True)
    if n % 200 == 0:
        logging.debug(f"nr      [{n}]")
        df_resut.to_csv(output_file_path)
df_resut.to_csv(output_file_path)
logging.debug(f"finished")
print("finished")