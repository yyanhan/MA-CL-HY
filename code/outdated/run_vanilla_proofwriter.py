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
import os
import code_ablation

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
path_output_log = f"/dss/dsshome1/0A/di35fer/code/code_vanilla/result/proofwriter_1000/{PROMPT_METHOD}_{model_short}_{seed}_{data_file_name}_{output_path_suffix}_log.txt"
output_file_path= f"/dss/dsshome1/0A/di35fer/code/code_vanilla/result/proofwriter_1000/{PROMPT_METHOD}_{model_short}_{seed}_{data_file_name}_{output_path_suffix}_res.csv"


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
    # facts_raw:list          = code.FR_decomposer_proofwriter(theory)['facts']
    # rules_raw:list          = code.FR_decomposer_proofwriter(theory)['rules']
    facts_raw:list          = code.FR_decomposer_proofwriter_new(theory)['facts']
    rules_raw:list          = code.FR_decomposer_proofwriter_new(theory)['rules']
    prompt_translate_facts  = code_ablation.prompt_formulator_translation_facts_2_adj_2_example(facts_raw)
    prompt_translate_rules  = code.prompt_formulator_translation_rules_2_ifthen(rules_raw)
    formulated_facts_raw    = ask(prompt_translate_facts, max_length=600)
    formulated_rules_raw    = ask(prompt_translate_rules, max_length=600)
    formulated_facts_list   = code.result_extractor_translation_fact_2_adj(formulated_facts_raw)
    formulated_rules_list   = code.result_extractor_translation_rules_2_ifthen(formulated_rules_raw)
    prompt_facts_rules      = code.formulate_facts_rules_2_str(facts=formulated_facts_list, rules=formulated_rules_list)
    prompt_query            = code.prompt_formulator_query_ifthen_multiQ(formulated_facts_rules=prompt_facts_rules)
    result_query_raw        = ask(prompt_query, max_length=1000)
    result_query_dict       = code.query_result_extractor(result_query_raw)
    result_query_dict['New Facts'] = formulated_facts_list + [conclude.strip() for conclude in result_query_dict['Conclusion'] 
                                                              if conclude.strip() not in formulated_facts_list]

    # process query
    prompt_translate_query  = code.prompt_formulator_translation_facts_2_adj([question])
    formulated_query_raw    = ask(prompt_translate_query)
    formulated_query_list   = code.result_extractor_translation_rules_2_ifthen(formulated_query_raw)
    answer_code             = code.answer_query(conclusions=result_query_dict['New Facts'], formulated_query=formulated_query_list[0])
    prompt_LLM_answer_query = code.formulate_answer_query_LLM(formulated_facts=result_query_dict['New Facts'], 
                                    formulated_query=formulated_query_list[0])
    answer_LLM_raw          = ask(prompt_LLM_answer_query, max_length=300)
    answer_LLM              = code.result_extractor_LLM_answer_query(answer_LLM_raw)
    # logging.debug(f"theory                    [{theory  }]")
    # logging.debug(f"question                  [{question}]")
    # logging.debug(f"prompt_translate_facts    [{prompt_translate_facts.strip() }]")
    # logging.debug(f"prompt_translate_rules    [{prompt_translate_rules.strip() }]")
    # logging.debug(f"formulated_facts_raw      [{formulated_facts_raw.strip()   }]")
    # logging.debug(f"formulated_rules_raw      [{formulated_rules_raw.strip()   }]")
    # logging.debug(f"formulated_facts_list     [{','.join(formulated_facts_list)  }]")
    # logging.debug(f"formulated_rules_list     [{','.join(formulated_rules_list)  }]")
    # logging.debug(f"prompt_facts_rules        [{prompt_facts_rules     }]")
    # logging.debug(f"prompt_query              [{prompt_query           }]")
    # logging.debug(f"result_query_raw          [{result_query_raw       }]")
    # logging.debug(f"result_query_dict['Facts']         [{','.join(result_query_dict['Provided Facts'])}]")
    # logging.debug(f"result_query_dict['Rules']         [{','.join(result_query_dict['Provided Rules'])}]")
    # logging.debug(f"result_query_dict[Conclusion]      [{','.join(result_query_dict['Conclusion'])}")
    # logging.debug(f"result_query_dict[Explanation]     [{result_query_dict['Explanation']}]")
    # logging.debug(f"result_query_dict[New Facts]       [{','.join(result_query_dict['New Facts'])}]")
    # logging.debug(f"prompt_translate_query    [{prompt_translate_query }]")
    # logging.debug(f"formulated_query_raw      [{formulated_query_raw   }]")
    # logging.debug(f"formulated_query_list     [{','.join(formulated_query_list)}]")
    # logging.debug(f"answer_code               [{answer_code            }]")
    # logging.debug(f"prompt_LLM_answer_query   [{prompt_LLM_answer_query}]")
    # logging.debug(f"answer_LLM_raw            [{answer_LLM_raw.strip()         }]")
    # logging.debug(f"answer_LLM                [{answer_LLM             }]")




    # formulate result
    dic_answer = {
        'prompt_translate_facts'  :   prompt_translate_facts.strip(),
        'prompt_translate_rules'  :   prompt_translate_rules.strip(),
        'formulated_facts_raw'    :   formulated_facts_raw.strip(),
        'formulated_rules_raw'    :   formulated_rules_raw.strip(),
        'formulated_facts_list'   :   ','.join(formulated_facts_list),
        'formulated_rules_list'   :   ','.join(formulated_rules_list),
        'dict[Facts]'             :   ','.join(result_query_dict['Provided Facts']),
        'dict[Rules]'             :   ','.join(result_query_dict['Provided Rules']),
        # 'result_query_raw'        :   result_query_raw.strip(),
        'dict[Conclusion]'        :   ','.join(result_query_dict['Conclusion']),
        'dict[Explanation]'       :   result_query_dict['Explanation'],
        'dict[New Facts]'         :   ','.join(result_query_dict['New Facts']),
        # 'formulated_query_raw'    :   formulated_query_raw.strip(),
        'formulated_query_list'   :   ','.join(formulated_query_list),
        # 'prompt_LLM_answer_query' :   prompt_LLM_answer_query.strip(),
        'answer_code'             :   str(answer_code), # note: 'answer' column has been used already for the ground-truth
        'answer_LLM'              :   answer_LLM, # note: 'answer' column has been used already for the ground-truth
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