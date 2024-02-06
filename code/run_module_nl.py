# version: 01.27
from    transformers    import AutoTokenizer
import  transformers
import  torch
import  json
import  pandas          as pd
from    tqdm            import tqdm
import  logging
import  time
from    datasets        import load_dataset
from    datasets        import Dataset
from    pathlib         import Path
import  sys
import  getopt
import  code
import  code_ablation
import  code_nl
import  os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
#%% arg
seed = 42
info = ""
model_short     = 'llama2-7b'
data_file_name  = ''
dataset_name    = ''

mode_inference      = ''    # multi / single
mode_conclusion     = ''    # multi / single
nr_prompt_inference = ''    # TODO: 
nr_prompt_conclusion= ''    # TODO: 
n_step              = ''

argv                = sys.argv[1:]

opts, args = getopt.getopt(argv, "d:s:m:e:i:a:b:c:d:f",  
        [
            "data_file_name=",          # CWA_depth-0_meta-test.jsonl
            "dataset_name=",            # proofwriter/FOLIO
            "model_short=",             # llama2-13b/llama2-7b
            "seed=",                    # default = 42
            "info=",
            "mode_inference=",          # multi / single 
            "mode_conclusion=",         # multi / single    
            "nr_prompt_inference=",     #
            "nr_prompt_conclusion=",    #
            "n_step=",                  # step of inference
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
    elif opt in ['--mode_inference']:
        mode_inference = arg
    elif opt in ['--mode_conclusion']:
        mode_conclusion = arg
    elif opt in ['--nr_prompt_inference']:
        nr_prompt_inference = str(arg)
    elif opt in ['--nr_prompt_conclusion']:
        nr_prompt_conclusion = str(arg)
    elif opt in ['--n_step']:
        n_step = str(arg)


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
path_output_log         = f"/dss/dsshome1/0A/di35fer/code/code_vanilla/result/module/{dataset_name}/NL_inf[{mode_inference}]_conc[{mode_conclusion}]_[{n_step}]_[{model_short}]_pinf[{nr_prompt_inference}]_pconc[{nr_prompt_conclusion}]_{seed}_{data_file_name}_{output_path_suffix}_{info}_log.txt"
output_file_path        = f"/dss/dsshome1/0A/di35fer/code/code_vanilla/result/module/{dataset_name}/NL_inf[{mode_inference}]_conc[{mode_conclusion}]_[{n_step}]_[{model_short}]_pinf[{nr_prompt_inference}]_pconc[{nr_prompt_conclusion}]_{seed}_{data_file_name}_{output_path_suffix}_{info}_res.csv"

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

logging.debug(f"mode_inference                      [{mode_inference  }]")
logging.debug(f"mode_conclusion                     [{mode_conclusion  }]")
logging.debug(f"nr_prompt_inference                 [{nr_prompt_inference  }]")
logging.debug(f"nr_prompt_conclusion                [{nr_prompt_conclusion  }]")
logging.debug(f"n_step                              [{n_step  }]")


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
    facts_rules_raw:dict     = code.FR_decomposer_proofwriter_new(theory)
    facts_list:list          = facts_rules_raw['facts']
    rules_list:list          = facts_rules_raw['rules']
    
    # n_step
    resulted_new_facts_list = code_nl.inference_module_n_step(facts_list, rules_list, 
                                n_step          = n_step,
                                nr_prompt       = nr_prompt_inference, 
                                mode_inference  = mode_inference, 
                                pipeline        = pipeline, 
                                tokenizer       = tokenizer)

    # answer with code
    answer_code             = code.answer_query(conclusions=resulted_new_facts_list, formulated_query=question)
    
    # answer with LLM (conclusion or comparision)
    answer_LLM, answer_dict = code_nl.conclusion_module(
                                facts = resulted_new_facts_list, 
                                query = question,
                                nr_prompt           = nr_prompt_conclusion, 
                                mode_conclusion  = mode_conclusion,
                                pipeline        = pipeline,
                                tokenizer       = tokenizer)
    
    # {'answer_labels':[], 
    #  'answer_reasons':[], 
    #  'answer_wo_prompts':[]}


    # logging.debug(f"theory                    [{theory  }]")
    # logging.debug(f"question                  [{question}]")
    # logging.debug(f"facts_raw                 [{code.number_list_to_str(facts_list)  }]")
    # logging.debug(f"rules_raw                 [{code.number_list_to_str(rules_list)  }]")
    # logging.debug(f"resulted_new_facts_list   [{code.number_list_to_str(resulted_new_facts_list)}]")
    # logging.debug(f"answer_code               [{answer_code            }]")

    # logging.debug(f"answer_LLM                [{answer_LLM             }]")
    # logging.debug(f"answer_labels             [{code.number_list_to_str(answer_dict['answer_labels'])             }]")       # list
    # logging.debug(f"answer_reasons            [{code.number_list_to_str(answer_dict['answer_reasons'])             }]")      # list
    # logging.debug(f"answer_wo_prompts         [{code.number_list_to_str(answer_dict['answer_wo_prompts'])             }]")   # list

    # formulate result
    dic_answer = {
        # 'prompt_translate_facts'  :   prompt_translate_facts.strip(),
        # 'prompt_translate_rules'  :   prompt_translate_rules.strip(),
        # 'formulated_facts_raw'    :   formulated_facts_raw.strip(),
        # 'formulated_rules_raw'    :   formulated_rules_raw.strip(),

        # 'result_query_raw'        :   result_query_raw.strip(),
        # 'formulated_query_raw'    :   formulated_query_raw.strip(),
        'resulted_new_facts_list'   :   code.number_list_to_str(resulted_new_facts_list),
        'answer_code'               :   str(answer_code), # note: 'answer' column has been used already for the ground-truth
        'answer_LLM'                :   answer_LLM, # note: 'answer' column has been used already for the ground-truth
        'answer_labels'             :   code.number_list_to_str(answer_dict['answer_labels']    ),
        'answer_reasons'            :   code.number_list_to_str(answer_dict['answer_reasons']   ),
        'answer_wo_prompts'         :   code.number_list_to_str(answer_dict['answer_wo_prompts']),
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