import code
import code_ablation
import copy

#%% Tools

def list_2_str(facts:list) -> str:
    res = ""
    for index, fact in enumerate(facts):
        res += f"{fact}\n"
    return res

def list_2_str_num(facts:list) -> str:
    # for inference single
    res = ""
    for index, fact in enumerate(facts):
        res += f"{index+1}. {fact}\n"
    return res

def list_2_str_num_semi(facts:list) -> str:
    # for inference multi
    res = ""
    for index, fact in enumerate(facts):
        res += f"{index+1}. {fact}; "
    return res


#%% Inference

def result_extractor_inference_single(prompt, result):
    result = result.strip()
    if prompt in result:
        result = result.replace(prompt, "")
    if '</Answer>' in result:
        result = result.split('</Answer>')[0].strip()
    result_lines = result.split('\n')
    answer_label_line = result_lines[0]
    if len(result_lines) > 1:
        answer_conclusion_line = result_lines[1]
    else:
        answer_conclusion_line = ''

    if 'yes' in answer_label_line.lower() and 'no' not in answer_label_line.lower():
        answer_label = 'yes'
    elif 'yes' not in answer_label_line.lower() and 'no' in answer_label_line.lower():
        answer_label = 'no'
    else:
        answer_label = 'error'

    answer_conclusion_after_produce = answer_conclusion_line.split('Produce:')
    if len(answer_conclusion_after_produce) > 1:
        answer_conclusion = answer_conclusion_after_produce[1].strip()
    else:
        answer_conclusion = answer_conclusion_after_produce[0]
    
    return answer_label, [answer_conclusion], answer_conclusion_line
    

def result_extractor_inference_multi(prompt, result):
    result = result.strip()
    if prompt in result:
        result = result.replace(prompt, "")
    if '</Answer>' in result:
        result = result.split('</Answer>')[0].strip()
    result_lines = result.split('\n')
    result_line = result_lines[0]
    results = result_line.split(';')
    conclusions_list = []
    for result in results:
        result = result.replace('[', "")
        result = result.replace(']', "")
        result = result.strip()
        if '. ' in result:  # 1. sentence
            result = result.split('. ')[1]
            result = result.strip()
        elif '.' in result: # sentence.
            result = result.split('.')[0]
            result = result.strip()
        if len(result) > 0:
            conclusions_list.append(result)
    return "", conclusions_list, result_line


#%% Inference 


def inference_module_n_step(facts:list, rules:list, n_step:str, mode_inference:str, pipeline, tokenizer, nr_prompt:str='0') -> list:
    """_summary_

    Args:
        facts (list): _description_
        rules (list): _description_
        n_step (str): _description_
        mode_inference (str): _description_
        pipeline (_type_): _description_
        tokenizer (_type_): _description_
        nr_prompt (str, optional): _description_. Defaults to '0'.

    Returns:
        list: _description_
    """
    n_step = int(n_step)
    resulted_new_facts_list = copy.deepcopy(facts)
    for step in range(n_step):
        answer_label_list, new_facts_list, answer_full = inference_module_1_step(
                facts = resulted_new_facts_list, 
                rules = rules, 
                nr_prompt = nr_prompt,
                mode_inference = mode_inference,
                pipeline=pipeline, tokenizer=tokenizer)
        resulted_new_facts_list = new_facts_list
        # resulted_new_facts_list.extend(resulted_new_facts_list)   # no need to extend here

    return resulted_new_facts_list

def inference_module_1_step(facts:list, rules:list, mode_inference:str, pipeline, tokenizer, nr_prompt:str='0') -> list:
    """_summary_

    Args:
        facts (list): _description_
        rules (list): _description_
        mode_inference (str): _description_
        pipeline (_type_): _description_
        tokenizer (_type_): _description_
        nr_prompt (str, optional): _description_. Defaults to '0'.

    Returns:
        list: answer_label_list, resulted_new_facts_list, answer_full_list
    """
    resulted_new_facts_list = copy.deepcopy(facts)
    answer_label_list       = []
    answer_full_list        = []
    if mode_inference in ['multi']: 
        prompt            = inference_select_prompt_nl(facts=facts, rules=rules, mode_inference=mode_inference, nr_prompt=nr_prompt)
        answer_raw        = code_ablation.ask(pipeline, tokenizer, prompt, max_length=600)
        _, answer_conclusions, answer_full = result_extractor_inference_multi(prompt=prompt, result=answer_raw)
        resulted_new_facts_list.extend(
            conclude.strip() for conclude in answer_conclusions
            if conclude.strip() not in resulted_new_facts_list
        )
        answer_full_list.append(answer_full)
        return [], resulted_new_facts_list, answer_full_list
    elif mode_inference in ['single']:
        for rule in rules:
            prompt            = inference_select_prompt_nl(facts=facts, rules=[rule], mode_inference = mode_inference, nr_prompt=nr_prompt)
            answer_raw        = code_ablation.ask(pipeline, tokenizer, prompt, max_length=600)
            answer_label, answer_conclusions, answer_full = result_extractor_inference_single(prompt=prompt, result=answer_raw)
            resulted_new_facts_list.extend(
                conclude.strip() for conclude in answer_conclusions
                if conclude.strip() not in resulted_new_facts_list
            )
            answer_full_list.append(answer_full)
            answer_label_list.append(answer_label)
        return answer_label_list, resulted_new_facts_list, answer_full_list
    else:
        return [], [], []

def inference_select_prompt_nl(facts:list, rules:list, mode_inference:str, nr_prompt:str='0') -> str:
    prompt = ''
    mode =  mode_inference+"_"+nr_prompt
    if mode == 'multi_0':
        facts_str = list_2_str_num_semi(facts)
        rules_str = list_2_str_num_semi(rules)
        prompt = ablation_inference_prompt_formulate_multi_nl(facts_str, rules_str)
        return prompt 
    elif mode == 'single_0':
        facts_str = list_2_str_num(facts)
        rule   = rules[0]
        prompt = ablation_inference_prompt_formulate_single_nl(facts_str, rule)
    return prompt 

def ablation_inference_prompt_formulate_multi_nl(facts_str_num:str, rules_str_num:str):
    prompt = f"""Task: please answer what can be produce from the rules with given facts?
<Example>
Facts: 1. Erin is round; 2. Erin is nice;
Rule: 1. If Erin is round then Erin is white; 2. If Erin is nice then Erin is cute;
Answer: 1. Erin is white; 2. Erin is cute;
Facts: 1. Bob is kind; 2. Bob is rude; 3. Bob is rough;
Rule: 1. If Bob is kind then Bob is blue; 2. If Bob is rough, then Bob is nice; 3. If Bob is cute then Bob is kind;
Answer: 1. Bob is blue; 2. Bob is nice; 
Facts: 1. Bob is kind; 2. Bob is rude;
Rule: 1. If Bob is pretty then Bob is blue; 2. If Bob is rough, then Bob is nice; 3. If Bob is cute then Bob is kind;
Answer: NOTHING
</Example>
Facts: {facts_str_num}.
Rule: {rules_str_num}. Please answer with the following format:
Answer: [your answer here].
<Answer>
Answer:"""
    return prompt

def ablation_inference_prompt_formulate_single_nl(facts_str_num:str, rules_str_num:str):
    prompt = f"""Task: please answer whether the following rule is satisfied under the provided given Facts?
If yes, say 'yes', if no, say 'no',
If it is satisfied, please answer what does it produce.
<Example>
Facts: Erin is round
Rule: If Erin is round then Erin is white.
Answer: yes, the condition can be found in facts.
Procude: Erin is white
Facts: Erin is kind
Rule: If Erin is round then Erin is white.
Answer: no, the condition can't be found in facts.
Procude: NOTHING
</Example>
Facts: {facts_str_num}
Rule: {rules_str_num},
Please answer with the following format:
Answer: [yes or no]
Produce: [your answer here]
<Answer>
Answer:"""
    return prompt

#%% Conclusion
    
def conclusion_module(facts:list, query:str, mode_conclusion:str, pipeline, tokenizer,  nr_prompt:str='0'):
    """_summary_

    Args:
        facts (list): _description_
        query (str): _description_
        mode_conclusion (str): multi: conclusion, single: conparison.
        pipeline (_type_): _description_
        tokenizer (_type_): _description_
        nr_prompt (str, optional): _description_. Defaults to '0'.

    Returns:
        _type_: 
    """
    if mode_conclusion in ['multi']: 
        prompt = conclusion_select_prompt_nl(facts, query, nr_prompt)   
        answer = code_ablation.ask(pipeline, tokenizer, prompt, max_length=600)
        answer_label, answer_reason = ablation_conclusion_result_extractor(prompt, answer)  # yes/no
        answer_wo_prompt = answer.replace(prompt, '')
        if 'yes' in [answer_label.lower]:
            return 'True', {'answer_labels':[answer_label], 'answer_reasons':[answer_reason], 'answer_wo_prompts':[answer_wo_prompt]}
        else:
            return 'False', {'answer_labels':[answer_label], 'answer_reasons':[answer_reason], 'answer_wo_prompts':[answer_wo_prompt]}

    elif mode_conclusion in ['single']: 
        # aka comparision
        answer_list        = [] # not save
        answer_labels      = []
        answer_reasons     = []
        answer_prompts     = [] # not save
        answer_wo_prompts  = []
        for fact in facts:
            # prompt
            prompt = comparison_select_prompt_nl(fact, query, nr_prompt)
            answer = code_ablation.ask(pipeline, tokenizer, prompt, max_length=600)
            answer_label, answer_reason = ablation_conclusion_result_extractor(prompt, answer)
            # answer_list.append(answer)               # not save
            answer_labels.append(answer_label)  # yes, no
            answer_reasons.append(answer_reason)
            # answer_prompts.append(prompt)       # not save
            answer_wo_prompts.append(answer.replace(prompt, ''))
        
        # answer_list        = [item.strip() for item in answer           ] # not save
        answer_labels      = [item.strip() for item in answer_labels    ]
        answer_reasons     = [item.strip() for item in answer_reasons   ]
        # answer_prompts     = [item.strip() for item in answer_prompts   ] # not save
        answer_wo_prompts  = [item.strip() for item in answer_wo_prompts]
        
        if 'yes' in [label.lower() for label in answer_labels]:
            return 'True', {'answer_labels':answer_labels, 'answer_reasons':answer_reasons, 'answer_wo_prompts':answer_wo_prompts}
        else:
            return 'False', {'answer_labels':answer_labels, 'answer_reasons':answer_reasons, 'answer_wo_prompts':answer_wo_prompts}


def conclusion_select_prompt_nl(facts:list, query:str, nr_prompt:str='0') -> str:
    prompt = ''
    if nr_prompt == '0':
        facts_str_num = ablation_conclusion_formulate_facts_number(facts)
        prompt = formulate_prompt_conclusion(facts_str_num, query)
        return prompt 
    return prompt 

def comparison_select_prompt_nl(fact:str, query:str, nr_prompt:str='0') -> str:
    prompt = ''
    if nr_prompt == '0':
        prompt = formulate_prompt_comparison(fact, query)
        return prompt 
    return prompt 

#%% prompts
def formulate_prompt_conclusion(context:str, query:str)->str:
    prompt = f"""Task: Determine if the query is mentioned in the given context.
If yes, please tell me the position of the query.
Please pay attention to spelling strictly,
<Example>
Context: 
1. John is kind.
2. Fiona is smart. 
3. Chris is kind.
Query: Bob is good.
Answer: no, because the query can not be found in the contexts.
</Example>
<Example>
Context: 
1. John is kind.
2. Fiona is smart. 
3. Chris is kind.
Query: Fiona is smart.
Answer: yes, because the query can be found in the contexts 2.
</Example>
<Question>
Context:
{context}Query: {query}
</Question>
<Format>
Please answer with the format:
Answer: <your answer here>
</Format>
<Answer>
Answer:"""
    return prompt

def formulate_prompt_comparison(context:str, query:str)->str:
    prompt = f"""Task: Determine if the two sentences are the same. Please pay attention to the spelling strictly.
<Example 1>
1. John is kind.
2. Fiona is smart. 
Answer: no
</Example 1>
<Example 2>
1. John is nice.
2. John is nice. 
Answer: yes
</Example 2>
<Sentence>
1: {context}
2: {query}
</Sentence>
<Answer>
Answer:"""
    return prompt

#%% Tools
def ablation_conclusion_parse_facts(facts:str)->list:
    facts_list = facts.split(',')
    facts_list = [fact.strip()+'.' for fact in facts_list]
    return facts_list

def ablation_conclusion_formulate_facts_number(facts:list)->str:
    res = ""
    for index, fact in enumerate(facts):
        res += f"{index+1}. {fact}.\n"
    return res

def ablation_conclusion_formulate_factstr_factnum(facts:str)->str:
    fact_list = ablation_conclusion_parse_facts(facts)
    fact_num_list = ablation_conclusion_formulate_facts_number(fact_list)
    return fact_num_list

def ablation_conclusion_result_extractor(prompt, result):
    result = result.strip()
    if prompt in result:
        result = result.replace(prompt, "")
    result = result.strip()
    if '</Answer>' in result:
        result = result.split('</Answer>')[0].strip()
    result = result.strip()
    if 'yes' in result.lower() and 'no' not in result.lower():
        return 'yes', result
    elif 'yes' not in result.lower() and 'no' in result.lower():
        return 'no', result
    else:
        return 'error', result