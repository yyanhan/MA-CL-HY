import code
import code_ablation
from code_ablation import ask
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def replace_token(sentence, from_token, to_token):
    sentence_token = sentence.split(' ')
    sentence_token = [token.strip() for token in sentence_token]
    result_token = []
    for token in sentence_token:
        if from_token == token:
            token = token.replace(from_token, to_token)
        result_token.append(token)
    result_token = [r for r in result_token if len(r) > 1]
    result = ' '.join(result_token)
    return result

def find_verb(sentence:str) -> str:
    tokens = word_tokenize(sentence)
    tagged_words = pos_tag(tokens)
    verbs = [word[0] for word in tagged_words if word[1].startswith('VB')]
    if len(verbs) > 0:
        return verbs[0]
    return ''

def lemma_verb(verb:str):
    lemmatizer = WordNetLemmatizer()
    verb_lemma = lemmatizer.lemmatize(verb, pos='v')
    return verb_lemma


def ablation_inference_parser_rule_nl(rule:str):
    rule = rule.replace('.', '')
    if 'someone' in rule:
        rule = rule.replace('someone', 'Bob')
        rule = rule.replace('they', 'Bob')
        if 'it' in rule:
            rule = replace_token(rule, 'it', 'the cat')
        if 'are' in rule:
            rule = replace_token(rule, 'are', 'is')
    elif 'something' in rule:
        rule = rule.replace('something', 'the cat')
        if 'it' in rule:
            rule = replace_token(rule, 'it', 'the cat')
        if 'are' in rule:
            rule = replace_token(rule, 'are', 'is')
    rule_if_then_list = rule.split('then')
    condition_str = rule_if_then_list[0]
    conclusion_str = rule_if_then_list[1]
    # remove if
    condition_str = condition_str.split('If ')[1]
    # process condition
    if 'and' in condition_str:
        conditions_list = condition_str.split('and')
    else:
        conditions_list = [condition_str]
    conditions_list = [condition.strip() for condition in conditions_list]

    # with one word, like something is red and young -> ['the cat is red', 'young']
    # ['the cat is red', 'young'] -> ['the cat is young']

    # first condition:
    condition_first = conditions_list[0]
    if 'is' in condition_first:
        noun = condition_first.split('is')[0]
    else:
        verb = find_verb(condition_first)
        noun = condition_first.split(verb)
        

    # if other conditions have no noun?
    # then add them
    for index, condition in enumerate(conditions_list[1:]):
        condition_token = condition.split(' ')
        if len(condition_token) < 3:
            # only adj
            conditions_list[index+1] = f"{noun.strip()} is {condition}"


    conclusion_str = conclusion_str.strip()
    result = {
        'condition_list'     :   conditions_list,
        'conclusion_str'    :   conclusion_str
    }
    return result



def ablation_inference_parser_rule_logic(condition_list:list) -> list:
    # condition_nl -> condition_logic
    lemmatizer = WordNetLemmatizer()
    conditions_logic_list = []
    for condition in condition_list:
        # condition = replace_token(condition, 'the', '')
        # condition = condition.strip()
        if ' is ' in condition: # '__is__' not 'is', because like 'visit'
            # condition = condition.replace('the', '')
            # condition = condition.strip()
            condition_list = condition.split(' is ')
            noun = condition_list[0]
            adj = condition_list[1]
            noun = replace_token(noun, 'the', '')
            adj = replace_token(adj, 'the', '')
            conditions_logic_list.append(f'{adj.capitalize().strip()}({noun.capitalize().strip()})')
        elif 'likes' in condition:
            # condition = condition.replace('the', '')
            # condition = condition.strip()
            condition_list = condition.split('likes')
            noun = condition_list[0]
            adj = condition_list[1]
            noun = replace_token(noun, 'the', '')
            adj = replace_token(adj, 'the', '')
            conditions_logic_list.append(f'Like({noun.capitalize().strip()}, {adj.capitalize().strip()})')
        elif 'like' in condition:
            # condition = condition.replace('the', '')
            # condition = condition.strip()
            condition_list = condition.split('like')
            noun = condition_list[0]
            adj = condition_list[1]
            noun = replace_token(noun, 'the', '')
            adj = replace_token(adj, 'the', '')
            conditions_logic_list.append(f'Like({noun.capitalize().strip()}, {adj.capitalize().strip()})')
        elif 'chases' in condition:
            # condition = condition.replace('the', '')
            # condition = condition.strip()
            condition_list = condition.split('chases')
            noun = condition_list[0]
            adj = condition_list[1]
            noun = replace_token(noun, 'the', '')
            adj = replace_token(adj, 'the', '')
            conditions_logic_list.append(f'Chase({noun.capitalize().strip()}, {adj.capitalize().strip()})')
        elif 'chase' in condition:
            # condition = condition.replace('the', '')
            # condition = condition.strip()
            condition_list = condition.split('chase')
            noun = condition_list[0]
            adj = condition_list[1]
            noun = replace_token(noun, 'the', '')
            adj = replace_token(adj, 'the', '')
            conditions_logic_list.append(f'Chase({noun.capitalize().strip()}, {adj.capitalize().strip()})')
        else:   # verb
            verb = find_verb(condition)
            # tokens = word_tokenize(condition)
            # tagged_words = pos_tag(tokens)
            # print(condition)
            # print(tagged_words)
            # verb = [word[0] for word in tagged_words if word[1].startswith('VB')][0]
            condition_list = condition.split(verb)
            # print(condition_list)
            noun_1 = condition_list[0]
            noun_2 = condition_list[1]
            noun_1 = replace_token(noun_1, 'the', '')
            noun_2 = replace_token(noun_2, 'the', '')
            verb_lemma = lemmatizer.lemmatize(verb, pos='v').strip()
            conditions_logic_list.append(f'{verb_lemma.capitalize()}({noun_1.capitalize().strip()}, {noun_2.capitalize().strip()})')
    return conditions_logic_list




#%%

def ablation_inference_prompt_formulate_nl(rule_nl:str, condition_list:list):
    if len(condition_list) > 0:
        facts = list_2_str(condition_list) 
    else:
        facts = "[no facts avaiable].\n"
    prompt = f"""Task: please answer whether the following rule is satisfied under the provided given Facts?
If yes, say 'yes', if no, say 'no',
If it is satisfied, please answer, what does it produce.
Rule: {rule_nl},
Facts: {facts}Please answer with the following format:
Answer: [yes or no]
Procude: [your answer here]
Output:"""
    return prompt



def ablation_inference_prompt_formulate_logic(rule_logic:str, condition_logic_list:list):
    if len(condition_logic_list) > 0:
        facts = list_2_str(condition_logic_list) 
    else:
        facts = "[no facts avaiable].\n"
    prompt = f"""Task: please answer whether the following rule is satisfied under the provided given Facts?
If yes, say 'yes', if no, say 'no',
If it is satisfied, please answer, what does it produce.
Rule: {rule_logic}
Facts: {facts}Please answer with the following format:
Answer: [yes or no]
Procude: [your answer here]
Output:"""
    return prompt


def ablation_inference_prompt_formulate_logic(rule_logic:str, condition_logic_list:list):
    if len(condition_logic_list) > 0:
        facts = list_2_str(condition_logic_list) 
    else:
        facts = "[no facts avaiable].\n"
    prompt = f"""Task: please answer whether the following rule is satisfied under the provided given Facts?
The rule is written in the IF <condition> THEN <concluision> form,
If all of the conditions of a rule can be found in the list of facts, the rule can be satisfied.
If the rule is satisfied and tell me the produce, say 'yes', if no, say 'no',
If it is be satisfied, please answer, what does it produce.
Rule: {rule_logic}
Facts: {facts}Please answer with the following format:
Answer: [yes or no]
Procude: [your answer here]
Output:"""
    return prompt


#%%
def ablation_inference_extract_result(prompt:str, response:str) -> str:
    result = {
        'answer'  : "",
        'produce' : "",
    }
    if prompt in response:
        response = response.replace(prompt, '')
    response = response.strip()
    if '\n\n' in response:
        response_list = response.split('\n\n')
        response_list = [response for response in response_list if 'Answer:' in response and 'Produce:' in response]
        if len(response_list) < 1:
            return result
        else: 
            response = response_list[0]
    responses = response.split('\n')
    for respond in responses:
        if 'Answer:' in respond:
            # print('Answer:', respond)
            result['answer'] = respond.replace('Answer:', '').strip()
        if 'Produce:' in respond:
            # print('Produce:', respond)
            result['produce'] = respond.replace('Produce:', '').strip()
    
    return result


#%% Inference experiment: process rule + experiment


def ablation_inference_positive(rule_nl:str):
    result = {
        'pos_num_condition'         : 0, 
        'pos_nl_answer_correct'            : 0, 
        'pos_nl_produce_correct'            : 0, 
        'pos_logic_answer_correct'         : 0,
        'pos_logic_produce_correct'         : 0,
        'pos_nl_condition'          : '',
        'pos_nl_conclusion'         : '',
        'pos_nl_answer_answer'      : '',
        'pos_nl_answer_produce'     : '',
        'pos_logic_condition'       : '',
        'pos_logic_conclusion'      : '',
        'pos_logic_answer_answer'   : '',
        'pos_logic_answer_produce'  : '',
    } 
    rule_condition_conclusion_dict = ablation_inference_parser_rule_nl(rule_nl)
    condition_list = rule_condition_conclusion_dict['condition_list']
    conclusion_str = rule_condition_conclusion_dict['conclusion_str']
    
    
    result['pos_num_condition']     = len(condition_list)
    result['pos_nl_condition']      = code.number_list_to_str(condition_list)
    result['pos_nl_conclusion']     = conclusion_str

    # nl
    prompt_nl         = ablation_inference_prompt_formulate_nl(rule_nl=rule_nl, condition_list=condition_list)   
    answer_nl         = ask(prompt_nl)
    answer_nl_parsed  = ablation_inference_extract_result(prompt_nl, answer_nl)                   # split with output
    result['pos_nl_answer_answer']      = answer_nl_parsed['answer']
    result['pos_nl_answer_produce']     = answer_nl_parsed['produce']

    if answer_nl_parsed['answer'].lower() == 'yes':
        result['pos_nl_answer_correct'] += 1
    if answer_nl_parsed['produce'].lower() == conclusion_str.lower():
        result['pos_nl_produce_correct'] += 1

    # logic
    condition_logic_list = ablation_inference_parser_rule_logic(condition_list)
    conclusion_logic_str = ablation_inference_parser_rule_logic([conclusion_str])[0]
    result['pos_logic_condition']     = code.number_list_to_str(condition_logic_list)
    result['pos_logic_conclusion']    = code.number_list_to_str(conclusion_logic_str)
    prompt_logic          = ablation_inference_prompt_formulate_logic(rule_logic=conclusion_logic_str, condition_logic_list=condition_logic_list)
    answer_logic          = ask(prompt_logic)
    answer_logic_parsed   = ablation_inference_extract_result(prompt_logic, answer_logic)
    result['pos_logic_answer_answer']      = answer_logic_parsed['answer']
    result['pos_logic_answer_produce']     = answer_logic_parsed['produce']

    if answer_logic_parsed['answer'].lower() == 'yes':
        result['pos_logic_answer_correct'] += 1
    if answer_logic_parsed['produce'].lower() == conclusion_logic_str.lower():
        result['pos_logic_produce_correct'] += 1

    return result


def ablation_inference_negative(rule_nl:str):
    # always one condition fehlt, rule should never run
    result = {
        'neg_num_condition'         : 0, 
        'neg_nl_answer_correct'     : 0, 
        'neg_nl_produce_correct'    : 0, 
        'neg_logic_answer_correct'  : 0,
        'neg_logic_produce_correct' : 0,
        'neg_nl_condition'          : '',
        'neg_nl_conclusion'         : '',
        'neg_nl_answer_answer'      : '', 
        'neg_nl_answer_produce'     : '', 
        'neg_logic_condition'       : '',
        'neg_logic_conclusion'      : '',
        'neg_logic_answer_answer'   : '',
        'neg_logic_answer_produce'  : '',
    } 
    rule_condition_conclusion_dict = ablation_inference_parser_rule_nl(rule_nl)
    condition_list = rule_condition_conclusion_dict['condition_list']
    conclusion_str = rule_condition_conclusion_dict['conclusion_str']
    result['neg_num_condition']     = len(condition_list)
    result['neg_nl_condition']      = code.number_list_to_str(condition_list)
    result['neg_nl_conclusion']     = conclusion_str

    # nl
    prompt_nl         = ablation_inference_prompt_formulate_nl(rule_nl=rule_nl, condition_list=condition_list[1:])   
    answer_nl         = ask(prompt_nl)
    answer_nl_parsed  = ablation_inference_extract_result(prompt_nl, answer_nl)
    result['neg_nl_answer_answer']      = answer_nl_parsed['answer']
    result['neg_nl_answer_produce']     = answer_nl_parsed['produce']
    if answer_nl_parsed['answer'].lower() == 'no':
        result['neg_nl_answer_correct'] += 1
    if answer_nl_parsed['produce'].lower() == conclusion_str.lower():
        result['neg_nl_produce_correct'] += 1

    # logic
    condition_logic_list = ablation_inference_parser_rule_logic(condition_list)
    conclusion_logic_str = ablation_inference_parser_rule_logic([conclusion_str])[0]
    result['neg_logic_condition']     = code.number_list_to_str(condition_logic_list)
    result['neg_logic_conclusion']    = code.number_list_to_str(conclusion_logic_str)
    prompt_logic          = ablation_inference_prompt_formulate_logic(rule_logic=conclusion_logic_str, condition_logic_list=condition_logic_list[1:])
    answer_logic          = ask(prompt_logic)
    answer_logic_parsed   = ablation_inference_extract_result(prompt_logic, answer_logic)
    result['neg_logic_answer_answer']      = answer_logic_parsed['answer']
    result['neg_logic_answer_produce']     = answer_logic_parsed['produce']
    if answer_logic_parsed['answer'].lower() == 'no':
        result['neg_logic_answer_correct'] += 1
    if answer_logic_parsed['produce'].lower() == conclusion_logic_str.lower():
        result['neg_logic_produce_correct'] += 1

    return result

def ablation_inference_empty(rule_nl:str):
    # no conditions
    result = {
        'empty_nl_answer_correct'       : 0, 
        'empty_nl_produce_correct'      : 0, 
        'empty_logic_answer_correct'    : 0,
        'empty_logic_produce_correct'   : 0,
        'empty_nl_conclusion'           : '',
        'empty_nl_answer_answer'        : '', 
        'empty_nl_answer_produce'       : '', 
        'empty_logic_conclusion'        : '',
        'empty_logic_answer_answer'     : '',
        'empty_logic_answer_produce'    : '',
    } 
    rule_condition_conclusion_dict = ablation_inference_parser_rule_nl(rule_nl)
    conclusion_str = rule_condition_conclusion_dict['conclusion_str']
    result['empty_nl_conclusion']     = conclusion_str
    # nl
    prompt_nl         = ablation_inference_prompt_formulate_nl(rule_nl=rule_nl, condition_list=[])   
    answer_nl         = ask(prompt_nl)
    answer_nl_parsed  = ablation_inference_extract_result(prompt_nl, answer_nl)
    result['empty_nl_answer_answer']      = answer_nl_parsed['answer']
    result['empty_nl_answer_produce']     = answer_nl_parsed['produce']
    if answer_nl_parsed['answer'].lower() == 'no':
        result['empty_nl_answer_correct'] += 1
    if answer_nl_parsed['produce'].lower() == conclusion_str.lower():
        result['empty_nl_produce_correct'] += 1

    # logic
    conclusion_logic_str = ablation_inference_parser_rule_logic([conclusion_str])[0]
    result['empty_logic_conclusion']     = conclusion_str

    prompt_logic          = ablation_inference_prompt_formulate_logic(rule_logic=conclusion_logic_str, condition_logic_list=[])
    answer_logic          = ask(prompt_logic)
    answer_logic_parsed   = ablation_inference_extract_result(prompt_logic, answer_logic)
    result['empty_logic_answer_answer']      = answer_logic_parsed['answer']
    result['empty_logic_answer_produce']     = answer_logic_parsed['produce']
    if answer_logic_parsed['answer'].lower() == 'no':
        result['empty_logic_answer_correct'] += 1
    if answer_logic_parsed['produce'].lower() == conclusion_logic_str.lower():
        result['empty_logic_produce_correct'] += 1

    return result


#%% inference experiment only inference


def ablation_inference_positive_nl(rule_nl:str, condition_list:list, conclusion_str:str, condition_logic_list:list, conclusion_logic_str:str):
    # nl
    prompt_nl         = ablation_inference_prompt_formulate_nl(rule_nl=rule_nl, condition_list=condition_list)   
    answer_nl         = ask(prompt_nl)
    answer_nl_parsed  = ablation_inference_extract_result(prompt_nl, answer_nl)                   # split with output
    pos_nl_answer_answer      = answer_nl_parsed['answer']
    pos_nl_answer_produce     = answer_nl_parsed['produce']

    pos_nl_answer_correct = 0
    pos_nl_produce_correct = 0
    pos_logic_answer_correct = 0
    pos_logic_produce_correct = 0

    if answer_nl_parsed['answer'].lower() == 'yes':
        pos_nl_answer_correct = 1
    if answer_nl_parsed['produce'].lower() == conclusion_str.lower():
        pos_nl_produce_correct = 1

    # logic
    prompt_logic          = ablation_inference_prompt_formulate_logic(rule_logic=conclusion_logic_str, condition_logic_list=condition_logic_list)
    answer_logic          = ask(prompt_logic)
    answer_logic_parsed   = ablation_inference_extract_result(prompt_logic, answer_logic)
    pos_logic_answer_answer      = answer_logic_parsed['answer']
    pos_logic_answer_produce     = answer_logic_parsed['produce']

    if answer_logic_parsed['answer'].lower() == 'yes':
        pos_logic_answer_correct = 1
    if answer_logic_parsed['produce'].lower() == conclusion_logic_str.lower():
        pos_logic_produce_correct = 1

    return pos_nl_answer_answer, pos_nl_answer_produce, pos_nl_answer_correct, pos_nl_produce_correct,pos_logic_answer_answer,pos_logic_answer_produce,pos_logic_answer_correct,pos_logic_produce_correct



def ablation_inference_empty_direct(rule_nl:str, condition_list:list, conclusion_str:str, condition_logic_list:list, conclusion_logic_str:str):
    # no conditions
    empty_nl_answer_correct         = 0
    empty_nl_produce_correct        = 0
    empty_logic_answer_correct      = 0
    empty_logic_produce_correct     = 0
    # nl
    prompt_nl         = ablation_inference_prompt_formulate_nl(rule_nl=rule_nl, condition_list=[])   
    answer_nl         = ask(prompt_nl)
    answer_nl_parsed  = ablation_inference_extract_result(prompt_nl, answer_nl)
    empty_nl_answer_answer      = answer_nl_parsed['answer']
    empty_nl_answer_produce     = answer_nl_parsed['produce']
    if answer_nl_parsed['answer'].lower() == 'no':
        empty_nl_answer_correct = 1
    if answer_nl_parsed['produce'].lower() == conclusion_str.lower():
        empty_nl_produce_correct = 1

    # logic

    prompt_logic          = ablation_inference_prompt_formulate_logic(rule_logic=conclusion_logic_str, condition_logic_list=[])
    answer_logic          = ask(prompt_logic)
    answer_logic_parsed   = ablation_inference_extract_result(prompt_logic, answer_logic)
    empty_logic_answer_answer      = answer_logic_parsed['answer']
    empty_logic_answer_produce     = answer_logic_parsed['produce']
    if answer_logic_parsed['answer'].lower() == 'no':
        empty_logic_answer_correct += 1
    if answer_logic_parsed['produce'].lower() == conclusion_logic_str.lower():
        empty_logic_produce_correct += 1

    return empty_nl_answer_answer, empty_nl_answer_produce, empty_nl_answer_correct, empty_nl_produce_correct, empty_logic_answer_answer, empty_logic_answer_produce, empty_logic_answer_correct, empty_logic_produce_correct


def ablation_inference_negative_direct_nl(rule_nl:str, condition_list:list, conclusion_str:str, condition_logic_list:list, conclusion_logic_str:str):
    # always one condition fehlt, rule should never run
    neg_nl_answer_correct   = 0
    neg_nl_produce_correct  = 0
    neg_logic_answer_correct    = 0
    neg_logic_produce_correct   = 0

    # nl
    prompt_nl         = ablation_inference_prompt_formulate_nl(rule_nl=rule_nl, condition_list=condition_list[1:])   
    answer_nl         = ask(prompt_nl)
    answer_nl_parsed  = ablation_inference_extract_result(prompt_nl, answer_nl)
    neg_nl_answer_answer      = answer_nl_parsed['answer']
    neg_nl_answer_produce     = answer_nl_parsed['produce']
    if answer_nl_parsed['answer'].lower() == 'no':
        neg_nl_answer_correct += 1
    if answer_nl_parsed['produce'].lower() == conclusion_str.lower():
        neg_nl_produce_correct += 1

    # logic
    prompt_logic          = ablation_inference_prompt_formulate_logic(rule_logic=conclusion_logic_str, condition_logic_list=condition_logic_list[1:])
    answer_logic          = ask(prompt_logic)
    answer_logic_parsed   = ablation_inference_extract_result(prompt_logic, answer_logic)
    neg_logic_answer_answer      = answer_logic_parsed['answer']
    neg_logic_answer_produce     = answer_logic_parsed['produce']
    if answer_logic_parsed['answer'].lower() == 'no':
        neg_logic_answer_correct = 1
    if answer_logic_parsed['produce'].lower() == conclusion_logic_str.lower():
        neg_logic_produce_correct = 1

    return neg_nl_answer_answer, neg_nl_answer_produce, neg_nl_answer_correct, neg_nl_produce_correct, neg_logic_answer_answer, neg_logic_answer_produce, neg_logic_answer_correct, neg_logic_produce_correct



#%% pandas help functions
def ablation_inference_positive_nl_apply(df):
    return ablation_inference_positive_nl(
        df['rule'],
        df['pos_nl_condition'].split(';'),
        df['pos_nl_conclusion'],
        df['pos_logic_condition'].split(';'),
        df['pos_logic_conclusion'],
    )


def pandas_split_list(string:str):
    return string.split(';')
