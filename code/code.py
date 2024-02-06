# version: 1.11

def list_2_str(facts:list) -> str:
    prompt = ""
    for fact in facts:
        prompt += f"*{fact}\n"
    return prompt

def prompt_formulator_translation_facts_2_adj(facts:list) -> str:
    prompt = f"""Task: 
please write the following Facts in the form as the following format,
The sentence "<Person> is <Property>" should be written as "<Property>(<Person>)".
Example: Sentence: "Bob is good" should be written as "Good(Bob)". Sentence: "Bob needs apple" should be written as "Need(Bob, Apple)".
Example is not a part of task sentences.
Do not write '<>'. First letter of property please in capital.
Facts: 
{list_2_str(facts)}Please only give me the answer.
Output:
"""
    return prompt

def result_extractor_translation_fact_2_adj(text:str) -> list:
    text = text.strip()
    text = text.split('\n\n')[0]    # remove additional information
    texts = text.split('Output:')
    texts = [text for text in texts if len(text)>1] # remove ""
    if len(texts) > 2:              # repeated question, [0]output[1]output[2], [2] is the answer
        text = texts[2].strip()
    elif len(texts) == 2:           # [0]output[1], [1] is the answer
        text = texts[1].strip()
    else:                           # no output at all
        text = texts[0].strip()

    texts = text.split('\n')
    result = []
    for text in texts:
        if '*' in text:
            result.append(text.split('*')[1].strip())
        else:
            result.append(text)
    return(result)
def prompt_formulator_translation_rules_2_ifthen(rules:list) -> str:
    prompt = f"""Task: 
please convert the following Rules in the form as IF-THEN expression,
The sentence "If <Person> is <Property>, then <Person> is <Property>" should be written as "IF <Property>(<Person>), THEN <Property>(<Person>)".
Example: Sentence: If Bob is good, then Bob is nice, output: IF Good(Bob) THEN Nice(Bob)
Sentences: 
{list_2_str(rules)}Output:
"""
    return prompt

def result_extractor_translation_rules_2_ifthen(text:str) -> list:
    text = text.strip()
    text = text.split('\n\n')[0]    # remove additional information
    texts = text.split('Output:')
    texts = [text for text in texts if len(text)>1] # remove ""
    if len(texts) > 2:              # repeated question, [0]output[1]output[2], [2] is the answer
        text = texts[2].strip()
    elif len(texts) == 2:           # [0]output[1], [1] is the answer
        text = texts[1].strip()
    else:                           # no output at all
        text = texts[0].strip()
        
    texts = text.split('\n')
    result = []
    for text in texts:
        if '*' in text:
            result.append(text.split('*')[1].strip())
        else:
            result.append(text)
    return(result)

def formulate_facts_rules_2_str(facts:list, rules:list) -> str:
    if not facts or not rules:
        print('NoneType Error formulate_facts_rules_2_str()')
        return ""
    prompt = "Provided Facts:\n"
    for fact in facts:
        prompt += f"*{fact},\n"
    prompt += "Provided Rules:\n"
    for rule in rules:
        prompt += f"*{rule},\n"
    return prompt
def formulate_facts_2_str(facts:list) -> str:
    prompt = "Provided Facts:\n"
    for fact in facts:
        prompt += f"*{fact},\n"
    return prompt
def formulate_rules_2_str(rules:list) -> str:
    prompt = "Provided Rules:\n"
    for rule in rules:
        prompt += f"*{rule},\n"
    return prompt

def prompt_formulator_query_ifthen_singleQ(formulated_facts: str, rule:str) -> str:
    prompt = f"""{formulated_facts}Provided Rules: 
* {rule}
Note:
* Consider the provided Facts are all true, and the facts which are not provided are all false,
* The syntax of 'IF-THEN' rule is: 'IF <conditions> THEN <conclusion>' 
* Only if the conditions of the Rules are satisfied, the Rules can generate output,
* Examples are not a part of facts.
Examples:
* with fact Handsome(Alice), the rule 'IF Handsome(Alice) THEN Cool(Alice)' will be fullfulled, Cool(Alice) will be concluded, since the condition 'Handsome(Alice)' is satisfied by the Fact;
* with fact Pretty(Alice), the rule 'IF Handsome(Alice) THEN Cool(Alice)' will NOT be fullfulled, Nothing will be concluded, since the condition 'Handsome(Alice)' is NOT satisfied by the Fact;
* with fact Handsome(Alice), Pretty(Alice), the rule 'IF Handsome(Alice) AND Pretty(Alice) THEN Cool(Alice)' will be fullfulled, Cool(Alice) will be concluded, since the condition 'Handsome(Alice)' and 'Pretty(Alice)' is satisfied by the Fact;
* with fact Pretty(Alice), the rule 'IF Handsome(Alice) AND Pretty(Alice) THEN Cool(Alice)' will NOT be fullfulled, Nothing will be concluded, since the condition 'Handsome(Alice)' is NOT satisfied by the Fact;
Question: 
1. Based on the provided Facts and Rules, Please tell me which facts are provided and which rules are provided?
2. Does the current provided Facts satisfy the provided Rules, Please tell me the conclusion, why? If nothing could be concluded, please answer 'NOTHING'.
Please answer with the following format:
Provided Facts: [your answer here]
Provided Rules: [your answer here]
Conclusion: [your answer here, do not provide addition information]
Output:"""
    return prompt

def query_result_extractor(text:str) -> dict:       # interence result 
    text = text.strip()
    text = text.split('\n\n')[0]    # remove additional information
    texts = text.split('Output:')
    texts = [text for text in texts if len(text) > 0] # remove ""
    if len(texts) > 2:              # repeated question, [0]output[1]output[2], [2] is the answer
        text = texts[2].strip()
    elif len(texts) == 2:           # [0]output[1], [1] is the answer
        text = texts[1].strip()
    else:                           # no output at all
        text = texts[0].strip()
    texts = text.split('\n')
    result = {
        'Provided Facts': [],   # TODO: init?
        'Provided Rules': [],   # TODO: init?
        'Conclusion'    : [],   # TODO: init?
        'Explanation'   : "",
        'New Facts'     : [],
              }
    for text in texts:
        text = text.strip()
        if 'Provided Facts' in text: 
            texts = text.split('Provided Facts:')
            if len(texts) > 1:  # normal: at least 2 parts
                text = texts[1].strip()
                texts = text.split(',')
                texts = [x.strip() for x in texts]
                result['Provided Facts'].extend(texts)
        if 'Provided Rules' in text: 
            texts = text.split('Provided Rules:')
            if len(texts) > 1:  # normal: at least 2 parts
                text = texts[1].strip()
                texts = text.split(',')
                texts = [x.strip() for x in texts]
                result['Provided Rules'].extend(texts)
        if 'Conclusion' in text: 
            texts = text.split('Conclusion:')
            if len(texts) > 1:  # normal: at least 2 parts
                text = texts[1].strip()
                texts = text.split(',')
                texts = [x.strip() for x in texts]
                result['Conclusion'].extend(texts)
        if 'Explanation' in text: 
            texts = text.split('Explanation:')
            if len(texts) > 1:  # normal: at least 2 parts
                text = texts[1].strip()
                result['Explanation'] = text
    # result['New Facts'] = result['Provided Facts'] +  [conclude for conclude in result['Conclusion'] if conclude not in result['Provided Facts']]
    # result['New Facts'] will be formulated facts + result['Conclusion'] and deduplicated
    return(result)

def answer_query(conclusions:list, formulated_query:str) -> str:
    if len(conclusions) < 1:
        return 'False'
    if formulated_query in conclusions or formulated_query.lower().strip() in [conclusion.lower().strip() for conclusion in conclusions]:
        return 'True'
    else:
        return 'False'

def prompt_formulator_query_ifthen_multiQ(formulated_facts_rules: str) -> str:
    prompt = f"""{formulated_facts_rules}Note:
* Consider the provided Facts are all true, and the facts which are not provided are all false,
* The syntax of 'IF-THEN' rule is: 'IF <conditions> THEN <conclusion>' 
* Only if the conditions of the Rules are provided, the Rules are satisfied, then the satisfied Rules can generate conclusion as output,
* Examples are not a part of facts.
Examples:
* with fact Handsome(Alice), the rule 'IF Handsome(Alice) THEN Cool(Alice)' will be fullfulled, Cool(Alice) will be concluded, since the condition 'Handsome(Alice)' is satisfied by the Fact;
* with fact Pretty(Alice), the rule 'IF Handsome(Alice) THEN Cool(Alice)' will NOT be fullfulled, Nothing will be concluded, since the condition 'Handsome(Alice)' is NOT satisfied by the Fact;
* with fact Handsome(Alice), Pretty(Alice), the rule 'IF Handsome(Alice) AND Pretty(Alice) THEN Cool(Alice)' will be fullfulled, Cool(Alice) will be concluded, since the condition 'Handsome(Alice)' and 'Pretty(Alice)' is satisfied by the Fact;
* with fact Pretty(Alice), the rule 'IF Handsome(Alice) AND Pretty(Alice) THEN Cool(Alice)' will NOT be fullfulled, Nothing will be concluded, since the condition 'Handsome(Alice)' is NOT satisfied by the Fact;
Question: 
1. Based on the provided Facts and Rules, Please tell me which facts are provided and which rules are provided?
2. Does the current provided Facts satisfy the provided Rules, Please tell me the conclusion? If nothing could be concluded, please answer 'NOTHING'.
Please answer with the following format:
Provided Facts: [your answer here]
Provided Rules: [your answer here]
Conclusion: [your answer here]
Output:"""
    return prompt

def FR_decomposer_proofwriter(facts_and_rules:str) -> dict:
    facts_and_rules_list = facts_and_rules.strip().split('.')
    facts = []
    rules = []
    for fact_or_rule in facts_and_rules_list:
        fact_or_rule = fact_or_rule.strip()
        if len(fact_or_rule) < 1:
            continue
        if 'If' in fact_or_rule or 'then' in fact_or_rule or 'if' in fact_or_rule:
            rules.append(fact_or_rule + ".")
        else:
            facts.append(fact_or_rule + ".")
    result = {
        'facts' : facts,
        'rules' : rules
    }
    return result

def result_extractor_LLM_answer_query(text:str) -> str:
    text = text.strip()
    text = text.split('\n\n')[0]    # remove additional information
    texts = text.split('Output:')
    texts = [text for text in texts if len(text)>1] # remove ""
    if len(texts) > 2:              # repeated question, [0]output[1]output[2], [2] is the answer
        text = texts[2].strip()
    elif len(texts) > 1:                           # [0]output[1], [1] is the answer
        text = texts[1].strip()
    else:                           # no output at all
        text = texts[0].strip()
    texts = text.split('\n')
    text = texts[0]
    if 'True' in text and 'False' not in text:
        return 'True' 
    if 'True' not in text and 'False' in text: 
        return 'False' 
    if 'True' in text and 'False' in text: 
        return 'Wrong'
    return 'Wrong'

def formulate_answer_query_LLM(formulated_facts: list, formulated_query:str) -> str:
    if len(formulated_query) < 1:
        facts_str = "Empty"
    else:
        facts_str = ", ".join(formulated_facts)
    prompt = f"""Context: {facts_str}
Query: was {formulated_query} mentioned in context? where is it? Please strictly compare the letters.
If yes, please output 'True', if no please output 'False'
Output:"""
    return prompt

def FR_decomposer_proofwriter_new(facts_and_rules:str) -> dict:
    facts_and_rules_list = facts_and_rules.strip().split('.')
    facts = []
    rules = []
    for fact_or_rule in facts_and_rules_list:
        fact_or_rule = fact_or_rule.strip()
        if len(fact_or_rule) < 1:
            continue
        if 'If' in fact_or_rule or 'then' in fact_or_rule or 'if' in fact_or_rule:
            rules.append(fact_or_rule + ".")
        elif 'are' in fact_or_rule:
            rules.append(fact_or_rule + ".")
        elif 'All' in fact_or_rule or 'all' in fact_or_rule:
            rules.append(fact_or_rule + ".")
        else:
            facts.append(fact_or_rule + ".")
    result = {
        'facts' : facts,
        'rules' : rules
    }
    return result

def number_list_to_str(lst:list) -> str:
    res = ''
    for index, element in enumerate(lst):
        res += f'{index+1}. {element}\n'
    return res