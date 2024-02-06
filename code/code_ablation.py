# version: 1.11

from code import list_2_str
import code
import copy
#%% Facts

def prompt_formulator_translation_facts_2_adj_0_example(facts:list) -> str:
    prompt = f"""Task: 
please write the following Facts in the form as the following format,
The sentence "<Person> is <Property>" should be written as "<Property>(<Person>)".
Do not write '<>'. First letter of property please in capital.
Facts: 
{list_2_str(facts)}Please only give me the answer.
Output:
"""
    return prompt


def prompt_formulator_translation_facts_2_adj_2_example(facts:list) -> str:
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

def prompt_formulator_translation_facts_2_adj_3_example(facts:list) -> str:
    prompt = f"""Task: 
please write the following Facts in the form as the following format,
The sentence "<Person> is <Property>" should be written as "<Property>(<Person>)".
Example: 
-Sentence: "Bob is good" should be written as "Good(Bob)". 
-Sentence: "Bob needs apple" should be written as "Need(Bob, Apple)".
-Sentence: "The cat likes the apple" should be written as "Like(Cat, Apple)".
Example is not a part of task sentences.
Do not write '<>'. First letter of property please in capital.
Facts: 
{list_2_str(facts)}Please only give me the answer.
Output:
"""
    return prompt


#%% Rules

def prompt_formulator_translation_rules_2_ifthen_0_example(rules:list) -> str:
    prompt = f"""Task: 
please convert the following Rules in the form as IF-THEN expression,
The sentence "If <Person> is <Property>, then <Person> is <Property>" should be written as "IF <Property>(<Person>), THEN <Property>(<Person>)".
The sentence "All <Property> <Object> are <Property>" should be written as "IF <Property>(<Object>), THEN <Property>(<Object>)".
Sentences: 
{list_2_str(rules)}Output:
"""
    return prompt

def prompt_formulator_translation_rules_2_ifthen_1_example(rules:list) -> str:
    prompt = f"""Task: 
please convert the following Rules in the form as IF-THEN expression,
The sentence "If <Person> is <Property>, then <Person> is <Property>" should be written as "IF <Property>(<Person>), THEN <Property>(<Person>)".
The sentence "All <Property> <Object> are <Property>" should be written as "IF <Property>(<Object>), THEN <Property>(<Object>)".
Example: Sentence: If Bob likes dog, then Bob is nice, output: IF Like(Bob, dog) THEN Nice(Bob)
Sentences: 
{list_2_str(rules)}Output:
"""
    return prompt

def prompt_formulator_translation_rules_2_ifthen_2_example(rules:list) -> str:
    prompt = f"""Task: 
please convert the following Rules in the form as IF-THEN expression,
The sentence "If <Person> is <Property>, then <Person> is <Property>" should be written as "IF <Property>(<Person>), THEN <Property>(<Person>)".
The sentence "All <Property> <Object> are <Property>" should be written as "IF <Property>(<Object>), THEN <Property>(<Object>)".
Example: Sentence: If Bob likes dog, then Bob is nice, output: IF Like(Bob, dog) THEN Nice(Bob)
Example: Sentence: All cute dog are nice, output: IF Cute(dog) THEN Nice(dog)
Sentences: 
{list_2_str(rules)}Output:
"""
    return prompt

def prompt_formulator_translation_rules_2_ifthen_3_example(rules:list) -> str:
    prompt = f"""Task: 
please convert the following Rules in the form as IF-THEN expression,
The sentence "If <Person> is <Property>, then <Person> is <Property>" should be written as "IF <Property>(<Person>), THEN <Property>(<Person>)".
The sentence "All <Property> <Object> are <Property>" should be written as "IF <Property>(<Object>), THEN <Property>(<Object>)".
Example: Sentence: If Bob likes dog, then Bob is nice, output: IF Like(Bob, dog) THEN Nice(Bob)
Example: Sentence: All cute dog are nice, output: IF Cute(dog) THEN Nice(dog)
Example: Sentence: Good people are nice, output: IF Good(people) THEN Nice(people)
Sentences: 
{list_2_str(rules)}Output:
"""
    return prompt

def prompt_formulator_translation_rules_2_ifthen_4_example(rules:list) -> str:
    prompt = f"""Task: 
please convert the following Rules in the form as IF-THEN expression,
The sentence "If <Person> is <Property>, then <Person> is <Property>" should be written as "IF <Property>(<Person>), THEN <Property>(<Person>)".
The sentence "All <Property> <Object> are <Property>" should be written as "IF <Property>(<Object>), THEN <Property>(<Object>)".
Example: Sentence: If Bob is good and cute, then Bob is nice, output: IF Good(Bob) AND Cute(Bob) THEN Nice(Bob)
Example: Sentence: If Bob likes dog, then Bob is nice, output: IF Like(Bob, dog) THEN Nice(Bob)
Example: Sentence: All cute dog are nice, output: IF Cute(dog) THEN Nice(dog)
Example: Sentence: Good people are nice, output: IF Good(people) THEN Nice(people)
Sentences: 
{list_2_str(rules)}Output:
"""
    return prompt

def prompt_formulator_translation_rules_2_ifthen_5_example(rules:list) -> str:
    prompt = f"""Task: 
please convert the following Rules in the form as IF-THEN expression,
The sentence "If <Person> is <Property>, then <Person> is <Property>" should be written as "IF <Property>(<Person>), THEN <Property>(<Person>)".
The sentence "All <Property> <Object> are <Property>" should be written as "IF <Property>(<Object>), THEN <Property>(<Object>)".
Example: Sentence: If Bob is good and cute, then Bob is nice, output: IF Good(Bob) AND Cute(Bob) THEN Nice(Bob)
Example: Sentence: If Bob likes dog, then Bob is nice, output: IF Like(Bob, dog) THEN Nice(Bob)
Example: Sentence: All cute dog are nice, output: IF Cute(dog) THEN Nice(dog)
Example: Sentence: All cute, rough cats are nice, output: IF Cute(cat) AND Rough(cat) THEN Nice(dog)
Example: Sentence: Good people are nice, output: IF Good(people) THEN Nice(people)
Sentences: 
{list_2_str(rules)}Output:
"""
    return prompt

def prompt_formulator_translation_rules_2_ifthen_6_example(rules:list) -> str:
    prompt = f"""Task: 
please convert the following Rules in the form as IF-THEN expression,
The sentence "If <Person> is <Property>, then <Person> is <Property>" should be written as "IF <Property>(<Person>), THEN <Property>(<Person>)".
The sentence "All <Property> <Object> are <Property>" should be written as "IF <Property>(<Object>), THEN <Property>(<Object>)".
Example: Sentence: If Bob is good, then Bob is nice, output: IF Good(Bob) THEN Nice(Bob)
Example: Sentence: If Bob is good and cute, then Bob is nice, output: IF Good(Bob) AND Cute(Bob) THEN Nice(Bob)
Example: Sentence: If Bob likes dog, then Bob is nice, output: IF Like(Bob, dog) THEN Nice(Bob)
Example: Sentence: All cute dog are nice, output: IF Cute(dog) THEN Nice(dog)
Example: Sentence: All cute, rough cats are nice, output: IF Cute(cat) AND Rough(cat) THEN Nice(dog)
Example: Sentence: Good people are nice, output: IF Good(people) THEN Nice(people)
Sentences: 
{list_2_str(rules)}Output:
"""
    return prompt

#%% others
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

#%% Evaluation
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

def find_solution_fact(fact:str) -> str:
    lemmatizer = WordNetLemmatizer()
    solution = ""
    fact = fact.replace('.', '')
    fact = fact.strip()
    if 'is' in fact:
        words = fact.split('is')
        processed_words = []
        # remove The/the
        for word in words:
            word = word.strip()
            word = word.replace('The', '')
            word = word.strip()
            word = word.replace('the', '')
            word = word.strip()
            processed_words.append(word)
        print(processed_words)
        if len(processed_words) < 2:
            return 'error'
        else:
            solution = f"{processed_words[1]}({processed_words[0]})"
    else:   # verb
        tokens = word_tokenize(fact)
        tagged_words = pos_tag(tokens)
        verb = [word[0] for word in tagged_words if word[1].startswith('VB')][0]
        verb_lemma = lemmatizer.lemmatize(verb, pos='v')

        words = fact.split(verb)
        processed_words = []
        for word in words:
            word = word.strip()
            word = word.replace('The', '')
            word = word.strip()
            word = word.replace('the', '')
            word = word.strip()
            processed_words.append(word)
        print(processed_words)
        if len(processed_words) < 2:
            return 'error'
        else:
            solution = f"{verb_lemma}({processed_words[0]}, {processed_words[1]})"
    return solution

def evaluation_translation_fact(facts_list:list, answers_list:list):
    # how many examples in answer?
    # how many are correct?
    answers_list_lower = [answer.lower() for answer in answers_list]
    num_correct = 0
    facts_no_answer = []
    for fact in facts_list:
        solution = find_solution_fact(fact)
        if solution.lower() in answers_list_lower:
            num_correct += 1
            answers_list_lower.remove(solution.lower())
        else:
            facts_no_answer.append(fact)
    bad_answers = [answer for answer in answers_list if answer.lower() in answers_list_lower]
    num_questions_without_answer = len(facts_no_answer)
    num_bad_answers = len(bad_answers)

    str_facts_no_answer = ""
    for index, fact in enumerate(facts_no_answer):
        str_facts_no_answer += f"{index+1}. {fact}\n"

    str_bad_answers     = ""
    for index, fact in enumerate(bad_answers):
        str_bad_answers += f"{index+1}. {fact}\n"

    result_dict = {}
    result_dict['num_all']                       = len(facts_list)
    result_dict['num_correct']                   = num_correct
    result_dict['num_questions_without_answer']  = num_questions_without_answer
    result_dict['num_bad_answers']               = num_bad_answers
    result_dict['str_facts_no_answer']           = str_facts_no_answer
    result_dict['str_bad_answers']               = str_bad_answers
    return result_dict


#%% evaluate rules


def rules_parse_answer(answer:str) -> dict:
    # convert answer to {condition:{set}, conclusion:{set}}
    if 'if' not in answer.lower() or 'then' not in answer.lower():
        return {answer:answer}
    else: 
        words = answer.lower().split('then')    # ['if xxx'] then ['xxx']
        if len(words) < 2:
            return {answer:answer}
        return find_solution_rules(words[0], words[1])

def rules_parse_rule(rule:str) -> dict:
    # convert rule sentence to {condition:{set}, conclusion:{set}}
    # can't find solution for 'are', or 'all'
    rule_lower = rule.lower()
    rule_lower = rule_lower.replace('all', '')
    rule_lower = rule_lower.replace('if', '')
    rule_lower = rule_lower.replace('is', '')
    rule_lower = rule_lower.replace('.', '')
    rule_lower = rule_lower.replace(',', '')
    if 'are' in rule_lower:
        # words = rule_lower.split('are')
        # if len(words) < 2:
        return {rule:rule}
    elif 'then' in rule_lower:
        words = rule_lower.split('then')
        if len(words) < 2:
            return {rule:rule}
    else: 
        return {rule:rule}
    return find_solution_rules(words[0], words[1])


def find_solution_rules(str_conditions:str, str_conclusion:str) -> dict:
    lemmatizer = WordNetLemmatizer()
    result = {
        'conditions'    : [],
        'conclusion'    : [],
    }
    str_conditions = str_conditions.strip()
    str_conclusion = str_conclusion.strip()
    

    # process condition
    str_conditions = str_conditions.replace('(', ' ')
    str_conditions = str_conditions.replace(')', ' ')
    str_conditions = str_conditions.replace('if', '')
    str_conditions = str_conditions.replace('and', '')
    str_conditions = str_conditions.replace('is', '')
    str_conditions = str_conditions.replace(',', '')

    str_conditions = str_conditions.strip()

    tokens = word_tokenize(str_conditions)
    tagged_words = pos_tag(tokens)
    nouns = [word[0] for word in tagged_words if word[1].startswith('NN')]
    nouns = list(set(nouns))
    adjs  = [word[0] for word in tagged_words if word[1].startswith('JJ')]
    adjs  = list(set(adjs))
    verbs = [word[0] for word in tagged_words if word[1].startswith('VB') and word[0] not in ['is', 'are']]
    verbs = list(set(verbs))
    nouns = [lemmatizer.lemmatize(word, pos='n') for word in nouns]
    adjs  = [lemmatizer.lemmatize(word, pos='a') for word in adjs ]
    verbs = [lemmatizer.lemmatize(word, pos='v') for word in verbs]
    result['conditions'].extend(nouns)
    result['conditions'].extend(adjs)
    result['conditions'].extend(verbs)

    # process conclusion
    str_conclusion = str_conclusion.replace('(', ' ')
    str_conclusion = str_conclusion.replace(')', ' ')
    str_conclusion = str_conclusion.replace('if', '')
    str_conclusion = str_conclusion.replace('and', '')
    str_conclusion = str_conclusion.replace('is', '')
    str_conclusion = str_conclusion.replace('are', '')
    str_conclusion = str_conclusion.replace(',', '')
    str_conclusion = str_conclusion.strip()

    tokens = word_tokenize(str_conclusion)
    tagged_words = pos_tag(tokens)
    nouns = [word[0] for word in tagged_words if word[1].startswith('NN')]
    nouns = list(set(nouns))
    adjs  = [word[0] for word in tagged_words if word[1].startswith('JJ')]
    adjs  = list(set(adjs))
    verbs = [word[0] for word in tagged_words if word[1].startswith('VB') and word[0] not in ['is', 'are']]
    verbs = list(set(verbs))
    nouns = [lemmatizer.lemmatize(word, pos='n') for word in nouns]
    adjs  = [lemmatizer.lemmatize(word, pos='a') for word in adjs ]
    verbs = [lemmatizer.lemmatize(word, pos='v') for word in verbs]
    result['conclusion'].extend(nouns)
    result['conclusion'].extend(adjs)
    result['conclusion'].extend(verbs)

    result['conditions'] = set(result['conditions'])
    result['conclusion'] = set(result['conclusion'])
    print(result)
    return result

def evaluation_translation_rule(rules_list:list, answers_list:list):
    # how many examples in answer?
    # how many are correct?
    answer_parsed = [rules_parse_answer(answer) for answer in answers_list]
    
    num_correct = 0
    rules_no_answer = []
    rules_are = [rule for rule in rules_list if 'are' in rule]
    num_rules_are = len(rules_are)
    for rule in rules_list:
        if 'are' in rule:
            continue
        rule_parsed = rules_parse_rule(rule)
        if rule_parsed in answer_parsed:
            num_correct += 1
            index = answer_parsed.index(rule_parsed)
            answers_list.remove(answers_list[index])
            answer_parsed.remove(rule_parsed)
        else:
            rules_no_answer.append(rule)

    bad_answers = answers_list
    num_rules_no_answer = len(rules_no_answer)
    num_bad_answers = len(bad_answers)

    str_rules_no_answer = ""
    for index, rule in enumerate(rules_no_answer):
        str_rules_no_answer += f"{index+1}. {rule}\n"

    str_bad_answers     = ""
    for index, rule in enumerate(bad_answers):
        str_bad_answers += f"{index+1}. {rule}\n"

    str_are_rules     = ""
    for index, rule in enumerate(rules_are):
        str_are_rules += f"{index+1}. {rule}\n"

    result_dict = {}
    result_dict['num_all']                       = len(rules_list)
    result_dict['num_rules_without_are']         = len(rules_list) - num_rules_are
    result_dict['num_correct']                   = num_correct
    result_dict['num_rules_no_answer']           = num_rules_no_answer
    result_dict['num_bad_answers']               = num_bad_answers
    result_dict['str_rules_no_answer']           = str_rules_no_answer
    result_dict['str_bad_answers']               = str_bad_answers
    result_dict['num_rules_are']                 = num_rules_are
    result_dict['str_are_rules']                 = str_are_rules
    return result_dict


#%% modules

def ask(pipeline, tokenizer, question:str, max_length=500) -> str:
    sequences = pipeline(
        question,
        do_sample=False,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=max_length,
    )
    for seq in sequences:
        return seq['generated_text']

def formulate_fact_module(facts_raw, mode_fact, num_example_fact, pipeline, tokenizer):
    if mode_fact in ['multi']:
        if num_example_fact in ['0']:
            prompt_translate_facts  = prompt_formulator_translation_facts_2_adj_0_example(facts_raw)
        if num_example_fact in ['2']:
            prompt_translate_facts  = prompt_formulator_translation_facts_2_adj_2_example(facts_raw)
        if num_example_fact in ['3']:
            prompt_translate_facts  = prompt_formulator_translation_facts_2_adj_3_example(facts_raw)
        formulated_facts_raw        = ask(pipeline, tokenizer, prompt_translate_facts, max_length=600)
        formulated_facts_list       = code.result_extractor_translation_fact_2_adj(formulated_facts_raw)
        return formulated_facts_list
    elif mode_fact in ['single']:
        formulated_facts_list = []
        for fact in facts_raw:
            if num_example_fact in ['0']:
                prompt_translate_facts  = prompt_formulator_translation_facts_2_adj_0_example([fact])
            if num_example_fact in ['2']:
                prompt_translate_facts  = prompt_formulator_translation_facts_2_adj_2_example([fact])
            if num_example_fact in ['3']:
                prompt_translate_facts  = prompt_formulator_translation_facts_2_adj_3_example([fact])
            formulated_facts_raw        = ask(pipeline, tokenizer, prompt_translate_facts, max_length=600)
            formulated_facts_list.extend([formulated_fact for formulated_fact in code.result_extractor_translation_fact_2_adj(formulated_facts_raw) if formulated_fact not in formulated_facts_list])
        return formulated_facts_list
    else:
        return []

def translation_select_rule(rules_raw, num_example_rule):
    prompt_translate_rules = ''
    if num_example_rule in ['0']:
        prompt_translate_rules  = prompt_formulator_translation_rules_2_ifthen_0_example(rules_raw)
    elif num_example_rule in ['1']:
        prompt_translate_rules  = prompt_formulator_translation_rules_2_ifthen_1_example(rules_raw)
    elif num_example_rule in ['2']:
        prompt_translate_rules  = prompt_formulator_translation_rules_2_ifthen_2_example(rules_raw)
    elif num_example_rule in ['3']:
        prompt_translate_rules  = prompt_formulator_translation_rules_2_ifthen_3_example(rules_raw)
    elif num_example_rule in ['4']:
        prompt_translate_rules  = prompt_formulator_translation_rules_2_ifthen_4_example(rules_raw)
    elif num_example_rule in ['5']:
        prompt_translate_rules  = prompt_formulator_translation_rules_2_ifthen_5_example(rules_raw)
    elif num_example_rule in ['6']:
        prompt_translate_rules  = prompt_formulator_translation_rules_2_ifthen_6_example(rules_raw)
    return prompt_translate_rules

def formulate_rule_module(rules_raw, mode_rule, num_example_rule, pipeline, tokenizer):
    if mode_rule in ['multi']:
        prompt_translate_rules  = translation_select_rule(rules_raw, num_example_rule)
        formulated_rules_raw    = ask(pipeline, tokenizer, prompt_translate_rules, max_length=700)
        formulated_rules_list   = code.result_extractor_translation_rules_2_ifthen(formulated_rules_raw)
        return formulated_rules_list
    elif mode_rule in ['single']:
        formulated_rules_list   = []
        for rule in rules_raw:
            prompt_translate_rules = translation_select_rule([rule], num_example_rule)
            formulated_rules_raw    = ask(pipeline, tokenizer, prompt_translate_rules, max_length=700)
            formulated_rules_list.extend([formulated_rule for formulated_rule in code.result_extractor_translation_rules_2_ifthen(formulated_rules_raw) if formulated_rule not in formulated_rules_list])
        return formulated_rules_list
    else:
        return []

def inference_module(formulated_facts_list, formulated_rules_list, mode_inference, pipeline, tokenizer):
    resulted_new_facts_list =  copy.deepcopy(formulated_facts_list)
    if mode_inference in ['multi']:
        prompt_facts_rules      = code.formulate_facts_rules_2_str(facts=formulated_facts_list, rules=formulated_rules_list)
        prompt_query            = code.prompt_formulator_query_ifthen_multiQ(formulated_facts_rules=prompt_facts_rules)
        result_query_raw        = ask(pipeline, tokenizer, prompt_query, max_length=1000)
        result_query_dict       = code.query_result_extractor(result_query_raw)
        resulted_new_facts_list.extend(
            conclude.strip() for conclude in result_query_dict['Conclusion']
            if conclude.strip() not in resulted_new_facts_list
        )
        return resulted_new_facts_list
    elif mode_inference in ['single']:
        for formulated_rule in formulated_rules_list:
            prompt_facts_rules  = code.formulate_facts_rules_2_str(facts=resulted_new_facts_list, rules=[formulated_rule])
            prompt_query        = code.prompt_formulator_query_ifthen_multiQ(formulated_facts_rules=prompt_facts_rules)
            result_query_raw    = ask(pipeline, tokenizer, prompt_query, max_length=1000)
            result_query_dict   = code.query_result_extractor(result_query_raw)
            resulted_new_facts_list.extend(
                conclude.strip() for conclude in result_query_dict['Conclusion']
                if conclude.strip() not in resulted_new_facts_list
            )
        return resulted_new_facts_list
    else:
        return []