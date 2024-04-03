import random
import json
import re
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from llmAPI import GLM

temperature = 0
glm_client = GLM.myapi(temperature=temperature)

# Prompt模板库
prompt_templates = [
    "Please select the correct answer for the following USMLE question: \n\n{question}\n\nOptions:\n{options}\n\nYour answer (only provide the letter, e.g., 'A'):",
    "Given the following USMLE question and options, choose the best answer:\n\n{question}\n\n{options}\n\nWhat is the correct answer? Respond with only the letter (A, B, C, D, or E):",
    "Analyze the USMLE question below and select the most appropriate answer from the provided options:\n\n{question}\n\nChoices:\n{options}\n\nAnswer (letter only):",
    "Here is a USMLE question. Review the question and options, then provide the letter corresponding to the correct answer.\n\n{question}\n\n{options}\n\nCorrect answer (letter only):"
]

def shuffle_options(options):
    options_list = list(options.items())
    random.shuffle(options_list)
    return dict(options_list)

def format_options(options):
    return "\n".join([f"{key}: {value}" for key, value in options.items()])

def extract_answer(llm_response):
    '''
    this function should be optimize if the model is already fine-tuned with response with certain formats and is inclined to give the answer in a limited range of formats
    
    '''
    match = re.search(r'[A-E]', llm_response, re.IGNORECASE)
    if match:
        return match.group(0).upper()
    else:
        return None

def test_question(question_data):
    correct_answer = question_data["answer_idx"]
    results = []

    for _ in range(5):
        prompt_template = random.choice(prompt_templates)
        shuffled_options = shuffle_options(question_data["options"])

        prompt = prompt_template.format(
            question=question_data["question"],
            options=format_options(shuffled_options)
        )

        llm_response = glm_client.predict(prompt)
        extracted_answer = extract_answer(llm_response)

        is_correct = False
        if extracted_answer:
            is_correct = shuffled_options[extracted_answer] == question_data["options"][correct_answer]

        results.append({
            "prompt": prompt,
            "llm_response": llm_response,
            "extracted_answer": extracted_answer,
            "is_correct": is_correct
        })

    accuracy = sum(1 for result in results if result["is_correct"]) / len(results)
    return accuracy, results




if __name__ == "__main__":
    # 读取问题数据
    with open("questions.json", "r") as f:
        questions = json.load(f)

    results = []
    for question_data in questions:
        accuracy, question_results = test_question(question_data)
        results.append({
            "question": question_data["question"],
            "accuracy": accuracy,
            "results": question_results
        })

        # 保存结果到文件
        with open("5voteresults.jsonl", "a") as f:
            json.dump(results, f, indent=4)
            f.write("\n")