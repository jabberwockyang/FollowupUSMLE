from llmAPI import GPT
import json

model = GPT.from_gpt4(temperature=1)
def process(line):
    json_obj = json.loads(line)
    question = json_obj['question'] 
    options = ['\n'+k+'. '+v for k,v in json_obj['options'].items()]
    options = ''.join(options)
    return f"{question}{options}"
with open('test.jsonl') as f:
    usmle_test_list = [process(line) for line in f]


# system_prompt = 'You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture. Knowledge cutoff: 2022-01 Current date: 2023-11-07'
# example text containing an insturction and one-shot example
example_text="We are evaluating the medical ability of an LLM through USMLE test. For the questions that the LLM got wrong, we want to **extract** the basic knowledge and comprehensive application capabilities tested by the USMLE questions, and ask the LLM separately to find out what aspects of the LLM's lack of knowledge or ability caused them to get the questions wrong.**The following is an example**.<br> **input:**'A 26-year-old man comes to the office because of a 1-week history of increased urinary frequency accompanied by excessive thirst. He says he has been urinating hourly. Physical examination shows no abnormalities. Serum chemistry studies are within the reference ranges. Urine osmolality is 50 mOsmol/kg H2O. After administration of ADH (vasopressin), his urine osmolality is within the reference range. The most likely cause of this patient's symptoms is dysfunction of which of the following structures?  <br>A. Anterior pituitary gland  <br>B. Bowman capsule  <br>C. Glomerulus  <br>D. Hypophysial portal system  <br>E. Loop of Henle  <br>F. Supraoptic nucleus '<br>**output**:{'question_for_basic_knowledge':[{'question':'How to diagnse diabetes insipidus', 'classification':'endocrinology'},{'question':'How to distinguish Central diabetes insipidus and renal diabetes insipidus','classification':'endocrinology'},{'question':'What is the mechanism of diabetes insipidus','classification':'endocrinology'}],'question_for_interpretation_and_association':[{'question':'What is the diagnosis of the patient','classification':'clinical'},{'question':'Which type of diabetes insipidus does the patient have','classification':'clinical'},{'question':'What is most likely causing the symptoms of the patient','classification':'clinical'}]}"


for usmle_test in usmle_test_list:
    
    prompt=f"{example_text}<br>**input**:'{usmle_test}'<br>**output**:"
    response = model.predict(prompt)
    newobj = {'usmle_test':usmle_test, 'prompt':prompt, 'gptgene':response}

    with open('/path/to/your/usmle/followup.jsonl','a') as f:
        f.write(json.dumps(newobj)+'\n')

# post processing is needed 
# the resulting file should be like FollowupGeneration/followup_question_list.json