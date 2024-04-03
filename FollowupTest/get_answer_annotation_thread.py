import os
import json
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from llmAPI import GPT, GLM
from concurrent.futures import ThreadPoolExecutor, wait
from queue import Queue

temperature = 0
gpt3_client = GPT.from_gpt3(temperature=temperature)
glm_client = GLM.myapi(temperature=temperature)

class PoisonPill:   
    pass

def fetch_answer(jsonobj,queue):
    test_text, question =  jsonobj['usmle_test'], jsonobj['question']
    getanswerprompt = f"This is a USMLE test, please read the test and answer the question below:<br>**test**:{test_text}<br>**question**:{question}<br>**answer**:"
    answer = gpt3_client.predict(getanswerprompt)
    
    jsonobj['answer'] = answer
    with open(os.environ.get('ANSWER_FILE'), 'a') as f:
        f.write(json.dumps(jsonobj) + '\n')

    queue.put(jsonobj)

def process_annotation(queue):
    while True:
        jsonobj = queue.get()
        if isinstance(jsonobj, PoisonPill):
            break
        test_text, question, answer = jsonobj['usmle_test'], jsonobj['question'], jsonobj['answer']
        getannotprompt  = f"this is a USMLE test,'{test_text}',the LLM did it wrong, to explore the reason of it, we ask a **follow-up question**: '{question}'<br> **LLM answer**:{answer}<br>,is it correct? if correct output 'true' , if incorrect output 'false'. **output**:"
        annotation = glm_client.predict(getannotprompt)

        jsonobj['annotate'] = annotation
        with open(os.environ.get('ANNOTATION_FILE'), 'a') as f:
            f.write(json.dumps(jsonobj) + '\n')
        
        queue.task_done()


def process_questions(jsonlist):
    queue = Queue()
    with ThreadPoolExecutor(max_workers=2) as executor:

        annot_thread = executor.submit(process_annotation, queue)

        futures1 = []
        
        for jsonobj in jsonlist:
            future1 = executor.submit(fetch_answer, jsonobj, queue)
            futures1.append(future1)

        
        wait(futures1)
        queue.put(PoisonPill())
        queue.join()
        annot_thread.result()


if __name__ == "__main__":
    os.environ['ANSWER_FILE'] = '/Users/yangyijun/my_github/FollowupPipeline-bak/FollowupTest/followup_question_list_answered1.jsonl'
    os.environ['ANNOTATION_FILE'] = '/Users/yangyijun/my_github/FollowupPipeline-bak/FollowupTest/followup_question_list_annotated1.jsonl'
    jsonfilepath = '/Users/yangyijun/my_github/FollowupPipeline-bak/FollowupGeneration/followup_question_list.json'
    with open(jsonfilepath, 'r') as file:
        jsonlist = json.load(file)
    jsonlist = jsonlist[10:20]  
    process_questions(jsonlist)

