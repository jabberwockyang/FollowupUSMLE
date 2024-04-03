import os
import json
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from llmAPI import GPT, GLM
import asyncio
import aiofiles

temperature = 0
gpt3_client = GPT.from_gpt3(temperature=temperature,async_mode=True)
glm_client = GLM.myapi(temperature=temperature)


async def fetch_answer(jsonobj):
    test_text, question =  jsonobj['usmle_test'], jsonobj['question']
    getanswerprompt = f"This is a USMLE test, please read the test and answer the question below:<br>**test**:{test_text}<br>**question**:{question}<br>**answer**:"
    answer = await gpt3_client.predict_async(getanswerprompt)
    
    jsonobj['answer'] = answer
    async with aiofiles.open(os.environ.get('ANSWER_FILE'), 'a') as f:
        await f.write(json.dumps(jsonobj) + '\n')

    return answer


async def process_annotation(jsonobj):
    test_text, question, answer = jsonobj['usmle_test'], jsonobj['question'], jsonobj['answer']
    getannotprompt  = f"this is a USMLE test,'{test_text}',the LLM did it wrong, to explore the reason of it, we ask a **follow-up question**: '{question}'<br> **LLM answer**:{answer}<br>,is it correct? if correct output 'true' , if incorrect output 'false'. **output**:"
    annotation = await glm_client.predict_async(getannotprompt)

    jsonobj['annotate'] = annotation
    async with aiofiles.open(os.environ.get('ANNOTATION_FILE'), 'a') as f:
        await f.write(json.dumps(jsonobj) + '\n')


async def getanswer_and_startannot(jsonobj):
    answer = await fetch_answer(jsonobj)
    jsonobj['answer'] = answer
    annotation_task = asyncio.create_task(process_annotation(jsonobj))
    return annotation_task


async def process_questions(jsonlist):
    annotation_tasks = []
    for jsonobj in jsonlist:
        annotation_task = await getanswer_and_startannot(jsonobj)
        annotation_tasks.append(annotation_task)
        
    await asyncio.gather(*annotation_tasks)


if __name__ == "__main__":
    os.environ['ANSWER_FILE'] = '/mnt/yangyijun/FollowupPipeline/FollowupTest/followup_question_list_answered.jsonl'
    os.environ['ANNOTATION_FILE'] = '/mnt/yangyijun/FollowupPipeline/FollowupTest/followup_question_list_annotated.jsonl'
    jsonfilepath = '/mnt/yangyijun/FollowupPipeline/FollowupGeneration/followup_question_list.json'
    with open(jsonfilepath, 'r') as file:
        jsonlist = json.load(file)
    jsonlist = jsonlist[:10]
    asyncio.run(process_questions(jsonlist))

    