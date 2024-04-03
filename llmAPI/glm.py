from __future__ import annotations  
from zhipuai import ZhipuAI
from enum import Enum
import asyncio

class ExplicitEnum(Enum):
    def __str__(self):
        return str(self.value)
    
class ModelID(ExplicitEnum):
    GLM3TURBO = "glm-3-turbo"
    GLM4 = "glm-4"



class GLM:
    def __init__(self, api_key,modelid = ModelID.GLM4
                ,temperature = 0):
        self.api_key = api_key
        self.modelid = modelid
        self.temperature = temperature
        self.client = ZhipuAI(api_key=api_key)

    @classmethod
    def myapi(cls, temperature=0.5):
        api_key="your_api_key_here"
        modelid = ModelID.GLM4
        return cls(
            api_key=api_key,
            modelid = modelid,
            temperature=temperature
        )

    def predict(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model= self.modelid.value,
            messages=[
                {"role": "user", "content": prompt}],
        )
        
        return response.choices[0].message.content
    
    async def predict_async(self, prompt: str) -> str:
        response = self.client.chat.asyncCompletions.create(
            model= str(self.modelid),
            messages=[{"role": "user", "content": prompt}],
        )
        task_id = response.id
        task_status = ''
        get_cnt = 0

        while task_status != 'SUCCESS' and task_status != 'FAILED' and get_cnt <= 40:
            result_response = self.client.chat.asyncCompletions.retrieve_completion_result(id=task_id)

            task_status = result_response.task_status
            await asyncio.sleep(2)
            get_cnt += 1
        
        if task_status == 'SUCCESS':
            return result_response.choices[0].message.content
        else:
            return "Failed to retrieve the completion result."
        

if __name__=='__main__':
    # sync mode
    model = GLM.myapi(temperature=1)
    print(model.predict("What is the meaning of life?"))

    # async mode
    model = GLM.myapi(temperature=1)
    async def main():
        result = await model.predict_async("What is the meaning of life?")
        print(result)

    asyncio.run(main())