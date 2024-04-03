from __future__ import annotations
from openai import OpenAI, AsyncOpenAI
from enum import Enum
import asyncio


class ExplicitEnum(Enum):
    def __str__(self):
        return str(self.value)
    

class OpenAIModel(ExplicitEnum):
    GPT3 = "gpt-3.5-turbo"
    GPT4 = "gpt-4-32k-0613"

class GPT():
    def __init__(self, api_key: str, base_url: str, model_id: str, temperature: float = 1, async_mode: bool = False):
        
        self.model_id = model_id
        self.temperature = temperature
        self.async_mode = async_mode
        if async_mode:
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    @classmethod
    def from_gpt4(cls,temperature=0.5,async_mode=False):
        return cls(
            api_key="your_api_key_here",
            base_url="https://api.kwwai.top/v1",
            model_id="gpt-4-32k-0613",
            temperature=temperature,
            async_mode=async_mode
        )
    
    @classmethod
    def from_gpt3(cls,temperature=0.5,async_mode=False):
        return cls(
            api_key="your_api_key_here",
            base_url="https://api.kwwai.top/v1",
            model_id="gpt-3.5-turbo",
            temperature=temperature,
            async_mode=async_mode
        )
    
    def predict(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_id,
            temperature=self.temperature
        )
        return completion.choices[0].message.content
    
    async def predict_async(self, prompt: str) -> str:
        if not self.async_mode:
            raise Exception("Async mode is not enabled.")
        completion = await self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_id,
            temperature=self.temperature
        )
        return completion.choices[0].message.content




if __name__=='__main__':
    ## sync mode
    model = GPT.from_gpt3(temperature=1)
    print(model.predict("What is the meaning of life?"))

    ## async mode
    model = GPT.from_gpt3(temperature=1,async_mode=True)
    async def main():
        # 确保使用 await 调用 predict_async 方法
        print(await model.predict_async("What is the meaning of life?"))

    asyncio.run(main())