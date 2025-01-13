import os
import time

import asyncio  
import nest_asyncio

from tqdm import tqdm

from litellm import acompletion

class AsyncLLMPipeline:
    def __init__(self):
        pass

    async def predict(self,
                      user_prompt,
                      max_retries:int=5):
        raise NotImplementedError

    async def model_response(self,
                             user_prompts):
        
        if not isinstance(user_prompts,list):
            user_prompts = [user_prompts]

        # tasks = [self.predict(user_prompt) for user_prompt in user_prompts]
        tasks = [asyncio.create_task(self.predict(user_prompt))
                 for user_prompt in user_prompts]
            
        responses = await asyncio.gather(*tasks)
        timeout_or_incorrect_resp = 0 #count for how many requests timed out or have incorrect responses
        
        decoded_responses = []
        for resp in responses:

            try:
                decoded_response = resp['choices'][0]['message']['content']
            
            except TypeError as te:
                print(resp)
                print(te)
                timeout_or_incorrect_resp += 1
                decoded_response = 'indeterminate'
                
            except Exception as e:
                print(resp, e)
                decoded_response = 'indeterminate'
            finally:
                decoded_responses.append(decoded_response)

        print(f"{timeout_or_incorrect_resp} requests out of {len(tasks)} requests either timed out or returned non-parseable outputs ...")
            
        return decoded_responses

    def batch_predict(self,
                      user_prompts):
        
        if not isinstance(user_prompts,list):
            user_prompts = [user_prompts]
        
        batched_prompts = [user_prompts[idx : idx+50]
                           for idx in range(0, len(user_prompts), 50)]
        
        outputs = []
        for batch in tqdm(batched_prompts):
            batch_output = asyncio.run(self.model_response(batch))
            outputs.extend(batch_output)

            time.sleep(1) #sleep for a second after each batch is processed!

        return outputs

nest_asyncio.apply()

class AsyncLLMPipelineBuilder(AsyncLLMPipeline):

    def __init__(self,
                 system_prompt:str,
                 few_shot_examples:list=[],
                 model:str="gemini-1.5-flash",
                 max_tokens:int=8192,
                 max_timeout_per_request:int=15):

        self.system_prompt = system_prompt
        self.few_shot_examples = few_shot_examples
        self.model = model

        self.max_timeout_per_request = max_timeout_per_request
        self.max_tokens = max_tokens

        super().__init__()

        print(f'Changed model name to : {self.model}...\n')

    def __enter__(self):
        # Actions to perform when entering the context (e.g., load model)
        # Typically, return 'self' to be used within the 'with' block
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        print('Exit called ... cleaning up')

        self.model = None #hack -- reassign model to None

        print('Cleanup complete!\n')

        return True

    async def predict(self,
                      user_prompt,
                      max_retries:int=3):

        messages = []

        system_prompt = [{"role" : "system", "content" : self.system_prompt}]
        messages.extend(system_prompt)

        if self.few_shot_examples != []:
            examples = [[{"role" : "user", "content" : examples[0]},{"role" : "assistant", "content" : examples[1]}]
                        for examples in self.few_shot_examples]
            examples = [arr for sublist in examples for arr in sublist]
            messages.extend(examples)

        user_prompt = [{"role" : "user", "content" : user_prompt}]
        messages.extend(user_prompt)

        retries = 0
        backoff_factor = 2
        min_sleep_time = 3
        
        # time.sleep(0.3) #to avoid getting rate limited immediately
        
        while retries < max_retries:
            try:

                completions = await acompletion(model = f"openai/{self.model}",
                                                messages=messages,
                                                timeout=self.max_timeout_per_request,
                                                user=f"mr-{os.getenv('ENV')}-gcp-bulk",
                                                temperature=0.0,
                                                max_tokens=self.max_tokens,
                                                base_url=os.getenv("LITELLM_BASE_URL"),
                                                api_key=os.getenv("LITELLM_API_KEY")
                                                )
                return completions
            
            except asyncio.TimeoutError as timeout_err:
                print("\ntimeout err : ",timeout_err)
                print('request sent : ',messages)
                return 'indeterminate'
                    
            except Exception as e:
                print('Exception: {}'.format(e))
                sleep_time = min_sleep_time * (backoff_factor ** retries)
                print(f"Rate limit hit. Retrying in {sleep_time} seconds.")
                await asyncio.sleep(sleep_time) 
                retries += 1
        
        return 'indeterminate'