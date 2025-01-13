from typing import List, Tuple

import ast
import json
import gc

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

from torch import cuda

class CustomLLMBuilder:

    def __init__(self,
                system_prompt:str="You are a helpful assistant",
                few_shot_examples:List[Tuple[str, str]]=[],
                max_timeout_per_request:int=15,
                max_tokens:int=2048,
                model:str="microsoft/Phi-3-mini-4k-instruct"):
        
        """
        Class to build a custom LLM model for generating text responses.

        args:
            system_prompt (str): the system prompt to be used for generating responses. Defaults to "You are a helpful assistant"
            few_shot_examples (List[Tuple[str, str]]): list of tuples containing a user example and the expected output. Defaults to an empty list.
            max_timeout_per_request (int): maximum time in seconds to wait for a response from the model. Defaults to 15.
            model (str): the model to be used for generating responses. Defaults to "microsoft/Phi-3-mini-4k-instruct"

        returns:
            None
        """
        
        self.system_prompt=system_prompt
        self.few_shot_examples=few_shot_examples

        self.max_timeout = max_timeout_per_request
        self.max_tokens = max_tokens

        if model not in ["meta-llama/Meta-Llama-3-8B-Instruct",
                         "mistralai/Mistral-7B-Instruct-v0.2",
                         "mistralai/Mistral-7B-Instruct-v0.1",
                         "microsoft/Phi-3-mini-4k-instruct",
                         "Sreenington/Phi-3-mini-4k-instruct-AWQ"]:
            
            model = "microsoft/Phi-3-mini-4k-instruct"

        self.model = model

        tensor_parallel_size = 1
       
        #if the model supports awq quantization, set it to awq quantization
        quantization = None if "awq" not in str(self.model).lower().split("-") else "AWQ"

        print(f'Using {quantization} quantization ...\n')

        self.model = LLM(
            model=self.model,
            dtype="half",
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=True,
            trust_remote_code=True,
            quantization=quantization,
            max_model_len=4096,
            gpu_memory_utilization=0.9
        )
        self.sampling_params = SamplingParams(
            n=1,
            temperature=0,
            max_tokens=max_tokens,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        print('Exit called ... cleaning up')
        
        destroy_model_parallel()
        del self.model.llm_engine.model_executor.driver_worker

        self.model = None
        del self.model, self.tokenizer, self.sampling_params

        gc.collect()
        cuda.empty_cache()

        print('Cleanup complete!\n')

        return True

    def _create_prompts(self,
                      texts: List[str],
                      system_prompt : str,
                      few_shot_examples: List[Tuple[str, str]] = []) -> List[str]:
        
        prompts = list()
            
        if few_shot_examples != []:
            few_shot_examples = [[{"role" : "user", "content" : example[0]},{"role" : "assistant", "content" : example[1]}]
                                 for example in few_shot_examples] #format as a list of objects, where first object is user example and second is the expected output
            few_shot_examples = [arr for sublist in few_shot_examples for arr in sublist] #flatten the examples into a single array of dicts
                                    
        for idx, text in enumerate(texts):
                        
            conv = [{"role" : "user", "content" : system_prompt}, {"role" : "assistant", "content" : "understood. continue"}] + few_shot_examples
            conv.append({"role" : "user", "content" : text})

            prompts.append(
                            self.tokenizer.apply_chat_template(conversation=conv,
                                                                tokenize=False,
                                                                add_generation_prompt=True,
                                                            )
                            )

        return prompts

    def predict(self,
                user_prompt:str) -> str:

        """
        Method to generate a response for a user prompt.

        args:
            user_prompt (str): user prompt for which response is to be generated
        """
        
        result = self.batch_predict([user_prompt])[0]
        return result

    def batch_predict(self,
                      user_prompts:List[str]) -> List[str]:
        
        """
        Method to generate responses for a list of user prompts.

        args:
            user_prompts (List[str]): list of user prompts for which responses are to be generated

        returns:
            List[str]: list of responses generated for the user prompts
        """
        
        if not isinstance(user_prompts, list):
            user_prompts = [user_prompts]

        inference_prompts = self._create_prompts(texts=user_prompts,
                                               system_prompt=self.system_prompt,
                                               few_shot_examples=self.few_shot_examples)

        responses = self.model.generate(prompts=inference_prompts,
                                        sampling_params=self.sampling_params)

        results = [output.outputs[0].text.strip() for output in responses]

        return results