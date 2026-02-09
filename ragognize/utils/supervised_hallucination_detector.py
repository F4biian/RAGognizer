from typing import Any

from utils.structured_output_llm import StructuredOutputLLM

class SupervisedHallucinationDetector(StructuredOutputLLM):
    def __init__(self, llm, prompt_template, response_format, system_message = None, system_message_support = True, structured_output_support = True, raw_pydantic_model_in_prompt = False, rate_limiter = None):
        super().__init__(llm, prompt_template, response_format, system_message, system_message_support, structured_output_support, raw_pydantic_model_in_prompt, rate_limiter)
    
    def annotate(
        self,
        source_chunk: dict,
        question: str,
        answerable: bool,
        answer_quote: str,
        output: str,
        verbose: bool=False
    ) -> Any:...