import json
import re
from typing import Any, List, Tuple
from pydantic import BaseModel
from langchain.prompts.chat import ChatPromptTemplate
from difflib import SequenceMatcher
import time
from wtpsplit import SaT

from utils.rate_limiter import RateLimiter

def pydantic_model_to_json(model: BaseModel) -> dict:
    json_model = model.model_json_schema()

    final_json = {}
    for name, prop in json_model["properties"].items():
        final_json[name] = f"<{prop['type']}: {prop['description']}>"

    return final_json

class StructuredOutputLLM:
    def __init__(self, llm, prompt_template: str, response_format: BaseModel, system_message: str=None, system_message_support: bool=True, structured_output_support: bool=True, raw_pydantic_model_in_prompt: bool=False, rate_limiter: RateLimiter=None) -> None:
        self.prompt_template = prompt_template
        self.system_message = system_message
        self.system_message_support = system_message_support
        self.structured_output_support = structured_output_support
        self.response_format = response_format
        self.raw_pydantic_model_in_prompt = raw_pydantic_model_in_prompt
        self.rate_limiter = rate_limiter
        if structured_output_support:
            self.llm = llm.with_structured_output(response_format)
        else:
            self.llm = llm
            
    def extract_json(self, string: str) -> Tuple[str, Any]:
        """ Extract and parse a json snippet from the given string. """

        # Define a regular expression pattern to match JSON snippets
        json_pattern = r'```json(.*?)```'

        # Use re.DOTALL to match across multiple lines
        match = re.search(json_pattern, string, re.DOTALL)

        if match:
            # Extract the JSON snippet
            json_str = match.group(1)

            try:
                # Load the JSON string into a dictionary
                return "Success", json.loads(json_str)
            except json.JSONDecodeError as e:
                return f"Error decoding JSON: {e}", {}
        else:
            try:
                # Load the JSON string into a dictionary
                return "Success", json.loads(string)
            except json.JSONDecodeError as e:
                return "No JSON snippet found in the text.", {}
            
    def is_similar(self, a: str, b: str, thr: float=0.9) -> bool:
        if a.strip() == b.strip():
            return True
        return SequenceMatcher(None, a, b).ratio() >= thr

    def isin(self, part: str, whole: str) -> bool:
        p = re.sub(r'\s|!|"|#|\$|%|&|\'|\(|\)|\*|\+|,|-|\.|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|`|{|\||}|~', '', part.lower())
        w = re.sub(r'\s|!|"|#|\$|%|&|\'|\(|\)|\*|\+|,|-|\.|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|`|{|\||}|~', '', whole.lower())
        return p in w

    def sentence_split(self, sentences: str) -> List[str]:
        if not hasattr(self, "sentence_splitter") or self.sentence_splitter is None:
            self.sentence_splitter = SaT("sat-12l-sm")
        return self.sentence_splitter.split(sentences)
    
    def _generate(
        self,
        variables: dict,
        verbose: bool = False
    ) -> Any:
        if self.system_message is not None:
            if self.system_message_support:
                chat_prompt = ChatPromptTemplate.from_messages([
                    ("system", self.system_message),
                    ("human", self.prompt_template)
                ])
            else:
                chat_prompt = ChatPromptTemplate.from_messages([
                    ("human", f"{self.system_message}\n\n{self.prompt_template}")
                ])
        else:
            chat_prompt = ChatPromptTemplate.from_messages([
                ("human", self.prompt_template)
            ])

        if self.structured_output_support:
            response_format_prompt_variable = "Ensure your response can be parsed using Python json.loads! and adheres to the Pydantic model schema provided."
        else:
            if self.raw_pydantic_model_in_prompt:
                schema = self.response_format.model_json_schema()
                response_format_prompt_variable = f"Write JSON according to the following Pydantic model JSON schema:\n{json.dumps(schema, indent=4, ensure_ascii=False)}"
            else:
                simplified_schema = pydantic_model_to_json(self.response_format)
                response_format_prompt_variable = f"Respond with JSON that adheres to this structure:\n{json.dumps(simplified_schema, indent=4, ensure_ascii=False)}"

        variables["response_format"] = response_format_prompt_variable

        if verbose:
            print("#"*50)
            formatted_chat_prompt = chat_prompt.format_messages(**variables)
            if self.system_message is not None and self.system_message_support:
                print("SYSTEM MESSAGE")
                print(formatted_chat_prompt[0].content)
                print()
                print("HUMAN MESSAGE")
                print(formatted_chat_prompt[1].content)
            else:
                print("HUMAN MESSAGE")
                print(formatted_chat_prompt[0].content)

        # Build and invoke chain
        if self.structured_output_support and hasattr(self.llm, "with_structured_output"):
            chain = chat_prompt | self.llm.with_structured_output(self.response_format)
        else:
            chain = chat_prompt | self.llm

        if self.rate_limiter is not None:
            self.rate_limiter.acquire()
        
        response_content = chain.invoke(variables)

        if verbose:
            print("#"*50)
            print("RAW LLM RESPONSE")
            print(response_content)

        if not isinstance(response_content, str) and hasattr(response_content, "model_dump"):
            if self.response_format and isinstance(response_content, self.response_format):
                return response_content

        # If response_content is AIMessage or similar, extract content
        if hasattr(response_content, "content") and isinstance(response_content.content, str):
            response_str = response_content.content
        elif isinstance(response_content, str):
            response_str = response_content
        else:
            # Fallback if response_content is an unexpected type but might be convertible to string
            try:
                response_str = str(response_content)
                if verbose:
                    print(f"Warning: LLM response was of type {type(response_content)}, converted to string.")
            except Exception as e:
                raise Exception(f"LLM response was of an unexpected type {type(response_content)} and could not be converted to string: {e}")


        if self.response_format is not None:
            success, parsed_json = self.extract_json(response_str.strip())
            if success.lower() != "success":
                raise Exception(f"Failed to parse JSON from LLM response. Status: {success}, Details: {parsed_json}")
            
            try:
                validated_response = self.response_format.model_validate(parsed_json)
            except Exception as e:
                raise Exception(f"Failed to validate JSON against Pydantic model {self.response_format.__name__}. Error: {e}. JSON: {parsed_json}")
            return validated_response
        else:
            return response_str


    def generate(
        self,
        variables: dict,
        verbose: bool = False,
        max_retries: int = 5,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 600.0
    ) -> Any:
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                return self._generate(variables=variables, verbose=verbose if attempt == 0 else False)
            except Exception as e:
                last_exception = e
                if attempt == max_retries:
                    if verbose:
                        print(f"Final attempt failed after {max_retries} retries. Raising last exception.")
                    raise e

                delay = min(base_delay_seconds * (2 ** attempt), max_delay_seconds)
                if verbose:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
        
        if last_exception:
            raise last_exception
        else:
            raise Exception("Retry logic completed without success or explicit exception.")