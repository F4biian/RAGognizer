from typing import List, Optional, Dict, Any
from pprint import pprint
from pydantic import BaseModel, Field
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from utils.supervised_hallucination_detector import SupervisedHallucinationDetector

class Hallucination(BaseModel):
    substring_with_context: str = Field(description="The substring taken from the response that is hallucinated plus some of its context capturing up to three preceding and three trailing words if they exist.")
    substring_analysis: str = Field(description="Thoroughly analyze if the substring_with_context can be shortened to just capture only the hallucinated information. Explain thoroughly using markdown!")
    final_substring: str = Field(description="The final substring as discussed in `substring_analysis`.")

class AnnotationResponse(BaseModel):
    hallucinations_analysis: str = Field(description="A long, detailed and thorough step-by-step analysis written in markdown format of: Consider every detail and parts of the response that is clearly hallucinated to gather text parts that are clearly hallucinated. Finish this thorough analysis with a conclusion.")
    hallucination_as_a_whole_analysis: str = Field(description="A long, detailed and thorough step-by-step analysis written in markdown format of: Is the entire response completely hallucinated? This might be, for example, because the LLM cannot know the certain information.")
    completely_hallucinated: bool = Field(description="`true` if the response is one complete hallucination. Otherwise, `false`. At most, only one of `completely_hallucinated` or `partially_hallucinated` can be true.")
    partially_hallucinated: bool = Field(description="`true` if the response contains at least one hallucination. Otherwise, `false`. At most, only one of `completely_hallucinated` or `partially_hallucinated` can be true.")
    hallucinations: List[Hallucination] = Field(description="If `partially_hallucinated` is `true`, list the hallucinations here by quoting them from the LLM's response.")
    
    cluelessness_analysis: str = Field(description="Your cluelessness analysis for answering the following question: 'Does the LLM state cluelessness/lack of information/lack of access to information in its response?'")
    cluelessness: bool = Field(description="`true` if the LLM states cluelessness, otherwise `false`.")
    
    addressed_user_request_analysis: str = Field(description="Your analysis for answering the following question: 'Does the LLM try to address the user's request?' Even if the LLM's Reply is incorrect, it might still have tried to address the user's request.")
    addressed_user_request: bool = Field(description="`true` if the LLM addressed the user's request, otherwise `false`.")

    correct_language_analysis: str = Field(description="Your analysis for answering the following question: 'Does the LLM consistently use correct grammar and language, or does it make any linguistic errors?' For this analysis, you solely focus on pure language, not the content.")
    correct_language: bool = Field(description="`true` if the LLM uses correct grammar and language and does not make any linguistic errors, otherwise `false`.")
    
SYSTEM_MESSAGE = "You are an objective and precise hallucination annotator. Your role is to check if the LLM has hallucinated information, while responding to me. You provide evidence by quoting the LLM for each hallucination. These substrings are minimal and are definitely a substring of the LLM's response (inlcuding grammar, punctuation and spelling mistakes)."

PROMPT_TEMPLATE = """### YOUR TASK
I will provide you my chat history with an LLM. Your task is to check if the LLM hallucinated, i.e. fabricated information that is factually not supported by the information I gave to the LLM. You output your results in the specified RESPONSE FORMAT.
Note: If the LLM mentions information that it was not asked for, but comes from the information I provided, it is no hallucination! Thus, you focus on fabricated information, not on task-following! Furthermore, information which can be derived very easily from the provided information is also not considered fabricated and not hallucinated!

### CHAT HISTORY
#### ME:
<me>
{me}
</me>

#### LLM:
<llm>
{llm}
</llm>

### RESPONSE FORMAT
{response_format}
"""

class SHD_SimpleChatTokenLevel(SupervisedHallucinationDetector):
    def __init__(self, llm, system_message_support: bool=True, structured_output_support: bool=True, rate_limiter = None) -> None:
        super().__init__(
            llm=llm,
            prompt_template=PROMPT_TEMPLATE,
            response_format=AnnotationResponse,
            system_message=SYSTEM_MESSAGE,
            system_message_support=system_message_support,
            structured_output_support=structured_output_support,
            raw_pydantic_model_in_prompt=True,
            rate_limiter=rate_limiter
        )

    def annotate_parallel(
        self,
        prompt: str,
        response: str,
        answerable: bool=None,
        source_chunk: dict=None,
        verbose: bool=False,
        samples: int=7,
    ) -> dict:
        def worker():
            return self.annotate(
                prompt=prompt,
                response=response,
                answerable=answerable,
                source_chunk=source_chunk,
                verbose=verbose,
            )

        sample_results = []
        original_outputs = []
        
        with ThreadPoolExecutor(max_workers=samples) as executor:
            futures = [executor.submit(worker) for _ in range(samples)]
            for future in as_completed(futures):
                annotation = future.result()
                sample_results.append(annotation["result"])
                original_outputs.append(annotation["original_output"])
        
        def get_values(field: str) -> bool:
            return np.array([res[field] for res in sample_results])
        
        addressed_user_prompt = np.mean(get_values("addressed_user_prompt"))
        cluelessness = np.mean(get_values("cluelessness"))
        correct_language = np.mean(get_values("correct_language"))
        completely_hallucinated = np.mean(get_values("completely_hallucinated"))
        all_valid = np.mean(get_values("all_valid"))
        
        # Count votes for every single annotated character
        char_votes = np.zeros(shape=(len(response),))
        for res in sample_results:
            for h in res["hallucinations"]:
                if h["valid"] and h["start"] is not None:
                    char_votes[h["start"]:h["end"]] += 1

        # Take only those chars that have enough votes
        final_char_mask = char_votes > samples // 2

        spans = []
        i = 0
        while i < len(final_char_mask):
            if final_char_mask[i]:
                start = i
                while i < len(final_char_mask) and final_char_mask[i]:
                    i += 1
                end = i
                span_text = response[start:end]
                spans.append({
                    "valid": True,
                    "text": span_text,
                    "start": start,
                    "end": end
                })
            else:
                i += 1
        
        aggregated_result = {
            "all_valid": all_valid,
            "addressed_user_prompt": addressed_user_prompt,
            "cluelessness": cluelessness,
            "correct_language": correct_language,
            "completely_hallucinated": completely_hallucinated,
            "hallucinations": spans,
        }
        
        aggregated_annotation = {
            "original_output": original_outputs,
            "result": aggregated_result,
        }
        
        if verbose:
            print("#" * 50)
            print("Aggregated Majority Voting Annotation:")
            pprint(aggregated_annotation, width=150, sort_dicts=False)
        
        return aggregated_annotation
        

    def _annotate_task_wrapper(self, args_tuple: tuple) -> dict:
        attempts = 0
        while attempts < 3:
            attempts += 1
            try:
                prompt, response, answerable, source_chunk, verbose_item = args_tuple
                res = self.annotate(
                    prompt=prompt,
                    response=response,
                    answerable=answerable,
                    source_chunk=source_chunk,
                    verbose=verbose_item
                )
                return res
            except Exception as e:
                print("Error in _annotate_task_wrapper:", e)
                if attempts < 3:
                    sl = 6.65**attempts
                    print(f"Waiting {sl} secs...")
                    time.sleep(sl) # 6.65^3 ~= 300 secs (5min)
                return None

    def annotate_batch(
        self,
        prompts: List[str],
        responses: List[str],
        answerables: Optional[List[Optional[bool]]] = None,
        source_chunks: Optional[List[Optional[Dict[str, Any]]]] = None,
        verbose: bool = False,
        max_workers: Optional[int] = None
    ) -> List[dict]:
        if not (len(prompts) == len(responses)):
            raise ValueError("Prompts and responses lists must have the same length.")

        num_items = len(prompts)

        # Prepare answerables list: if None, create a list of Nones
        if answerables is None:
            actual_answerables = [None] * num_items
        elif len(answerables) == num_items:
            actual_answerables = answerables
        else:
            raise ValueError("Answerables list must have the same length as prompts or be None.")

        # Prepare source_chunks list: if None, create a list of Nones
        if source_chunks is None:
            actual_source_chunks = [None] * num_items
        elif len(source_chunks) == num_items:
            actual_source_chunks = source_chunks
        else:
            raise ValueError("Source_chunks list must have the same length as prompts or be None.")
        
        tasks_args = []
        for i in range(num_items):
            tasks_args.append((
                prompts[i],
                responses[i],
                actual_answerables[i],
                actual_source_chunks[i],
                verbose
            ))

        all_annotations: List[dict] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results_iterator = executor.map(self._annotate_task_wrapper, tasks_args)
            all_annotations = list(results_iterator)

        return all_annotations


    def annotate(
        self,
        prompt: str,
        response: str,
        answerable: bool=None,
        source_chunk: dict=None,
        verbose: bool=False
    ) -> dict:
        if answerable is not None:
            if source_chunk is None:
                raise Exception("`answerable` is set although the `source_chunk` is not None.")
        if answerable is None:
            if source_chunk is not None:
                raise Exception("`source_chunk` is set although `answerable` is None.")

        if source_chunk is not None:
            chunk = f"Title: '{source_chunk['title'].strip()}'\n{source_chunk['chunk'].strip()}"

            if answerable:
                variables = {
                    "me": f"Here is context you might need to answer my prompt:\n<context>\n{chunk}\n</context>\nMy prompt:\n<my_prompt>\n{prompt}\n</my_prompt>",
                    "llm": response,
                }
            else:
                variables = {
                    "me": f"Here is context you might need to answer my prompt:\n<Empty>\n\nMy prompt:\n<my_prompt>\n{prompt}\n</my_prompt>",
                    "llm": response,
                }
        else:
            # `answerable` has no effect when chunk(s) and prompt are given as one full prompt. 
            variables = {
                "me": prompt,
                "llm": response,
            }


        annotation_response = super().generate(variables=variables, verbose=verbose)

        annotation_response_json = json.loads(annotation_response.model_dump_json())

        annotations = {
            "original_output": annotation_response_json,
            "result": {
                "all_valid": True,
                "addressed_user_prompt":  annotation_response.addressed_user_request,
                "cluelessness": annotation_response.cluelessness,
                "correct_language": annotation_response.correct_language,
                "completely_hallucinated": annotation_response.completely_hallucinated,
                "hallucinations": []
            }
        }

        if annotation_response.completely_hallucinated:
            annotations["result"]["all_valid"] = not annotation_response.partially_hallucinated
            annotations["result"]["hallucinations"].append({
                "valid": not annotation_response.partially_hallucinated,
                "text": response,
                "start": 0,
                "end": len(response),
            })
        else:
            if annotation_response.partially_hallucinated and len(annotation_response.hallucinations) == 0:
                annotations["result"]["all_valid"] = False
            if not annotation_response.partially_hallucinated and len(annotation_response.hallucinations) > 0:
                annotations["result"]["all_valid"] = False
            
            if annotation_response.partially_hallucinated:
                for h in annotation_response.hallucinations:
                    broad_quote = h.substring_with_context
                    quote = h.final_substring

                    if quote is not None and quote.strip() != "":
                        is_quote_valid = self.isin(quote, response)

                        start = None
                        end = None
                        if is_quote_valid and quote in response:
                            if response.count(quote) == 1:
                                start = response.find(quote)
                                end = start + len(quote)
                            else:
                                # If quote not unique -> try to use broad_quote to identify start and end
                                if broad_quote is not None and broad_quote.strip() != "":
                                    if broad_quote in response:
                                        if response.count(broad_quote) == 1:
                                            if broad_quote.count(quote) == 1:
                                                broad_start = broad_quote.find(quote)
                                                start = response.find(broad_quote) + broad_start
                                                end = start + len(quote)

                        annotations["result"]["hallucinations"].append({
                            "valid": is_quote_valid,
                            "text": quote,
                            "start": start,
                            "end": end,
                        })
                    else:
                        is_quote_valid = False

                    annotations["result"]["all_valid"] = annotations["result"]["all_valid"] and is_quote_valid
                
        if verbose:
            print("#"*50)
            print("ANNOTATIONS")
            pprint(annotations, width=150, sort_dicts=False)

        return annotations