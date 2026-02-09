import json
from pydantic import BaseModel, Field
import random

from utils.structured_output_llm import StructuredOutputLLM

INFORMATION_TYPES = ["NUMERICAL", "TEMPORAL", "SPATIAL", "IDENTITY", "OTHER"]
CATEGORIES = ["MOVIES_AND_TV", "SPORTS", "MUSIC", "LITERATURE", "POLITICS", "SCIENCE", "FOOD", "HEALTH", "TECHNOLOGY", "BUSINESS", "EDUCATION", "PHILOSOPHY", "OTHER"]

class UserPromptGenerationResponse(BaseModel):
    scratchpad_what: str = Field(description="Write down some possibilities what information of the PASSAGE a user might ask for in a RAG application to an LLM. The answer to that prompt needs to be a definitive substring of the PASSAGE! Select only one of those possibilities that will be asked for!")
    scratchpad_requirements: str = Field(description="Go through each USER PROMPT REQUIREMENT and explain how you would fulfill each requirement by concretely specifying the future content of the user_prompt that is needed to fulfill the requirement.")
    user_prompt: str = Field(description="The prompt that the user would write in a RAG application.")
    user_prompt_requirements_fulfillment: str = Field(description="Go through each USER PROMPT REQUIREMENT and examine if the user_prompt fulfills each. After you examined all USER PROMPT REQUIREMENTS, conclude whether all USER PROMPT REQUIREMENTS are met. If not, explain in detail how you have to change the user_prompt in order to fulfill all USER PROMPT REQUIREMENTS.")
    user_prompt_final: str = Field(description="If the user_prompt did not fulfill all USER PROMPT REQUIREMENTS before, rephrase the user_prompt so it fulfill all USER PROMPT REQUIREMENTS. If it fulfilled already everything, repeat it here.")
    answer_quote: str = Field(description="The minimal answer to the user prompt as a definitive substring of the PASSAGE. Do not correct spelling, grammar, punctuation or other mistakes. Ensure it is a substring of the PASSAGE! Quote solely the words that are really necessary for answering the prompt! Just the bare minimum!")
    answer_quote_minimal_check: str = Field(description="Validate and explain if the answer_quote can be even shorter, so that it represents only the word(s) that contain solely the information that is asked for, the bare minimum! Mostly, quoting the entire PASSAGE is not minimal!")
    answer_quote_minimal: str = Field(description="If the answer_quote was not minimal, write down the minimal answer_quote here. If it was already minimal, repeat it here.")
    answer_quote_minimal_substring_check: str = Field(description="Validate and explain if the answer_quote_minimal is definitely a substring of the PASSAGE. The substring should even have the same grammar, punctuation and spelling mistakes as in the PASSAGE if there are any.")
    answer_quote_final: str = Field(description="If the answer_quote_minimal was not a substring of the PASSAGE, write down the new version here so it is definitively a substring of the PASSAGE. If it was already a substring, repeat it here.")
    category: str = Field(description=f"Classify in what domain or category we are currently in. One of {', '.join(CATEGORIES)}. Do not invent new categories, just use one of these {len(CATEGORIES)}!")
    information_type: str = Field(description=f"Classify what type of information is asked for. One of: {', '.join(INFORMATION_TYPES)}. Do not invent new types, just use one of these {len(INFORMATION_TYPES)}!")
    repeat_user_prompt: str = Field(description="Just repeat the user_prompt here to bring it to mind, so you phrase better answers in the following.")
    golden_answer_attempt: str = Field(description="Write the golden answer to the prompt that the LLM assistant would write kindly to the user. Also, kindly correct wrong information from the user if there is any. Use up to five sentences.")
    golden_answer_check: str = Field(description="Give a brief step-by-step analysis using markdown format of checking if the following aspects are fulfilled:\n1) The golden_answer_attempt is really 100% correct and 100% grounded in the information provided (Do a thorough step-by-step analysis!).\n2) The golden_answer_attempt just states the requested information and, thus, does not explicitly mention 'comes from the provided text', 'as stated in the document', 'is in the given information' or any similar phrases.\n3) The answer is kindly phrased, not too short, not too long.\n4) Finally, if one of the previous requirements was not entirely fulfilled, correct the golden_answer_attempt to fulfill all previous requirements.")
    golden_answer: str = Field(description="Write the final and perfect golden_answer to the user prompt as discussed in the previous golden_answer_check!")

    cluelessness_answer_attempt: str = Field(description="Imagine the LLM assistant does not know the answer. Phrase a perfect and kind answer to the prompt where the LLM states its cluelessness, thus, explains that it cannot provide an answer. Again, do not explicitly mention a provided text/chunk/document/information etc. Either the LLM states it knows or it does not know. Start with 'I am sorry' and do not incorporate the true answer to the prompt in the cluelessness_answer!")
    cluelessness_answer_check: str = Field(description="Give a brief step-by-step analysis using markdown format of checking if the following aspects are fulfilled:\n1) The cluelessness_answer_attempt really states cluelessness and does not reveal any information from the information provided (Do a thorough step-by-step analysis!).\n2) The cluelessness_answer_attempt just does not explicitly mention 'not contained in the provided text', 'as never stated in the documents', 'not found in the given information' or any similar phrases.\n3) The answer is kindly phrased, not too short, not too long.\n4) Finally, if one of the previous requirements was not entirely fulfilled, correct the cluelessness_answer_attempt to fulfill all previous requirements.")
    cluelessness_answer: str = Field(description="Write the final and perfect cluelessness_answer to the user prompt as discussed in the previous cluelessness_answer_check!")



PROMPT_REQUIREMENTS = [
    [
        "- INSTRUCTION: The prompt is phrased as an instruction/a command, not as a question.",
        "- QUESTION: The prompt is phrased as a question using a question mark (?).",
    ],

    [
        "- IMMEDIATE: The user asks immediately for the information, no unnecessary talking.",
        "- ADDING_CONTEXT: The user very briefly states what he/she already knows as a form of providing context. After that, the user finally phrases the text for asking for information that he/she lacks.",
    ],

    [
        "- YOU: The user directly speaks to the LLM using literally 'you'.",
        "- OBJECTIVELY: The user does speak to the LLM using 'you' or other words but tries to be objective and factual.",
    ],

    [
        "- PERFECT: The prompt has no grammar, puncutation or spelling flaws.",
        "- FLAWED: The prompt has some  1) grammar flaws,  2) missing punctuation or  3) spelling errors.",
    ],

    [
        # Repeated twice for equal weight
        # NON_CONFIRMATORY
        "- NON_CONFIRMATORY: The prompt is not phrased confirmatory/affirmative.",
        "- NON_CONFIRMATORY: The prompt is not phrased confirmatory/affirmative.",

        # CONFIRMATORY
        "- CONFIRMATORY_WRONG: The user thinks that he/she already knows the answer, but provides wrong information. The user wants that this wrongly stated information gets verified.",
        "- CONFIRMATORY_CORRECT: The user already knows the answer and provides the correct information (answer). The user wants that this already correctly stated information gets verified.",
    ]
]

SYSTEM_MESSAGE = "You are perfect at creating a user prompt to a certain passage taken from a wikipedia article. Your user prompt is such phrased that it could also be asked by a human user in a Retrieval-Augmented Generation application, i.e., it is 'globally' phrased, so the asked LLM has no clue about the wikipedia article's content. You always fulfill the USER PROMPT REQUIREMENTS."

PROMPT_TEMPLATE = """### TITLE of the Wikipedia Article
```
{title}
```

### BEGINNING of the Wikipedia Article
```
{art_beginning}
```

### CONTEXT
```
{context}
```

### PASSAGE
```
{passage}
```

### USER PROMPT REQUIREMENTS
- GLOBALLY_PHRASED: For example, refer to entities by name and not pronouns when they are called for the first time. This user prompt is the first message in the RAG application chat, so the LLM has initially no clue what is talked about. Be precise and use particular names! 
- DIFFERENT_WORDING: The wording of the user prompt is completely different from the PASSAGE's wording. Employ other vocabulary if possible!
- ONE_FACT_ONLY: The user asks only for one atomic fact from the PASSAGE, not multiple ones!
{requirements}

### OBJECTIVE
Your task is to create a user prompt that could be asked in a Retrieval-Augmented Generation (RAG) application by a human user. This prompt asks solely for information from the PASSAGE. The CONTEXT, BEGINNING and TITLE of the Wikipedia Article are only a help for you to phrase the prompt in a global manner and to fulfill other USER PROMPT REQUIREMENTS, which the user prompt must align with. Use the scratchpad to think step-by-step first, before you get to a final user prompt. Ensure that the answer_quote is 100% a substring of the PASSAGE and the bare minimum of words! So keep spelling, punctuation, grammar and other mistakes! You ALWAYS explain HOW you fulfill each requirement in the step `scratchpad_requirements`. At the end, you provide the `golden_answer` to the `user_prompt` using one or multiple sentences and also a `cluelessness_answer` stating cluelessness in one or two sentences. Be kind!
Note: It is normal that the CONTEXT will contain the PASSAGE. The CONTEXT is just for providing more context to the PASSAGE (what is written before and after the PASSAGE).

### YOUR RESPONSE FORMAT
{response_format}
"""

class UserPromptGenerator(StructuredOutputLLM):
    def __init__(self, llm, system_message_support: bool=True, structured_output_support: bool=True, rate_limiter = None, seed = 432) -> None:
        super().__init__(
            llm=llm,
            prompt_template=PROMPT_TEMPLATE,
            response_format=UserPromptGenerationResponse,
            system_message=SYSTEM_MESSAGE,
            system_message_support=system_message_support,
            structured_output_support=structured_output_support,
            raw_pydantic_model_in_prompt=False,
            rate_limiter=rate_limiter
        )
        if seed is not None:
            random.seed(seed)
            
    def generate(self, title: str, art_beginning: str, context: str, passage: str, verbose: bool=False):
        requirements = []
        requirement_labels = []
        for reqs in PROMPT_REQUIREMENTS:
            random_req = random.choice(reqs)
            requirements.append(random_req)
            requirement_labels.append(random_req.split(":")[0][1:].strip())
        requirements = "\n".join(requirements)

        if verbose:
            print("#"*50)
            print("Sampled USER PROMPT REQUIREMENTS:")
            print(requirements)

        variables = {
            "title": title.strip(),
            "art_beginning": art_beginning.strip(),
            "context": context.strip(),
            "passage": passage.strip(),
            "requirements": requirements
        }

        response: UserPromptGenerationResponse = super().generate(variables=variables, verbose=verbose)
        if response is None:
            return None

        # from pprint import pprint
        # pprint(self.extract_json(response), indent=4, width=150, sort_dicts=False)

        user_prompt = response.user_prompt_final
        answer_quote = response.answer_quote_final
        category = response.category.upper()
        information_type = response.information_type.upper()
        golden_answer = response.golden_answer
        cluelessness_answer = response.cluelessness_answer

        valid = True

        if not self.isin(answer_quote, passage):
            valid = False
        if category not in CATEGORIES:
            valid = False
        if information_type not in INFORMATION_TYPES:
            valid = False

        output = {
            "original_output": json.loads(response.model_dump_json()),
            "result": {
                "valid": valid,
                "sampled_requirements": requirement_labels,
                "user_prompt": user_prompt,
                "answer_quote": answer_quote,
                "category": category,
                "information_type": information_type,
                "golden_answer": golden_answer,
                "cluelessness_answer": cluelessness_answer,
            }
        }

        return output