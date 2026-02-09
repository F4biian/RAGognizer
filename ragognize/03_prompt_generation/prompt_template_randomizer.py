########################################################################################
# IMPORTS

from pprint import pprint
import random

########################################################################################

random.seed(432)

PROMPT_MODULES = {
    "INFORMATION_LABEL": [
        {
            "singular": "knowledge",
            "plural": "knowledge",
        },
        {
            "singular": "context",
            "plural": "context",
        },
        {
            "singular": "information",
            "plural": "information",
        },
        {
            "singular": "chunk",
            "plural": "chunks",
        },
        {
            "singular": "document",
            "plural": "documents",
        },
        {
            "singular": "passage",
            "plural": "passages",
        }
    ],
    "SYSTEM_MESSAGE": [
        "You are a friendly assistant for question-answering solely using retrieved {{info_label_plural}}. Use the provided {{info_label_plural}} to accurately answer the question or perform the tasks requested by the user. Keep your response concise and to the point.",
        "You're here to assist by answering questions using only the retrieved {{info_label_plural}}. Stay on-topic and ensure your response is clear, concise, and helpful. Avoid speculation—stick to the provided {{info_label_plural}}.",
        "Your role is to answer user queries strictly based on the retrieved {{info_label_plural}}. Ensure responses are factual, accurate, and concise, using only the {{info_label_plural}} provided.",
        "You are a friendly and professional AI assistant designed to answer questions using only the retrieved {{info_label_plural}}. Your responses should be accurate, concise, and based strictly on the provided information. Always maintain a helpful and kind tone, ensuring that your answers are respectful, ethical, and free from bias or discrimination. Do not generate harmful, misleading, or offensive content. Your goal is to provide clear, reliable, and well-structured assistance while adhering to principles of fairness, integrity, and inclusivity.",
        "You are a responsible and ethical AI assistant. Answer questions using only the retrieved {{info_label_plural}}, ensuring accuracy, fairness, and kindness. Stay clear of bias, speculation, or harmful content—be helpful, respectful, and to the point."
    ],
    "INSTRUCTIONS": [
        "You are given the retrieved {{info_label_plural}} and a query by the user. Write a kind and brief response to the user that is only based on the {{info_label_plural}}. Only use the {{info_label_plural}} that are relevant to the user's query. Thus, do not refer to {{info_label_plural}} that are not necessary for answering the user's query!",
        "Answer the user’s query based strictly on the provided {{info_label_plural}}. Keep your response concise, informative, and helpful. Refrain from introducing unrelated details that do not contribute to answering the query.",
        "Read the retrieved {{info_label_plural}} carefully and compose a brief response that aligns strictly with the given information. Maintain a professional yet approachable tone. Do not incorporate any extraneous content that is not directly relevant to the user’s question.",
        "Generate a concise response that is helpful, ethical, and based entirely on the retrieved {{info_label_plural}}. Do not infer or assume details beyond what is provided. Stick to the information given and disregard anything that does not aid in answering the user’s query.",
        "Write a short reply that is both clear and friendly, using nothing but the retrieved {{info_label_plural}}. Avoid speculation or external knowledge. Ensure that your response remains focused and excludes unnecessary information."
    ],
    "INFO_PROVISION": [
        "Your retrieved {{info_label_plural}}:\n{{info}}",
        "Your response must be based on the following {{info_label_plural}}:\n{{info}}",
        "Respond to the user using the following {{info_label_plural}}:\n{{info}}",
        "Use the retrieved {{info_label_plural}} below as the sole source for your response:\n{{info}}",
        "The provided {{info_label_plural}} might contain the relevant details:\n{{info}}"
    ],
    "UNANSWERABILITY_HINT": [
        "If the requested information is not contained in the {{info_label_plural}}, kindly say that you cannot provide an answer.",
        "If the retrieved {{info_label_plural}} do not contain the necessary information, politely indicate that you cannot answer.",
        "If the necessary details are missing from the retrieved {{info_label_plural}}, inform the user that you cannot answer.",
        "Avoid speculation—if the retrieved {{info_label_plural}} do not provide the required information, clearly state that you cannot assist.",
        "Stick to the retrieved {{info_label_plural}}. If they do not contain the necessary details, acknowledge the limitation and refrain from making assumptions."
    ],
    "QUESTION_PHRASE": [
        "User: {{question}}",
        "The user asks the following: {{question}}",
        "User's request: {{question}}",
        "{{question}}",
    ],
    "USER_PROMPT_SUFFIX": [
        "Answer:",
        "Output:",
        "Response:",
        "Reply:",
        "Your answer:",
        "Your output:",
        "Your response:",
        "Your reply:",
    ],
    "SECTION_EMBRACEMENT": [
        "NONE",
        "TAGS",
        "MARKDOWN",
        "LINE",
    ],
    "FORMAT": [
        "NONE",
        "MARKDOWN",
    ],

    "SYNONYMS": {
        "Instructions": [
            "Instructions",
            "Your Task",
            "Task",
            "Todo",
        ],
        "Question": [
            "Question",
            "Request",
            "Query",
            "User"
        ],
        "Information Provision": [
            "{{info_label_plural}}",
            "Your {{info_label_plural}}",
            "The provided {{info_label_plural}}",
            "Retrieved {{info_label_plural}}",
            "Your only {{info_label_plural}}",
        ]
    }
}

def get_prompt_template(
        has_sys_msg: bool,
            instructions_in_sys_msg: bool,
            info_in_sys_msg: bool,
        question_after_info: bool,
        unanswerability_hint: bool,
            unanswerability_hint_in_sys_msg: bool,
        answer_suffix: bool
) -> list:
    info_label = random.choice(PROMPT_MODULES["INFORMATION_LABEL"])
    prompt_format = random.choice(PROMPT_MODULES["FORMAT"])
    section_emb = random.choice(PROMPT_MODULES["SECTION_EMBRACEMENT"])

    info_prov = random.choice(PROMPT_MODULES["INFO_PROVISION"])
    instructions = random.choice(PROMPT_MODULES["INSTRUCTIONS"])
    question_phrase = random.choice(PROMPT_MODULES["QUESTION_PHRASE"])

    unansw_hint = ""
    if unanswerability_hint:
        unansw_hint = random.choice(PROMPT_MODULES["UNANSWERABILITY_HINT"])

    question_title = random.choice(PROMPT_MODULES["SYNONYMS"]["Question"])

    sample_log = {
        "info_label": info_label,
        "prompt_format": prompt_format,
        "section_emb": section_emb,
        "info_prov": info_prov,
        "instructions": instructions,
        "question_phrase": question_phrase,
        "unansw_hint": unansw_hint,
        "md_hashtags": None,
        "md_capitalize": None,
        "md_instr_title": None,
        "md_info_prov_title": None,
        "system_message": None,
        "answer_suffix": None,
        "instr_before_info": None,
    }

    if section_emb == "MARKDOWN":
        info_prov = info_prov.replace("{{info}}", "```\n{{info}}\n```")
        question_phrase = question_phrase.replace("{{question}}", "\n```\n{{question}}\n```")
    elif section_emb == "TAGS":
        info_prov = info_prov.replace("{{info}}", f"<{info_label['plural'].lower()}>\n{{{{info}}}}\n</{info_label['plural'].lower()}>")
        question_phrase = question_phrase.replace("{{question}}", f"\n<{question_title.lower()}>\n{{{{question}}}}\n</{question_title.lower()}>")
    elif section_emb == "LINE":
        line = "--------------"
        info_prov = info_prov.replace("{{info}}", f"{line}\n{{{{info}}}}\n{line}")
        question_phrase = question_phrase.replace("{{question}}", f"\n{line}\n{{{{question}}}}\n{line}")
    
    if prompt_format == "MARKDOWN":
        hashtags = "#" * random.randint(1, 3)
        capitalize = bool(random.getrandbits(1))
        
        instr_title = random.choice(PROMPT_MODULES["SYNONYMS"]["Instructions"])
        info_prov_title = random.choice(PROMPT_MODULES["SYNONYMS"]["Information Provision"])

        sample_log["md_hashtags"] = hashtags
        sample_log["md_capitalize"] = capitalize
        sample_log["md_instr_title"] = instr_title
        sample_log["md_info_prov_title"] = info_prov_title

        if capitalize:
            instr_title = instr_title.upper()
            question_title = question_title.upper()
            info_prov_title = info_prov_title.upper()

        instructions = f"{hashtags} {instr_title}\n{instructions}\n"
        question_phrase = f"{hashtags} {question_title}\n{question_phrase}\n"
        info_prov = f"{hashtags} {info_prov_title}\n{info_prov}\n"

    if has_sys_msg:
        system_msg = random.choice(PROMPT_MODULES["SYSTEM_MESSAGE"])
        sample_log["system_message"] = system_msg

        if unansw_hint:
            if unanswerability_hint_in_sys_msg:
                system_msg = f"{system_msg} {unansw_hint}"
                unansw_hint = ""
            else:
                instructions = f"{instructions} {unansw_hint}"
                unansw_hint = ""
        if instructions_in_sys_msg:
            system_msg = f"{system_msg}\n{instructions}"
            instructions = ""
        if info_in_sys_msg:
            system_msg = f"{system_msg}\n{info_prov}"
            info_prov = ""
    else:
        system_msg = None

    if bool(random.getrandbits(1)):
        to_concat = [info_prov, instructions]
        sample_log["instr_before_info"] = False
    else:
        to_concat = [instructions, info_prov]
        sample_log["instr_before_info"] = True

    if question_after_info:
        to_concat = to_concat + [question_phrase]
    else:
        to_concat = [question_phrase] + to_concat
        
    human_msg = "\n".join(to_concat).strip()

    suffix = ""
    if answer_suffix:
        suffix = random.choice(PROMPT_MODULES["USER_PROMPT_SUFFIX"])
    human_msg = f"{human_msg}\n\n{suffix}".strip()
    sample_log["answer_suffix"] = answer_suffix

    human_msg = human_msg.replace("{{info_label_plural}}", info_label["plural"])
    human_msg = human_msg.replace("{{info_label_plural}}".upper(), info_label["plural"].upper())
    human_msg = human_msg.replace("{{info_label_singular}}", info_label["singular"])
    human_msg = human_msg.replace("{{info_label_singular}}".upper(), info_label["singular"].upper())
    if system_msg is not None:
        system_msg = system_msg.replace("{{info_label_plural}}", info_label["plural"])
        system_msg = system_msg.replace("{{info_label_plural}}".upper(), info_label["plural"].upper())
        system_msg = system_msg.replace("{{info_label_singular}}", info_label["singular"])
        system_msg = system_msg.replace("{{info_label_singular}}".upper(), info_label["singular"].upper())

    if system_msg is None:
        return sample_log, [
            {
                "role": "user",
                "content": human_msg
            }
        ]
    else:
        return sample_log, [
            {
                "role": "system",
                "content": system_msg
            },
            {
                "role": "user",
                "content": human_msg
            }
        ]

def get_random_template() -> list:
    has_sys_msg = bool(random.getrandbits(1))
    instructions_in_sys_msg = has_sys_msg and bool(random.getrandbits(1))
    info_in_sys_msg = has_sys_msg and bool(random.getrandbits(1))
    question_after_info = bool(random.getrandbits(1))
    unanswerability_hint = bool(random.getrandbits(1))
    unanswerability_hint_in_sys_msg = has_sys_msg and unanswerability_hint and bool(random.getrandbits(1))
    question_after_info = (not info_in_sys_msg) and bool(random.getrandbits(1))
    answer_suffix = bool(random.getrandbits(1))

    args = {
        "has_sys_msg": has_sys_msg,
        "instructions_in_sys_msg": instructions_in_sys_msg,
        "info_in_sys_msg": info_in_sys_msg,
        "question_after_info": question_after_info,
        "unanswerability_hint": unanswerability_hint,
        "unanswerability_hint_in_sys_msg": unanswerability_hint_in_sys_msg,
        "question_after_info": question_after_info,
        "answer_suffix": answer_suffix,
    }

    sample_log, template = get_prompt_template(**args)

    return sample_log, template, args

# sample_log, template, settings = get_random_template()

# pprint(sample_log, width=150, sort_dicts=False)
# pprint(settings, width=150, sort_dicts=False)
# pprint(template, width=150, sort_dicts=False)