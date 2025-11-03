from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(), override=True)

import gradio as gr
import threading

from ragognizer.detectors.RAGognizer import RAGognizer


model_lock = threading.Lock()
detector = RAGognizer(use_postprocessor=False)
def interpolate_color(score):
    if score <= 0.4:
        # White
        return "rgb(255, 255, 255)"
    elif score <= 0.65:
        # yellow
        return "rgb(255, 249, 196)"
    elif score <= 0.85:
        # Orange red
        return "rgb(255, 204, 128)"
    else:
        # Red (a strong, saturated red)
        return "rgb(239, 68, 68)"
    
def generate_stream(prompt):
    """
    Dummy function for the "Generation" tab. Simulates a model generating tokens
    and scores, with each token colored individually based on its score.
    """
    if not prompt:
        return ""

    max_score = 0
    with model_lock:
        full_styled_response = ""
        for tok in detector.stream_generate([{
                "role": "user",
                "content": prompt,
            }],
            max_new_tokens = 1024
        ):
            token_str = tok["text"]
            score = tok["prob"]
            color = interpolate_color(score)
            if score > max_score:
                max_score = score
            full_styled_response += (
                f"<span class='score-bubble' "
                f"data-score='Score: {score:.2f}' "
                f"style='display:inline-block; vertical-align:middle; background-color:{color}; color: black; border-radius:6px; padding:2px 2px;'>"
                f"{token_str}</span>"
            )
            yield full_styled_response, f"Max Score: {max_score:.4f}"

def detect_and_color(system_prompt, prompt, response):
    if not response:
        return "", "Max Score: N/A"
    
    chat = []

    if system_prompt.strip():
        chat += [{
            "role": "system",
            "content": system_prompt
        }]

    chat += [
        {
            "role": "user",
            "content": prompt
        },
        {
            "role": "assistant",
            "content": response,
        }
    ]

    with model_lock:
        tokens = detector.predict(chat=chat)

    max_score = 0
    full_styled_response = ""

    for token in tokens:
        token_str = token["text"]
        score = token["prob"]
        if score > max_score:
            max_score = score
        color = interpolate_color(score)
        full_styled_response += (
            f"<span class='score-bubble' "
            f"data-score='Score: {score:.2f}' "
            f"style='display:inline-block; vertical-align:middle; background-color:{color}; color: black; border-radius:6px; padding:2px 2px;'>"
            f"{token_str}</span>"
        )

    return full_styled_response, f"Max Score: {max_score:.4f}"


custom_css = """
<style>
    .gradio-container {
        max-width: 800px !important;
        margin: auto !important;
    }
    #prompt-container-generation, #prompt-container-detection {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
    }

    .score-bubble:hover::after {
        content: attr(data-score);
        position: absolute;
        top: -2px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0,0,0,0.85);
        color: white;
        padding: 4px 6px;
        border-radius: 4px;
        font-size: 10px;
        white-space: nowrap;
        pointer-events: none;
        opacity: 1;
        transition: opacity 0.05s;
        z-index: 999;
    }

    .score-bubble::after {
        opacity: 0;
    }
</style>
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Token-Level Hallucination Detection Using F4biian/RAGognizer-Qwen3-4B-Instruct-2507")
    gr.Markdown(
        """
- ✅ **Trained on RAGognize (English only)**, but might even work in other languages
- 🎯 Designed for primarily detecting **closed-domain hallucinations** (**relative to the provided context**)
- 🎨 **Color Score Legend** (Model confidence that a token is hallucinated):
  - **≤ 0.40** — White
  - **≤ 0.65** — Yellow
  - **≤ 0.85** — Orange
  - **≤ 1.00** — Red
- This space uses [**F4biian/RAGognizer-Qwen3-4B-Instruct-2507**](https://huggingface.co/F4biian/RAGognizer-Qwen3-4B-Instruct-2507) (bfloat16 & transformer heads library **without** a post-processing MLP** on the final token scores). It sometimes struggles with showing emojis.
- **Tip**: When writing RAG prompts, clearly separate the provided context from the user’s request. For example, start with a directive such as “Only use the following information” to help the model distinguish between the two. You can also structure your prompt using tags (e.g., ```<context> ... </context>``` and ```<user> ... </user>```) to ensure the model correctly understands what information it should rely on.
        """
    )

    gr.Markdown("Use the tabs below to switch between **Detection** (analyze existing text) and **Generation** (a new response).")

    with gr.Tabs():
        with gr.TabItem("Detection", id=0):
            gr.Markdown("Enter a (RAG) prompt and a corresponding response. When you click 'Analyze', each token in the response will be colored based on the score returned by the RAGognizer model, and the maximum score will be displayed below.")
            with gr.Column(elem_id="prompt-container-detection"):
                system_prompt_input = gr.Textbox(label="Optional System Prompt", placeholder="e.g., You are a helpful assistant.", lines=3)
                prompt_input_detect = gr.Textbox(label="Prompt", placeholder="e.g., What is the capital of France?", lines=5)
                response_input = gr.Textbox(label="Response", placeholder="e.g., The capital of France is Paris.", lines=5)
                hallucination_status = gr.Textbox(value="", visible=False)
                analyze_button = gr.Button("Analyze", variant="primary")

                gr.Examples(
                    examples=[
                        # Example 0 — Hallucination
                        [
                            "",
                            """Context: The capital of France is Rome.
Based solely on the provided context, answer this query: What is the capital of France?""",
                            """Hi, thank you for reaching out!
The capital of France is Paris. It’s home to millions of people.
Do you have any other questions I can help you with?""",
                            "❌"
                        ],

                        # Example 1 — No Hallucination
                        [
                            "Only use the following information.",
                            """<context>
The Leaning Tower of Pisa is located in Pisa, Italy, and is famous for its unintended tilt.
Construction began in the 12th century and continued over 199 years due to wars.
Engineers have performed stabilization work to prevent further tilting.
Tourists can climb the spiral staircase to reach the top.

Pisa Cathedral and the Pisa Baptistery are part of the same Piazza dei Miracoli complex.
The tower is mainly constructed of white and grey marble.
</context>
<user>Where is the Leaning Tower of Pisa located?</user>""",
                            "The Leaning Tower of Pisa is located in Pisa, Italy.",
                            "✅"
                        ],

                        # Example 2 — No Hallucination
                        [
                            "Only use the following information.",
                            """<context>
The human body contains several vital organs. The heart pumps blood throughout the body and has four chambers.
The lungs supply oxygen to the blood and remove carbon dioxide.
The liver processes nutrients and detoxifies chemicals.

The brain is part of the central nervous system and is responsible for cognition, memory, and motor control.
It is protected by the skull and surrounded by cerebrospinal fluid.
</context>
<user>What is the primary function of the brain?</user>""",
                            "The primary function of the brain is to control cognition, memory, and motor functions.",
                            "✅"
                        ],

                        # Example 3 — Has Hallucination
                        [
                            "Only use the following information.",
                            """<context>
Saturn is a gas giant planet in our solar system known for its rings made mostly of ice particles.
It has 83 confirmed moons, one of the largest being Titan.
Voyager and Cassini have both studied Saturn up close.

Titan has a dense, nitrogen-rich atmosphere and surface lakes of liquid hydrocarbons.
Titan orbits Saturn roughly every 16 Earth days.
</context>
<user>How many moons does Saturn have?</user>""",
                            "Saturn has 85 moons in total, with Titan being the largest.",
                            "❌"
                        ],

                        # Example 4 — Has Hallucination
                        [
                            "Only use the following information.",
                            """<context>
Marie Curie was a physicist and chemist known for pioneering research on radioactivity.
She discovered the elements polonium and radium.
She won two Nobel Prizes: one in Physics and one in Chemistry.

X-ray machines were rapidly adopted during World War I due to mobile radiography units designed by Curie.
Curie worked in Paris at the Sorbonne.

Her husband, Pierre Curie, collaborated closely with her scientific work.
</context>
<user>Which Nobel Prizes did Marie Curie receive?</user>""",
                            "Marie Curie received Nobel Prizes in Physics, Chemistry, and later one in Medicine for her contributions to radiation therapy.",
                            "❌"
                        ],

                        # Example 5 — Has no Hallucination
                        [
                            """Only use the following information to answer the user's question. Do not rely on outside knowledge. Base your answer entirely on the docs below and follow them content-wise (not necessarily word-for-word). Do not invent anything.""",
                            """Only use the information from there provided documents: <context>
<doc id="a01" source="official">
Project Nebula – Internal Storage Overview (Rev 4.1, approved)
- Standard storage tier: S3-compatible bucket "neb-data-prod"
- Encryption: AES-256, server-side only
- Default retention: 365 days
- File size limit: 100 MB per object
</doc>

<doc id="a02" source="draft">
Nebula storage supports objects up to 120 MB. Retention currently 360 days.
</doc>

<doc id="a03" source="admin">
Retention policy (consolidated notes)
- Production: 365 days
- Exceptions exist for staging (180 days)
Note: Some early docs mention 366 days for leap-year alignment, but not applied.
</doc>

<doc id="a04" source="official">
Project Nebula Authentication
- Access token length: 48 characters
- Token pattern: alphanumeric only (A–Z, a–z, 0–9)
- Renew interval: 24 hours
</doc>

<doc id="a05">
Dev chatter (outdated)
Tokens used to be 50 chars long incl. dashes.
</doc>

<doc id="a06" source="official">
Nebula Logging Configuration
- Log archive: "neb-log-arch"
- Log retention: 30 days
- Compression: GZIP
</doc>

<doc id="a07">
Confluence page (typos)
File limt maybe 100mb but some says 101 MB durin beta.
</doc>

<doc id="a08" source="official">
Approved Interfaces
- Storage tier: S3-compatible
- Must use prefix: "neb/" for all stored objects
- Public access: forbidden
</doc>

<doc id="a09">
Old spreadsheet (contradict)
Retention: 366 days
</doc>

<doc id="a10" source="official">
Release Notes Snippet (final)
Reminder: Token length is fixed at 48. No punctuation allowed.
</doc>
</context>
<user>
What is the maximum file size per object in storage, the default retention period, and the token length?
</user>
Answer in one concise line and cite the doc ids you used in brackets.""",
                            """According to the current and officially approved Nebula documentation, the maximum file size per object in storage in the S3-compatible “neb-data-prod” bucket is 100 MB, and objects are retained by default for 365 days in production environments under the standard policy. Additionally, access into Nebula requires an authentication token that is always exactly 48 alphanumeric characters long, with no punctuation allowed. [a01, a03, a04, a10]""",
                            "✅"
                        ],

                        # Example 6 — Has Hallucination
                        [
                            """Only use the following information to answer the user's question. Do not rely on outside knowledge. Base your answer entirely on the docs below and follow them content-wise (not necessarily word-for-word). Do not invent anything.""",
                            """Only use the information from there provided documents: <context>
<doc id="a01" source="official">
Project Nebula – Internal Storage Overview (Rev 4.1, approved)
- Standard storage tier: S3-compatible bucket "neb-data-prod"
- Encryption: AES-256, server-side only
- Default retention: 365 days
- File size limit: 100 MB per object
</doc>

<doc id="a02" source="draft">
Nebula storage supports objects up to 120 MB. Retention currently 360 days.
</doc>

<doc id="a03" source="admin">
Retention policy (consolidated notes)
- Production: 365 days
- Exceptions exist for staging (180 days)
Note: Some early docs mention 366 days for leap-year alignment, but not applied.
</doc>

<doc id="a04" source="official">
Project Nebula Authentication
- Access token length: 48 characters
- Token pattern: alphanumeric only (A–Z, a–z, 0–9)
- Renew interval: 24 hours
</doc>

<doc id="a05">
Dev chatter (outdated)
Tokens used to be 50 chars long incl. dashes.
</doc>

<doc id="a06" source="official">
Nebula Logging Configuration
- Log archive: "neb-log-arch"
- Log retention: 30 days
- Compression: GZIP
</doc>

<doc id="a07">
Confluence page (typos)
File limt maybe 100mb but some says 101 MB durin beta.
</doc>

<doc id="a08" source="official">
Approved Interfaces
- Storage tier: S3-compatible
- Must use prefix: "neb/" for all stored objects
- Public access: forbidden
</doc>

<doc id="a09">
Old spreadsheet (contradict)
Retention: 366 days
</doc>

<doc id="a10" source="official">
Release Notes Snippet (final)
Reminder: Token length is fixed at 48. No punctuation allowed.
</doc>
</context>
<user>
What is the maximum file size per object in storage, the default retention period, and the token length?
</user>
Answer in one concise line and cite the doc ids you used in brackets.""",
                            """Based on the officially maintained Nebula documentation, any individual file uploaded to the “neb-data-prod” storage tier can be up to 100 MB in size, and the platform’s default production retention period is set to 366 days to ensure long-term storage. Nebula authentication uses fixed-length access tokens that are always 48 alphanumeric characters and are required to be renewed every 24 hours. [a01, a03, a04, a10]""",
                            "❌"
                        ],
                    ],
                    inputs=[system_prompt_input, prompt_input_detect, response_input, hallucination_status],
                    label="📌 Click an Example to Auto-Fill the Inputs"
                )

            output_display_detect = gr.HTML(label="Analyzed Response")
            max_score_display = gr.Label(label="Result")

            analyze_button.click(
                fn=detect_and_color,
                inputs=[system_prompt_input, prompt_input_detect, response_input],
                outputs=[output_display_detect, max_score_display]
            )

        with gr.TabItem("Generation", id=1):
            gr.Markdown("Enter a (RAG) prompt and press Enter or click 'Submit'. The response will be shown below, with each token's background color corresponding to its hallucination score.")
            with gr.Column(elem_id="prompt-container-generation"):
                prompt_input_gen = gr.Textbox(
                    label="Enter your prompt here",
                    placeholder="e.g., How is your day so far?",
                    scale=4,
                    lines=5
                )
                submit_button_gen = gr.Button("Submit", variant="primary", scale=1)

                gr.Examples(
                    examples=[
                        # Example 1
                        [
                            """<context>
The Leaning Tower of Pisa is located in Pisa, Italy, and is famous for its unintended tilt.
Construction began in the 12th century and continued over 199 years due to wars.
Engineers have performed stabilization work to prevent further tilting.
Tourists can climb the spiral staircase to reach the top.

Pisa Cathedral and the Pisa Baptistery are part of the same Piazza dei Miracoli complex.
The tower is mainly constructed of white and grey marble.
</context>
<user>Where is the Leaning Tower of Pisa located?</user>""",
                        ],

                        # Example 2
                        [
                            """<context>
The human body contains several vital organs. The heart pumps blood throughout the body and has four chambers.
The lungs supply oxygen to the blood and remove carbon dioxide.
The liver processes nutrients and detoxifies chemicals.

The brain is part of the central nervous system and is responsible for cognition, memory, and motor control.
It is protected by the skull and surrounded by cerebrospinal fluid.
</context>
<user>What is the primary function of the brain?</user>""",
                        ],

                        # Example 3
                        [
                            """<context>
Saturn is a gas giant planet in our solar system known for its rings made mostly of ice particles.
It has 83 confirmed moons, one of the largest being Titan.
Voyager and Cassini have both studied Saturn up close.

Titan has a dense, nitrogen-rich atmosphere and surface lakes of liquid hydrocarbons.
Titan orbits Saturn roughly every 16 Earth days.
</context>
<user>How many moons does Saturn have?</user>""",
                        ],

                        # Example 4
                        [
                            """<context>
Marie Curie was a physicist and chemist known for pioneering research on radioactivity.
She discovered the elements polonium and radium.
She won two Nobel Prizes: one in Physics and one in Chemistry.

X-ray machines were rapidly adopted during World War I due to mobile radiography units designed by Curie.
Curie worked in Paris at the Sorbonne.

Her husband, Pierre Curie, collaborated closely with her scientific work.
</context>
<user>Which Nobel Prizes did Marie Curie receive?</user>""",
                        ],

                        # Example 5
                        [
                            """Only use the information from there provided documents: <context>
<doc id="a01" source="official">
Project Nebula – Internal Storage Overview (Rev 4.1, approved)
- Standard storage tier: S3-compatible bucket "neb-data-prod"
- Encryption: AES-256, server-side only
- Default retention: 365 days
- File size limit: 100 MB per object
</doc>

<doc id="a02" source="draft">
Nebula storage supports objects up to 120 MB. Retention currently 360 days.
</doc>

<doc id="a03" source="admin">
Retention policy (consolidated notes)
- Production: 365 days
- Exceptions exist for staging (180 days)
Note: Some early docs mention 366 days for leap-year alignment, but not applied.
</doc>

<doc id="a04" source="official">
Project Nebula Authentication
- Access token length: 48 characters
- Token pattern: alphanumeric only (A–Z, a–z, 0–9)
- Renew interval: 24 hours
</doc>

<doc id="a05">
Dev chatter (outdated)
Tokens used to be 50 chars long incl. dashes.
</doc>

<doc id="a06" source="official">
Nebula Logging Configuration
- Log archive: "neb-log-arch"
- Log retention: 30 days
- Compression: GZIP
</doc>

<doc id="a07">
Confluence page (typos)
File limt maybe 100mb but some says 101 MB durin beta.
</doc>

<doc id="a08" source="official">
Approved Interfaces
- Storage tier: S3-compatible
- Must use prefix: "neb/" for all stored objects
- Public access: forbidden
</doc>

<doc id="a09">
Old spreadsheet (contradict)
Retention: 366 days
</doc>

<doc id="a10" source="official">
Release Notes Snippet (final)
Reminder: Token length is fixed at 48. No punctuation allowed.
</doc>
</context>
<user>
What is the maximum file size per object in storage, the default retention period, and the token length?
</user>
Answer in one concise line and cite the doc ids you used in brackets.""",
                        ],
                    ],
                    inputs=[prompt_input_gen],
                    label="📌 Click an Example to Auto-Fill the Inputs"
                )

            output_display_gen = gr.HTML(label="Model Output")
            max_score_display_gen = gr.Label(label="Result")

            def clear_input():
                return ""

            submit_action = submit_button_gen.click(
                fn=generate_stream,
                inputs=prompt_input_gen,
                outputs=[output_display_gen, max_score_display_gen]
            )

demo.queue()
demo.launch()