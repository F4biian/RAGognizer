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
    
def generate_stream(system_prompt, prompt):
    if not prompt:
        yield "", "Max Score: N/A"
        return

    chat = []
    if system_prompt and system_prompt.strip():
        chat.append({"role": "system", "content": system_prompt})
    chat.append({"role": "user", "content": prompt})
    
    max_score = 0
    with model_lock:
        full_styled_response = ""
        
        for tok in detector.stream_generate(
            chat,
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
  - **0.00 – 0.40** — White
  - **0.40 – 0.65** — Yellow
  - **0.65 – 0.85** — Orange
  - **0.85 – 1.00** — Red
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

            output_display_detect = gr.HTML(label="Analyzed Response")
            max_score_display = gr.Label(label="Result")

            analyze_button.click(
                fn=detect_and_color,
                inputs=[system_prompt_input, prompt_input_detect, response_input],
                outputs=[output_display_detect, max_score_display]
            )
            
            gr.Examples(
                examples=[
                    # Example 0: Simple Fictional (No Hallucination)
                    [
                        "Answer strictly based on the provided context.",
                        "Context: The planet Zorblax has 3 moons and is made entirely of green cheese.\n\nQuestion: How many moons does Zorblax have?",
                        "Zorblax has 3 moons.",
                        "✅"
                    ],

                    # Example 1: Simple Fictional (Entity Hallucination)
                    [
                        "Answer strictly based on the provided context.",
                        "Context: The CEO of CyberDyne Systems is John Connor.\n\nQuestion: Who is the CEO of CyberDyne?",
                        "The CEO of CyberDyne Systems is Michael Connor.",
                        "❌"
                    ],

                    # Example 2: Parametric Interference (Hallucination; The "Paris" test)
                    [
                        "Answer strictly based on the provided context. Do not use outside knowledge.",
                        "Context: Due to a recent treaty in 2030, the capital of France was officially moved to Lyon.\n\nQuestion: What is the capital of France?",
                        "The capital of France is Paris.",
                        "❌"
                    ],

                    # Example 3: Parametric Interference (No Hallucination; The "Pisa" test)
                    [
                        "Answer strictly based on the provided context.",
                        "Context: The Leaning Tower of Pisa is located in Pisa, Italy. It was built in the 12th century.\n\nQuestion: Where is the Leaning Tower of Pisa?",
                        "The Leaning Tower of Pisa is located in Pisa, Italy.",
                        "✅"
                    ],

                    # Example 4: Difficult Fictional (Subtle Hallucination; Altered conditions)
                    [
                        "Use only the context provided.",
                        "Context: The XJ-9000 smartphone features a 6-inch screen, 12 hours of battery life, and is water-resistant up to 2 meters for 30 minutes.\n\nQuestion: What are the water resistance specs of the XJ-9000?",
                        "The XJ-9000 is water-resistant up to 2 meters for an hour. Do you want to know more about this smartphone?",
                        "❌"
                    ],

                    # Example 5: Difficult Fictional (Extrinsic Hallucination; Adding plausible unstated info)
                    [
                        "Use only the context provided.",
                        "Context: Dr. Aris Thorne discovered the luminescent moss 'Glowleaf' in the caves of Mount Ebon. The moss emits a faint blue light when exposed to moisture.\n\nQuestion: Tell me about Dr. Thorne's discovery.",
                        "Dr. Aris Thorne discovered Glowleaf moss in the caves of Mount Ebon. It emits a blue light when wet and is highly toxic to touch.",
                        "❌"
                    ],

                    # Example 6: Long Context (Subtle Hallucinations with Distractors)
                    [
                        "Use only the context provided.",
                        "Context: Project Chronos was initiated in 2042 by Apex Dynamics to explore the Mariana Trench using the 'Abyss-1' submersible. The team consisted of 12 marine biologists and 4 engineers. The primary objective was to locate the rare 'Lumina' coral. During the 3-week expedition, they discovered a new species of blind shrimp, but the Lumina coral was not found. A secondary mission, Project Helios, was planned for 2045 to explore the Arctic Ocean. The Abyss-1 submersible suffered minor thruster damage on day 14, forcing the team to rely on backup navigation.\n\nQuestion: Summarize the details and outcome of Project Chronos.",
                        "Project Chronos was launched in 2042 by Apex Dynamics to explore the Mariana Trench. The team of 16 members spent 3 weeks underwater. During the mission, the Abyss-1 submersible suffered major thruster damage on day 14. Ultimately, they successfully located the Lumina coral and a new species of blind shrimp.",
                        "❌"
                    ],

                    # Example 7: Long Context (Clean Response)
                    [
                        "Use only the context provided.",
                        "Context: Project Chronos was initiated in 2042 by Apex Dynamics to explore the Mariana Trench using the 'Abyss-1' submersible. The team consisted of 12 marine biologists and 4 engineers. The primary objective was to locate the rare 'Lumina' coral. During the 3-week expedition, they discovered a new species of blind shrimp, but the Lumina coral was not found. A secondary mission, Project Helios, was planned for 2045 to explore the Arctic Ocean. The Abyss-1 submersible suffered minor thruster damage on day 14, forcing the team to rely on backup navigation.\n\nQuestion: Summarize the details and outcome of Project Chronos.",
                        "Project Chronos was launched in 2042 by Apex Dynamics to explore the Mariana Trench using the Abyss-1 submersible. The 16-person team spent 3 weeks searching for the Lumina coral, which they did not find. However, they did discover a new species of blind shrimp.",
                        "✅"
                    ],

                    # Example 8: Unanswerable Context (Hallucination falling for distractor)
                    [
                        "Answer strictly based on the provided context.",
                        "Context: The Orion V2 smartwatch features a titanium casing, a heart-rate monitor, and a 4-day battery life. It is available in Silver and Midnight Black. The previous model, Orion V1, included a sleep tracking feature that was highly praised by users.\n\nQuestion: Does the Orion V2 smartwatch have sleep tracking?",
                        "Yes, the Orion V2 smartwatch includes a sleep tracking feature that is highly praised by users.",
                        "❌"
                    ],

                    # Example 9: Unanswerable Context (Clean Response)
                    [
                        "Answer strictly based on the provided context. If the answer is not in the context, say 'I cannot answer'.",
                        "Context: The Orion V2 smartwatch features a titanium casing, a heart-rate monitor, and a 4-day battery life. It is available in Silver and Midnight Black. The previous model, Orion V1, included a sleep tracking feature that was highly praised by users.\n\nQuestion: Does the Orion V2 smartwatch have sleep tracking?",
                        "Sorry, but I cannot answer that based on my knowledge. The text only mentions that the previous model, Orion V1, had a sleep tracking feature, but it does not say if the Orion V2 has it.",
                        "✅"
                    ],
                ],
                inputs=[system_prompt_input, prompt_input_detect, response_input, hallucination_status],
                label="📌 Click an Example to Auto-Fill (Note: These examples are manually made up to demonstrate specific hallucination types, including parametric interference, and were not generated by an LLM. They serve as synthetic experimental samples.)"
            )


        with gr.TabItem("Generation", id=1):
            gr.Markdown("Enter a (RAG) prompt and press Enter or click 'Submit'. The response will be shown below, with each token's background color corresponding to its hallucination score.")
            with gr.Column(elem_id="prompt-container-generation"):
                system_prompt_input_gen = gr.Textbox(
                    label="Optional System Prompt", 
                    placeholder="e.g., You are a helpful assistant.", 
                    lines=3
                )
                prompt_input_gen = gr.Textbox(
                    label="Enter your prompt here",
                    placeholder="e.g., How is your day so far?",
                    lines=5
                )
                submit_button_gen = gr.Button("Submit", variant="primary")


            output_display_gen = gr.HTML(label="Model Output")
            max_score_display_gen = gr.Label(label="Result")

            def clear_input():
                return ""

            submit_action = submit_button_gen.click(
                fn=generate_stream,
                inputs=[system_prompt_input_gen, prompt_input_gen],
                outputs=[output_display_gen, max_score_display_gen]
            )
            
            gr.Examples(
                examples=[
                    # Example 0: Simple Fictional
                    [
                        "Answer strictly based on the provided context.",
                        "Context: The planet Zorblax has 3 moons and is made entirely of green cheese.\n\nQuestion: How many moons does Zorblax have?"
                    ],
                    # Example 1: Parametric Interference (Paris)
                    [
                        "Answer strictly based on the provided context. Do not use outside knowledge.",
                        "Context: Due to a recent treaty in 2030, the capital of France was officially moved to Lyon.\n\nQuestion: What is the capital of France?"
                    ],
                    # Example 2: Parametric Interference (Pisa)
                    [
                        "Answer strictly based on the provided context.",
                        "Context: The Leaning Tower of Pisa is located in Pisa, Italy. It was built in the 12th century.\n\nQuestion: Where is the Leaning Tower of Pisa?"
                    ],
                    # Example 3: Difficult Fictional (Conditions)
                    [
                        "Use only the context provided.",
                        "Context: The XJ-9000 smartphone features a 6-inch screen, 12 hours of battery life, and is water-resistant up to 2 meters for 30 minutes.\n\nQuestion: What are the water resistance specs of the XJ-9000?"
                    ],
                    # Example 4: Difficult Fictional (Extrinsic)
                    [
                        "Use only the context provided.",
                        "Context: Dr. Aris Thorne discovered the luminescent moss 'Glowleaf' in the caves of Mount Ebon. The moss emits a faint blue light when exposed to moisture.\n\nQuestion: Tell me about Dr. Thorne's discovery."
                    ],
                    # Example 5: Long Context with Distractors
                    [
                        "Use only the context provided.",
                        "Context: Project Chronos was initiated in 2042 by Apex Dynamics to explore the Mariana Trench using the 'Abyss-1' submersible. The team consisted of 12 marine biologists and 4 engineers. The primary objective was to locate the rare 'Lumina' coral. During the 3-week expedition, they discovered a new species of blind shrimp, but the Lumina coral was not found. A secondary mission, Project Helios, was planned for 2045 to explore the Arctic Ocean. The Abyss-1 submersible suffered minor thruster damage on day 14, forcing the team to rely on backup navigation.\n\nQuestion: Summarize the details and outcome of Project Chronos."
                    ],
                    # Example 6: Unanswerable Context with Distractor
                    [
                        "Answer strictly based on the provided context.",
                        "Context: The Orion V2 smartwatch features a titanium casing, a heart-rate monitor, and a 4-day battery life. It is available in Silver and Midnight Black. The previous model, Orion V1, included a sleep tracking feature that was highly praised by users.\n\nQuestion: Does the Orion V2 smartwatch have sleep tracking?"
                    ]
                ],
                inputs=[system_prompt_input_gen, prompt_input_gen],
                label="📌 Click an Example to Auto-Fill (Note: These examples are manually made up to demonstrate specific hallucination types, including parametric interference, and were not generated by an LLM. They serve as synthetic experimental samples.)"
            )


demo.queue()
demo.launch()
