from typing import Literal
from ragognizer.benchmarks.benchmark import Benchmark
from datasets import load_dataset

class RAGognize(Benchmark):
    def __init__(
        self,
        llm_name: Literal["Llama-2-7b-chat-hf", "Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.1", "Mistral-7B-Instruct-v0.3", "all"],
        ignore_post_hallucination: bool = False
    ) -> None:
        super().__init__(
            name=f"RAGognize ({llm_name})",
            ignore_post_hallucination=ignore_post_hallucination
        )
        self.llm_name = llm_name

    def get_entries(self, token_level: bool) -> list[dict[str, list[dict[str, str]] | str | list[str] | list[str, str | int]]]:
        dataset = load_dataset("F4biian/RAGognize")
        data = dataset["test"]

        all_entries = []

        for entry in data:
            user = entry["user_prompt"]
            doc  = entry["documents_str"]
            chat = entry["rag_prompt"]

            docs = []
            for document in entry["documents"]:
                docs.append(f"#### {document['title']}\n{document['text']}")

            # Get hallucination annotations
            responses = []
            annotations = []
            chats = []
            if self.llm_name == "all":
                for llm_name in entry["responses"]:
                    if llm_name == "golden_answer":
                        continue
                    response_dict = entry["responses"][llm_name]
                    responses.append(response_dict["text"])
                    chats.append(chat + [{
                        "content": response_dict["text"],
                        "role": "assistant",
                    }])
                    annotations.append(self._fill_non_hallu_annotations(response_dict["text"], response_dict["hallucinations"]))
            else:
                response_dict = entry["responses"][self.llm_name]
                responses.append(response_dict["text"])
                chats.append(chat + [{
                    "content": response_dict["text"],
                    "role": "assistant",
                }])
                annotations.append(self._fill_non_hallu_annotations(response_dict["text"], response_dict["hallucinations"]))

            for i in range(len(responses)):
                entry = {
                    "chat": chats[i],
                    "document": doc,
                    "documents": docs,
                    "user": user,
                    "response": responses[i],
                    "annotations": annotations[i],
                }
                all_entries.append(entry)

        if not token_level:
            self._aggregate_annotations_to_response_level(all_entries)

        return all_entries