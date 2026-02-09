from typing import Literal
from ragognizer.benchmarks.benchmark import Benchmark
from datasets import load_dataset

class RAGognize(Benchmark):
    def __init__(
        self,
        llm_name: Literal["Llama-2-7b-chat-hf", "Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.1", "Mistral-7B-Instruct-v0.3", "all"],
        train_split: bool = False,
        include_samples: bool = True,
    ):
        super().__init__(
            name=f"RAGognize ({llm_name})",
            # ignore_post_hallucination=True
        )
        self.llm_name = llm_name
        self.train_split = train_split
        self.include_samples = include_samples

        if train_split:
            print("WARNING: You are using the train split of the RAGognize dataset. Do not use this for benchmarking or reporting any interpretable performances!")

    def get_entries(self, token_level: bool) -> list[dict[str, list[dict[str, str]] | str | list[str] | list[str, str | int]]]:
        if self.include_samples:
            dataset = load_dataset("F4biian/RAGognize-with-samples-test")
            if self.train_split:
                raise Exception("There is not train split for dataset 'F4biian/RAGognize-with-samples-test'!")
            data = dataset["test"]
        else:
            dataset = load_dataset("F4biian/RAGognize")
            data = dataset["test"] if not self.train_split else dataset["train"]

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
            response_samples = []
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

                    samples = None
                    if self.include_samples:
                        samples = [s["output"] for s in response_dict["samples"]]
                    response_samples.append(samples)
            else:
                response_dict = entry["responses"][self.llm_name]
                responses.append(response_dict["text"])
                chats.append(chat + [{
                    "content": response_dict["text"],
                    "role": "assistant",
                }])
                annotations.append(self._fill_non_hallu_annotations(response_dict["text"], response_dict["hallucinations"]))

                samples = None
                if self.include_samples:
                    samples = [s["output"] for s in response_dict["samples"]]
                response_samples.append(samples)

            for i in range(len(responses)):
                entry = {
                    "chat": chats[i],
                    "document": doc,
                    "documents": docs,
                    "user": user,
                    "response": responses[i],
                    "response_samples": response_samples[i],
                    "annotations": annotations[i],
                    "llm_name": None,
                }
                all_entries.append(entry)

        if not token_level:
            self._aggregate_annotations_to_response_level(all_entries)

        return all_entries
    
    def get_train_entries(
        self,
        token_level: bool
    ) -> list[dict[str, str | list[str]]] | None:
        temp = self.train_split
        self.train_split = True
        entries = self.get_entries(token_level=token_level)
        self.train_split = temp
        return entries

    def generate_samples(
        self,
        entries: list,
    ) -> list[dict[str, str | list[str]]]:
        return entries
    

if __name__ == "__main__":
    rg = RAGognize(llm_name="all", train_split=False)
    entries = rg.get_all_entries(token_level=True)
    print(entries[42]) # {'chat': [{'content': 'You are a friendly [...] restaurant is Line Cook.', 'label': 0}]}
    print(len(entries)) # 11124