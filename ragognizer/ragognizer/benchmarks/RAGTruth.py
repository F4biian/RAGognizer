import json
import os
from pathlib import Path
from typing import Literal

import requests
from ragognizer.benchmarks.benchmark import Benchmark

def get_cache_directory():
    """
    Determines the appropriate cache directory.
    
    1. Checks for the HF_HOME environment variable.
    2. If not found, uses the default ~/.cache directory.
    
    Creates a subdirectory for this specific tool to avoid cluttering the root.
    """
    # Hugging Face's convention is to use HF_HOME.
    hf_home = os.getenv("HF_HOME")
    
    if hf_home:
        # If HF_HOME is set, use it
        base_cache_dir = Path(hf_home)
    else:
        # Otherwise, fall back to the standard .cache directory in the user's home
        base_cache_dir = Path.home() / ".cache"
        
    # We create a specific subdirectory for our downloader to keep things organized
    tool_cache_dir = base_cache_dir / "python_repo_downloader"
    
    return tool_cache_dir

def download_file_with_cache(owner, repo, version, file_path):
    """
    Downloads a specific version of a file from a public GitHub repository,
    using a local cache to avoid re-downloading.

    The cache location respects the HF_HOME environment variable.

    Args:
        owner (str): The owner or organization of the repository.
        repo (str): The name of the repository.
        version (str): The commit hash or tag for a fixed version.
        file_path (str): The path to the file within the repository.

    Returns:
        pathlib.Path: The local path to the (potentially cached) file, or None if download fails.
    """
    cache_root = get_cache_directory()
    
    # 1. Construct a unique local path based on the file's identity
    # e.g., ~/.cache/python_repo_downloader/psf/requests/a087.../requests/api.py
    local_filepath = cache_root / owner / repo / version / file_path
    
    # 2. Check if the file already exists in the cache
    if local_filepath.exists():
        print(f"File found in cache. Loading from: {local_filepath}")
        return local_filepath

    # 3. If not in cache, proceed with the download
    print(f"File not in cache. Downloading...")
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{version}/{file_path}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes

        # 4. Create the parent directories if they don't exist
        # This is crucial for the nested cache structure.
        local_filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 5. Save the downloaded content to the cache location
        with open(local_filepath, 'wb') as f:
            f.write(response.content)
            
        print(f"Successfully downloaded and cached to: {local_filepath}")
        return local_filepath

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during download: {e}")
        return None



class RAGTruth(Benchmark):
    def __init__(
        self,
        llm_name: Literal["gpt-4-0613", "gpt-3.5-turbo-0613", "mistral-7B-instruct", "llama-2-7b-chat", "llama-2-13b-chat", "llama-2-70b-chat", "all"],
        task: Literal["Summary", "Data2txt", "QA"] = "QA",
        is_test: bool = True,
    ):
        super().__init__(
            name=f"RAGTruth ({task} | {llm_name})",
            # ignore_post_hallucination=True
        )
        self.llm_name = llm_name
        self.task = task
        self.is_test = is_test

    def get_entries(self, token_level: bool) -> list[dict[str, list[dict[str, str]] | str | list[str] | list[str, str | int]]]:
        responses_file = download_file_with_cache(
            owner="ParticleMedia",
            repo="RAGTruth",
            version="1d52a81c9e28e79e252a1945d858eb8dfd975c23",
            file_path="dataset/response.jsonl"
        )
        source_info_file = download_file_with_cache(
            owner="ParticleMedia",
            repo="RAGTruth",
            version="1d52a81c9e28e79e252a1945d858eb8dfd975c23",
            file_path="dataset/source_info.jsonl"
        )

        responses = []
        with open(responses_file, "r") as file:
            for row in file:
                if row:
                    responses.append(json.loads(row))

        source_info_by_id = {}
        with open(source_info_file, "r") as file:
            for row in file:
                if row:
                    d = json.loads(row)
                    source_info_by_id[d["source_id"]] = d

        all_entries = []

        for response_entry in responses:
            if self.is_test:
                if response_entry["split"] != "test":
                    continue
            else:
                if response_entry["split"] == "test":
                    continue

            # Ignore other models if llm_name is set
            if self.llm_name != "all":
                if response_entry["model"] != self.llm_name:
                    continue

            source_info_entry = source_info_by_id[response_entry["source_id"]]
            
            if self.task != source_info_entry["task_type"]:
                continue

            response = response_entry["response"]


            if self.task == "QA":
                document = source_info_entry["source_info"]["passages"]

                documents = []
                for doc in document.split("\npassage "):
                    if doc.strip():
                        if not doc.startswith("passage "):
                            doc = f"passage {doc}"
                        documents.append(doc.strip())

                user = source_info_entry["source_info"]["question"]
            elif self.task == "Data2txt":
                prompt = source_info_entry["prompt"]
                user = prompt.split("Structured data:\n")[0].lstrip("Instruction:").strip()
                document = "Structured data:\n" + prompt.split("Structured data:\n", 1)[1].split("\nOverview:", 1)[0].strip()
                documents = [document]
            elif self.task == "Summary":
                prompt = source_info_entry["prompt"]
                document = source_info_entry["source_info"]
                user = prompt.split(document)[0].strip()
                documents = [document]

            chat = [
                {
                    "role": "user",
                    "content": source_info_entry["prompt"]
                },
                {
                    "role": "assistant",
                    "content": response
                }
            ]

            annotations = response_entry["labels"]

            # Only keep start, end and text
            new_annotations = []
            for anno in annotations:
                new_annotations.append({
                    "start": anno["start"],
                    "end": anno["end"],
                    "text": anno["text"],
                })
            annotations = new_annotations

            llm_name = {
                "gpt-4-0613":          "openai/gpt-4",
                "gpt-3.5-turbo-0613":  "openai/gpt-3.5-turbo-0613",
                "llama-2-70b-chat":    "meta-llama/llama-2-70b-chat",
                "llama-2-13b-chat":    "meta-llama/llama-2-13b-chat",
                "llama-2-7b-chat":     "meta-llama/llama-2-7b-chat",
                "mistral-7B-instruct": "mistralai/mistral-7b-instruct-v0.1"
            }[response_entry["model"]]

            entry = {
                "chat": chat,
                "document": document,
                "documents": documents,
                "user": user,
                "response": response,
                "response_samples": None,
                "annotations": self._fill_non_hallu_annotations(response=response, hallucinations=annotations),
                "llm_name": llm_name,
            }
            all_entries.append(entry)

        if not token_level:
            self._aggregate_annotations_to_response_level(all_entries)

        return all_entries
    
    def get_train_entries(
        self,
        token_level: bool
    ) -> list[dict[str, str | list[str]]] | None:
        temp = self.is_test
        self.is_test = False
        entries = self.get_entries(token_level=token_level)
        self.is_test = temp
        return entries

if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    r = RAGTruth(llm_name="llama-2-7b-chat", task="Data2txt")

    e = r.get_entries(token_level=True)

    print(len(e))