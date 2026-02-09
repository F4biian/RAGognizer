########################################################################################
# IMPORTS

import json
import os
import pandas as pd
import random
from pprint import pprint
from typing import List

########################################################################################


CURR_DIR = os.path.dirname(os.path.realpath(__file__))
ARTICLES_DIR = os.path.join(CURR_DIR, "data", "articles")
DATA_DIR = os.path.join(CURR_DIR, "data")

date = os.getenv("CUTOFF_DATE", None)
if date is None:
    raise ValueError("Please set a suitable CUTOFF_DATE as an env variable, e.g., \"2024-05-23 00:00:00\"")

CUTOFF_DATE = pd.to_datetime(os.getenv("CUTOFF_DATE", "2024-05-23 00:00:00"))

def count_passages(articles):
    count = 0
    for art in articles:
        count += len(art["passage_data"])
    return count

def filter_article(art) -> dict:
    art_copy = art.copy()
    art_copy["passage_data"] = []

    # If one other version (in another language) has been created before the cutoff date, don't use it.
    if art["earliest_creation_date"] is not None and pd.to_datetime(art["earliest_creation_date"]) < CUTOFF_DATE:
        return None
    
    # If the English article has been created before the cutoff date, don't use it. (not possible due to the scraping technique, but safety first)
    if pd.to_datetime(art["created_en"]) < CUTOFF_DATE:
        return None
    
    # If there have been difficulties to uniquely identify references, don't use it.
    if art["removed_duplicates"]:
        return None
    
    # If the article's title contains the word 'list', don't use it.
    if "list" in art["title"].lower():
        return None

    # Now: Check every passage whether it can be used
    for passage in art["passage_data"]:

        # If the passage contains another Wikipedia article, don't use it.
        if passage["contains_article"]:
            continue

        # If the passage is not backed by references, don't use it. (not possible due to scraping technique, but safey first)
        if len(passage["references"]) <= 0:
            continue
        
        # If the passage is too short, don't use it.
        if passage["end"] - passage["start"] < 30:
            continue

        # Use this passage, if its `access_date` and `date` are after the cutoff date and if it has an `archive_date`, then this should also be after the cutoff date.
        for ref in passage["references"]:
            if ref["access_date"] is None or pd.to_datetime(ref["access_date"]) < CUTOFF_DATE:
                break
            if ref["date"] is None or pd.to_datetime(ref["date"]) < CUTOFF_DATE:
                break
            if ref["archive_date"] is not None and pd.to_datetime(ref["archive_date"]) < CUTOFF_DATE:
                break
        else:
            art_copy["passage_data"].append(passage)

    # If all passage have been removed, the entire article will not be used.
    # If there are still to many passages, the entire article will not be used. (Assuming it indicates low quality passages)
    if len(art_copy["passage_data"]) <= 0 or len(art_copy["passage_data"]) >= 10:
        return None

    # If all criteria have been met, use this article with its remaining passage.
    return art_copy

def get_articles() -> List[dict]:
    articles = []

    # Read all json files and put them into the articles list
    for file in os.listdir(ARTICLES_DIR):
        if file.endswith(".json"):
            with open(os.path.join(ARTICLES_DIR, file), "r") as file:
                articles.extend(json.load(file))
    
    return articles

def get_suitable_articles(articles: List[dict]=None):
    if articles is None:
        articles = get_articles()
        
    # Only use those article that meet certain criteria
    suitable_articles = []
    for art in articles:
        filtered_art = filter_article(art)
        if filtered_art:
            suitable_articles.append(filtered_art)

    return suitable_articles

if __name__ == "__main__":
    articles = get_articles()
    suitable_articles = get_suitable_articles(articles)

    rand_art = random.choice(suitable_articles)
    print("#"*50)
    print("URL:", rand_art["url"])
    print("-"*50)
    for passage in rand_art["passage_data"]:
        print("Passage:", rand_art["content"][passage["start"]:passage["end"]])
        print()
        pprint(passage["references"])
        print("-"*50)

    print()
    print(f"Before: {len(articles)}\tarticles with\t{count_passages(articles)} passages.")
    print(f" After: {len(suitable_articles)}\tarticles with\t{count_passages(suitable_articles)} passages.")
    print()

    with open(os.path.join(DATA_DIR, "suitable_articles.json"), "w") as file:
        json.dump(suitable_articles, file, indent=4, ensure_ascii=False)