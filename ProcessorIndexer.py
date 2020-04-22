#!/usr/bin/env python
# coding: utf-8

# In[1]:


from elasticsearch import Elasticsearch
import pandas as pd
import os
from elasticsearch import helpers
import numpy as np
from datetime import datetime
import collections
import spacy
from spacy_langdetect import LanguageDetector
import time
from multiprocessing import Process, Queue, Manager
from multiprocessing.pool import Pool
from functools import partial
from itertools import repeat

# In[2]:
from tqdm import tqdm


def doc_generator(df, index_name):
    # This function creates documents from dataframe rows, and then 
    # indexes those documents in ElasticSearch
    df_iter = df.iterrows()
    for index, document in df_iter:
        yield {
            "_index": index_name,
            "article": document.to_dict(),
        }
    return True


def safe_date(date_value):
    # This method fixes dates so that they don't break ES indexing
    try:
        return (
            pd.to_datetime(date_value) if not pd.isna(date_value)
            else datetime(1970, 1, 1, 0, 0)
        )
    except:
        return (datetime(2000, 1, 1, 0, 0))


def clean_and_enrich(df, nlp):
    # This method does basic pre-processing of the text to make it 
    # ready for indexing to ES.

    for i in df.columns:
        if i in ["filename", "image_url", "localpath", "title_page", "title_rss", "date_modify", "Unnamed: 0"]:
            df.drop(columns=[i], inplace=True)

    # Fill empty text with description, else with title, else drop
    df['maintext'].fillna(df['description'], inplace=True)
    df['maintext'].fillna(df['title'], inplace=True)
    df.dropna(subset=["maintext"], inplace=True)

    # Fix date columns. Convert columns to datetime, 
    # fill NaTs with date_download (Empty dates can break ES indexing)
    date_cols = ["date_publish", "date_download"]
    for col in date_cols:
        dates = df[col].to_list()
        dates_fixed = [str(i).replace("{'$date': '", "").replace("'}", "") if isinstance(i, str) else i for i in dates]
        df.loc[:, f"{col}_fixed"] = dates_fixed

    df['date_download'] = df['date_download_fixed']
    df['date_publish'] = df['date_publish_fixed']
    df.drop(columns=[str(i) + "_fixed" for i in date_cols], inplace=True)

    df['date_publish'].fillna(df['date_publish'].apply(safe_date), inplace=True)
    df['date_publish'] = df['date_publish'].apply(safe_date)
    df['date_download'] = df['date_download'].apply(safe_date)

    df.fillna("", inplace=True)

    # Drop duplicates by text
    df.drop_duplicates(subset="maintext", inplace=True)

    # Simple wash of text 
    df.replace('\n', ' ', regex=True, inplace=True)

    docs = list(nlp.pipe(df["maintext"].astype(str)))
    people = [[ent.text.strip('\'s').strip('’') for ent in doc.ents if (ent.label_ == "PERSON")] if doc._.language[
                                                                                                        "language"] == "en" else []
              for doc in docs]
    places = [[ent.text for ent in doc.ents if (ent.label_ == "LOC" or ent.label_ == "FAC" or ent.label_ == "GPE")] if
              doc._.language["language"] == "en" else [] for doc in docs]
    orgs = [[ent.text for ent in doc.ents if (ent.label_ == "ORG")] if doc._.language["language"] == "en" else [] for
            doc in docs]
    concepts = [[ent.text for ent in doc.ents if
                 ent.label_ not in ["LOC", "ORG", "FACE", "PERSON", "GPE", "PERCENT", "ORDINAL", "CARDINAL"]] if
                doc._.language["language"] == "en" else [] for doc in docs]
    languages = [doc._.language["language"] for doc in docs]
    lemmas = [[token.lemma_ for token in doc] for doc in docs]

    df["language"] = languages
    df["people"] = people
    df["places"] = places
    df["orgs"] = orgs
    df["concepts"] = concepts
    df["lemma"] = lemmas

    return (df)


def scan_for_files(directories):
    filesDone = [directories[0] + f for f in os.listdir(directories[0])]
    return (filesDone)


def cleanup(filesToDelete):
    for i in filesToDelete:
        os.remove(i)


def index_to_es(df, index_name):
    es = Elasticsearch("10.94.253.5")
    helpers.bulk(es, doc_generator(df, index_name))


# In[5]:

def processCsvFile(tuple):
    try:
        df = pd.read_csv(tuple[0], engine="python")
        print("Successfully read file", tuple[0])
    except:
        print("Problem with file", tuple[0])
        return
    df = clean_and_enrich(df=df, nlp=tuple[1])

    # Index into Elasticsearch
    index_to_es(df, index_name="november2019")


def main(interval=60):
    nlp = spacy.load("en_core_web_lg")
    nlp.max_length = 2000000
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
    directories = ["/data/tmp/"]

    pool = Pool(6)  # 6 Cores for starters
    while True:
        # Get files to process
        filesToProcess = scan_for_files(directories)

        pool.map(processCsvFile, zip(tqdm(filesToProcess), repeat(nlp)))
        pool.close()
        pool.join()

        # Delete files
        cleanup(filesToProcess)

        # Wait a minute!
        time.sleep(interval)


# In[4]:


if __name__ == "__main__":
    main()
