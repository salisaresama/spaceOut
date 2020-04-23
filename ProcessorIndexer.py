#!/usr/bin/env python
# coding: utf-8

# In[1]:
import threading
from queue import Queue

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


def init_langdetect():
    langdetect = spacy.blank("xx")
    sentencizer = langdetect.create_pipe("sentencizer")
    langdetect.add_pipe(sentencizer)
    langdetect.add_pipe(LanguageDetector(), name='language_detector')
    return langdetect


def get_language(model, text):
    return model(text)._.language["language"]


def process_csv_file(file):
    try:
        df = pd.read_csv(file, engine="python")
        # Find all languages that are present in file. Each line might have different language
        # df['language'] = get_language(langdetect, df['maintext'])
        df['language'] = [get_language(langdetect, str(text)) for text in df["maintext"].to_list()]
        # Partition dataframe by languages
        unique_languages = df['language'].unique()
        print("INPUT: Successfully read file", file, "with languages", unique_languages)
        for language in unique_languages:
            # Fixme: Pass language to this method. and check what is inside.
            tmpdf = clean_and_enrich(df=df[df['language'] == language], nlp=nlp_models[language])

            # Index into Elasticsearch
            index_to_es(tmpdf, index_name="november2019")

        print("Cleaning up file", file)
        os.remove(file)

        return file
    except:
        print("INPUT: Problem with file", file)
        return


class WorkerThread(threading.Thread):
    def __init__(self, id, q):
        threading.Thread.__init__(self)
        self.threadID = id
        self.name = "Worker" + str(id)
        self.q = q

    def run(self):
        print("Starting " + self.name)
        while not exitFlag:
            if not filesQueue.empty():
                queueLock.acquire()
                file_name = self.q.get()
                queueLock.release()
                process_csv_file(file_name)
        print("Exiting " + self.name)


def main(interval=60):
    number_of_parallel_threads = 20
    global nlp_models, queueLock, exitFlag, filesQueue, langdetect
    langdetect = init_langdetect()
    exitFlag = False
    queueLock = threading.Lock()
    nlp_models = {
        "en": spacy.load("en_core_web_lg"),
        "de": spacy.load("de_core_news_md"),
        "el": spacy.load("el_core_news_md"),
        "es": spacy.load("es_core_news_md"),
        "fr": spacy.load("fr_core_news_md"),
        "it": spacy.load("it_core_news_sm"),
        "lt": spacy.load("lt_core_news_sm"),
        "nb": spacy.load("nb_core_news_sm"),
        "nl": spacy.load("nl_core_news_sm"),
        "pt": spacy.load("pt_core_news_sm"),
    }
    filesQueue = Queue(number_of_parallel_threads * 2)
    threads = []
    # Create new threads
    for id in range(number_of_parallel_threads):
        thread = WorkerThread("Worker" + str(id), filesQueue)
        thread.start()
        threads.append(thread)

    for nlp in nlp_models.values():
        nlp.max_length = 40000

    directories = ["/data/tmp/"]

    print("Monitoring directories: ", directories)

    while True:
        files_to_process = scan_for_files(directories)[:filesQueue.qsize()]
        for f in files_to_process:
            filesQueue.put(f)

        print("Found files to process: ", files_to_process)

        # Wait a minute!
        print("Waiting a minute to start all over")
        # FIXME: This is not efficient. We need to continue all the time, just need to make sure not to add same file
        #  twice to the queue
        filesQueue.join()
        time.sleep(interval)


if __name__ == "__main__":
    main()
