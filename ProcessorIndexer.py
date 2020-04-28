#!/usr/bin/env python
# coding: utf-8

# In[1]:
import re
import sys
import threading
import traceback
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

    df['maintext'] = df['maintext'].replace(to_replace='^https?:\/\/.*[\r\n]*', value='', regex=True)

    # Simple wash of text 
    df.replace('\n', ' ', regex=True, inplace=True)

    if not nlp is None:
        docs = list(nlp.pipe(df["maintext"].astype(str)))
        people = [[ent.text.strip('\'s').strip('’') for ent in doc.ents if (ent.label_ == "PERSON")] for doc in docs]
        places = [[ent.text for ent in doc.ents if (ent.label_ == "LOC" or ent.label_ == "FAC" or ent.label_ == "GPE")]
                  for doc in docs]
        orgs = [[ent.text for ent in doc.ents if (ent.label_ == "ORG")] for doc in docs]
        concepts = [[ent.text for ent in doc.ents if
                     ent.label_ not in ["LOC", "ORG", "FACE", "PERSON", "GPE", "PERCENT", "ORDINAL", "CARDINAL"]] for
                    doc in docs]

        # Language was already detected previously
        # languages = [doc._.language["language"] for doc in docs]
        lemmas = [
            [token.lemma_ for token in doc if not token.is_stop and re.search("[^\W\d_]", token.text, flags=re.UNICODE)]
            for doc in docs]

        # df["language"] = languages
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
        print("INPUT: Reading file", file, "to dataframe", flush=True)
        df = pd.read_csv(file, engine="python")
        print("INPUT: Reading file", file, "done - OK", flush=True)
        # Find all languages that are present in file. Each line might have different language
        # Synthesize longer texts for more accurate language detection
        texts = [" ".join([str(x) + str(y) + str(z) for x, y, z in zip(df["title"].to_list(), df["description"].to_list(), df["maintext"].to_list())]
        df['language'] = [get_language(langdetect, text) for text in texts]
        # Partition dataframe by languages
        unique_languages = df['language'].unique()
        print("INPUT: In file", file, "are languages", unique_languages, flush=True)
        for language in unique_languages:
            # Fixme: Pass language to this method. and check what is inside.
            tmpdf = clean_and_enrich(df[df['language'] == language], nlp_models.get(language))

            # Index into Elasticsearch
            index_to_es(tmpdf, index_name="november2019")

        print("Cleaning up file", file, flush=True)
        os.remove(file)
    except:
        print("INPUT: Problem with file", file, flush=True)
        traceback.print_exc()
        sys.stdout.flush()


class WorkerThread(threading.Thread):
    def __init__(self, id, input_queue, processed_queue):
        threading.Thread.__init__(self)
        self.threadID = id
        self.name = "Worker" + str(id)
        self.input_queue = input_queue
        self.processed_queue = processed_queue

    def run(self):
        print("Starting " + self.name)
        while not exitFlag:
            if not self.input_queue.empty():
                file_name = self.input_queue.get()
                process_csv_file(file_name)
                self.processed_queue.put(file_name)
        print("Exiting " + self.name)


class ModelLoaderThread(threading.Thread):
    def __init__(self, lang_model_name):
        threading.Thread.__init__(self)
        self.threadID = lang_model_name
        self.name = "ModelLoader: " + str(lang_model_name)
        self.loaded_model = None

    def run(self):
        print("INFO: Loading nlp model: ", self.threadID, flush=True)
        self.loaded_model = spacy.load(self.threadID)
        print("INFO: Loaded nlp model: ", self.threadID, flush=True)

    def get_loaded_model(self):
        return self.loaded_model


def main(interval=60):
    number_of_parallel_threads = 30
    global nlp_models, exitFlag, langdetect
    langdetect = init_langdetect()
    exitFlag = False  # Not used currently. Set it to true to make threads terminate gracefully
    nlp_models_raw = {
        "en": "en_core_web_lg",
        "de": "de_core_news_md",
        "el": "el_core_news_md",
        "es": "es_core_news_md",
        "fr": "fr_core_news_md",
        "it": "it_core_news_sm",
        "lt": "lt_core_news_sm",
        "nb": "nb_core_news_sm",
        "nl": "nl_core_news_sm",
        "pt": "pt_core_news_sm",
    }
    print("INFO: Loading nlp models: ", nlp_models_raw, flush=True)

    nlp_models = dict()
    model_loading_threads = dict()
    for lang_code, lang_model_name in nlp_models_raw.items():
        thread = ModelLoaderThread(lang_model_name)
        model_loading_threads[lang_code] = thread
        thread.start()

    for lang_code, loading_thread in model_loading_threads.items():
        loading_thread.join()
        nlp_models[lang_code] = loading_thread.get_loaded_model()

    print("INFO: NLP models loaded successfully: ", nlp_models_raw, flush=True)

    # Queue with files that needs to be processed
    input_files_queue = Queue(number_of_parallel_threads * 2)
    # Queue with files that are already processed
    done_files_queue = Queue(number_of_parallel_threads * 3)
    # Files that are already in queue or currently processing.
    files_in_processing = set()
    threads = []
    # Create new threads
    print("INFO: Preparing ", number_of_parallel_threads, "worker threads", flush=True)
    for id in range(number_of_parallel_threads):
        thread = WorkerThread("Worker" + str(id), input_files_queue, done_files_queue)
        thread.start()
        threads.append(thread)

    for nlp in nlp_models.values():
        nlp.max_length = 40000

    directories = ["/data/tmp/"]

    print("INFO: Monitoring directories: ", directories, flush=True)

    while True:
        files_to_process = scan_for_files(directories)
        # All files that are processed by this time should be deleted from `files_in_processing` set
        while True:
            try:
                files_in_processing.remove(done_files_queue.get_nowait())
            except:
                # When no more files in queue - continue further
                break

        # Add new files from directory to processing queue and "mark" them as in-processing
        # When no more files can be added to queue - it is full, continue loop
        for f in files_to_process:
            if not f in files_in_processing:
                try:
                    input_files_queue.put_nowait(f)
                    files_in_processing.add(f)
                except:
                    # If no more free space in queue - continue further
                    break

        print("Files currently in processing: ", files_in_processing, flush=True)

        # Wait a minute!
        print("Waiting for ", interval, "seconds to start all over", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    main()
