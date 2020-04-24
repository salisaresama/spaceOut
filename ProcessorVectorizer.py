import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import time
import argparse
from laserembeddings import Laser
from elasticsearch import Elasticsearch
from typing import Generator, Any, Dict, List, Optional
from elasticsearch import helpers
from datetime import datetime


class Vectorizer(object):
    
    __valid_methods = ['muse', 'laser', 'use']
    
    def __init__(self, method: str = 'muse', 
                 path_to_model: str = '../../../../models/muse/'):
        
        assert method in self.__valid_methods
        
        self.method = method
        
        if self.method == 'muse':
            self.__vectorizer = hub.load(path_to_model)
        elif self.method == 'use':
            self.__vectorizer = hub.load(path_to_model)
        elif self.method == 'laser':
            self.__vectorizer = Laser()
        else:
            self.__vectorizer = None
    
    def vectorize(self, docs: List[str], **kwargs) -> List[List[float]]:
        
        if self.method in {'muse', 'use'}:
            result = self.__vectorizer(docs).numpy().tolist()
        elif self.method == 'laser':
            result = self.__vectorizer.embed_sentences(docs, **kwargs).tolist()
        
        return result


def vector_generator(es: Elasticsearch,
                     es_scan: Generator[Dict[str, Any], None, None],
                     vectorizer: Vectorizer, 
                     params_vectorizer: Optional[Dict[str, Any]] = None,
                     index_to: str = 'recent_vectors',
) -> Generator[Dict[str, Any], None, None]:
    
    if params_vectorizer is None:
        params_vectorizer = {}
    
    for doc in es_scan:
        if not es.exists(index=index_to, id=doc['_id']):
            
            try:
                text = doc['_source']['article']['maintext']
                text = text[:min(TEXT_MAX, len(text))]
                vector = vectorizer.vectorize(text, **params_vectorizer)
                yield {
                    'vector': vector,
                    '_id': doc['_id'],
                    'date_published': doc['_source']['article']['date_publish'],
                    '_index': index_to,
                }
            except Exception as e:
                print(e)
                exit()
                

def clean_vector_index(es: Elasticsearch,
                       index_doc: str = 'november2019', 
                       index_vectors: str = 'recent_vectors'):
    print(f'{timestamp()}: Running cleaning of the vector index')
    BODY_RM = {
        "size":100,
        "query":{
            "bool":{
                "filter":[
                    {
                        "terms":{
                            "_id": [item['_id'] for item in 
                                    helpers.scan(es, 
                                         index=index_vectors, 
                                         query={'query': {'match_all': {}}})]
                        }
                    },
                    {
                        "range":{
                            "article.date_publish":{
                                "lt":f"now-{PERIOD}"
                            }
                        }
                    }
                ]
            }
        }
    }

    # result = es.search(index='november2019', body=BODY_RM)
    # n_rm = result['hits']['total']['value']
    idx_to_rm = [doc['_id'] for doc in helpers.scan(es, index=index_doc, query=BODY_RM)]
    print(f'\t-> removal of {len(idx_to_rm)} old vectors')
    for idx in idx_to_rm:
        es.delete(index=index_vectors, id=idx)

def update_vector_index(es: Elasticsearch,
                        vectorizer: Vectorizer,
                        params_vectorizer: Optional[Dict[str, Any]] = None,
                        index_doc: str = 'november2019', 
                        index_vectors: str = 'recent_vectors'):
    print(f'{timestamp()}: running update of the vector index')
    
    BODY_ADD = {
        "size":1000,
        "query":{
            "bool":{
                "must":[
                    {
                        "terms":{
                            "article.language":LANGUAGES
                        }
                    },
                    {
                        "range":{
                            "article.date_publish":{
                                "gte":f"now-{PERIOD}",
                                "lte":"now"
                            }
                        }
                    }
                ]
            }
        }
    }
    es_scan = helpers.scan(es, index=index_doc, query=BODY_ADD)
    result = helpers.bulk(es, 
                          vector_generator(es, 
                                           es_scan=es_scan, 
                                           vectorizer=vectorizer,
                                           params_vectorizer=params_vectorizer,
                                           index_to=index_vectors))
    print(f'\t-> addition of {result[0]} vectors')


def timestamp() -> str:
    return str(datetime.utcnow())[:-7]


def main():
    
    es = Elasticsearch(ES_SERVER)
    vr = Vectorizer(method=MODEL, path_to_model=MODEL_PATH)
    
    while True:
        update_vector_index(es=es, vectorizer=vr)
        clean_vector_index(es)
        
        print(f'{timestamp()}: sleeping {TIME_SLEEP}s')
        time.sleep(5)


if __name__ == '__main__':
    
    ES_SERVER = "10.94.253.5"
    
    # Curently, the list contains only languages supported by MUSE
    LANGUAGES = [
#             "ar",
#             "zh",
#             "zh-tw",
#             "nl",
            "en",
#             "de",
#             "fr",
#             "it",
#             "pt",
#             "es",
#             "ja",
#             "ko",
#             "ru",
#             "pl",
#             "th",
#             "tr"
    ]

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--sleep', default=120, type=int)
    parser.add_argument('--window_type', default='h', type=str)
    parser.add_argument('--window_length', default=24, type=int)
    parser.add_argument('--model', default='muse', type=str)
    parser.add_argument('--model_path', default='../../models/muse/', type=str)
    parser.add_argument('--text_max', default=5000, type=int)

    # Assign arguments
    kwargs = parser.parse_args()
    TIME_SLEEP = kwargs.sleep
    PERIOD = f'{kwargs.window_length}{kwargs.window_type}'
    MODEL = kwargs.model
    MODEL_PATH = kwargs.model_path
    TEXT_MAX = kwargs.text_max

    # Run
    main()

