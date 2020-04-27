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

    """
    Encoding/Vectorization of text wrapper for various models.

    @:param method: str, optional (default: 'muse');
        alias of the encoding/vectorization method to use
        - 'use' - Universal Sentence Encoder
            (https://tfhub.dev/google/universal-sentence-encoder/4)
        - 'muse' - Multilingual Universal Sentence Encoder
            (https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3)
        - 'laser' - Language-Agnostic SEntence Representations
            (https://github.com/facebookresearch/LASER)
    @:param path_to_model: str, optional (default: './models/muse/');
        path to models (not needed for LASER; in case of tf-hub models,
        the parameter may either contain a link or the path to a locally saved
        model)

    """

    __valid_methods = ['muse', 'laser', 'use']

    def __init__(self, method: str = 'muse',
                 path_to_model: str = './models/muse/'):

        assert method in self.__valid_methods, \
            f'Expected method aliases: {self.__valid_methods}'

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
        else:
            raise ValueError(f'Method {self.method} is not available')

        return result


def vector_generator(es: Elasticsearch,
                     es_scan: Generator[Dict[str, Any], None, None],
                     vectorizer: Vectorizer,
                     params_vectorizer: Optional[Dict[str, Any]] = None,
                     index_to: str = 'recent_vectors',
                     ) -> Generator[Dict[str, Any], None, None]:
    """
    The function serving as a generator for bulk insertion
    of vectors to the Elasticsearch index.

    :param es: the Elasticsearch instance
    :param es_scan: iterator for the index containing texts for vectorization
    :param vectorizer: vectorizer to use for vectorization
    :param params_vectorizer: parameters for the vectorizer
    :param index_to: name of the Elasticsearch index to write vectors to
    :return: dictionary corresponding to what will be inserted into the index
    """

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
                       index_vectors: str = 'recent_vectors') -> None:
    """
    Removal of such vectors from the index that are older then the specified
    PERIOD (the global variable).

    :param es: the Elasticsearch instance
    :param index_doc: name of the index containing documents
    :param index_vectors: name of the index containing vectors
    :return: None
    """
    print(f'{timestamp()}: Running cleaning of the vector index')
    BODY_RM = {
        "size": 100,
        "query": {
            "bool": {
                "filter": [
                    {
                        "terms": {
                            "_id": [
                                item['_id'] for item in
                                helpers.scan(
                                    es,
                                    index=index_vectors,
                                    query={'query': {'match_all': {}}}
                                )
                            ]
                        }
                    },
                    {
                        "range": {
                            "article.date_publish": {
                                "lt": f"now-{PERIOD}"
                            }
                        }
                    }
                ]
            }
        }
    }

    # result = es.search(index='november2019', body=BODY_RM)
    # n_rm = result['hits']['total']['value']
    idx_to_rm = [doc['_id'] for doc in helpers.scan(es,
                                                    index=index_doc,
                                                    query=BODY_RM)]
    print(f'\t-> removal of {len(idx_to_rm)} old vectors')
    for idx in idx_to_rm:
        es.delete(index=index_vectors, id=idx)


def update_vector_index(es: Elasticsearch,
                        vectorizer: Vectorizer,
                        params_vectorizer: Optional[Dict[str, Any]] = None,
                        index_doc: str = 'november2019',
                        index_vectors: str = 'recent_vectors') -> None:
    """
    Function responsible for vector insertion into the Elasticsearch index.
    Firstly, it queries for articles. Then these articles are vectorized and
    the respective vectors are saved within the Elasticsearch instance.

    :param es: the Elasticsearch instance
    :param vectorizer: vectorizer to use for vectorization
    :param params_vectorizer: parameters for the vectorizer
    :param index_doc: name of the Elasticsearch index containing documents
    :param index_vectors: name of the Elasticsearch index to write vectors to
    :return: None
    """
    print(f'{timestamp()}: running update of the vector index')

    BODY_ADD = {
        "size": 1000,
        "query": {
            "bool": {
                "must": [
                    {
                        "terms": {
                            "article.language": LANGUAGES
                        }
                    },
                    {
                        "range":{
                            "article.date_publish": {
                                "gte": f"now-{PERIOD}",
                                "lte": "now"
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


def main() -> None:

    es = Elasticsearch(ES_SERVER)
    vr = Vectorizer(method=MODEL, path_to_model=MODEL_PATH)

    while True:
        update_vector_index(es=es, vectorizer=vr)
        clean_vector_index(es)

        print(f'{timestamp()}: sleeping {TIME_SLEEP}s')
        time.sleep(TIME_SLEEP)


if __name__ == '__main__':

    ES_SERVER = "10.94.253.5"

    # Currently, the list contains only languages supported by MUSE
    LANGUAGES = [
        # "ar",
        # "zh",
        # "zh-tw",
        # "nl",
        "en",
        # "de",
        # "fr",
        # "it",
        # "pt",
        # "es",
        # "ja",
        # "ko",
        # "ru",
        # "pl",
        # "th",
        # "tr"
    ]

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--sleep', default=120, type=int,
                        help='time in seconds to wait for the next iteration')
    parser.add_argument('--window_type', default='h', type=str,
                        help='units (according to the Elasticsearch DSL) '
                             'to measure time from now to the marginal '
                             'value (e.g. "d" - days, "h" - hours)')
    parser.add_argument('--window_length', default=24, type=int,
                        help='length of the time window in specified units')
    parser.add_argument('--model', default='muse', type=str,
                        help='name of the vectorization model to use')
    parser.add_argument('--model_path', default='../../models/muse/', type=str,
                        help='path to the vectorization model '
                             '(for more information, read the documentation '
                             'for the "Vectorizer" class)')
    parser.add_argument('--text_max', default=5000, type=int,
                        help='maximal number of characters '
                             '(starting from the beginning of the article) '
                             'to account for')

    # Assign arguments
    kwargs = parser.parse_args()
    TIME_SLEEP = kwargs.sleep
    PERIOD = f'{kwargs.window_length}{kwargs.window_type}'
    MODEL = kwargs.model
    MODEL_PATH = kwargs.model_path
    TEXT_MAX = kwargs.text_max

    # Run
    main()
