# spaceOut
SpaceOut for DCE.

## ProcessorIndexer.py

This Python script monitors a folder and processes files as they come in, through a spaCy pipeline. Output files are indexed into Elasticsearch.

## ProcessorVectorizer.py

This script runs a query on previously indexed documents and identifies the ones which have to go through vectorization. Vectors are stored in a separate index of Elasticsearch for a limited amount of time.

For more information on hyperparameters, run

```console
python ProcessorVectorizer.py --help
```
