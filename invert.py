#!/usr/bin/env python3
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import sparse


def count(words):
    counts = defaultdict(int)
    for word in words:
        counts[word] += 1
    return counts


def tokenize(doc):
    return [x for x in doc.split(' ') if len(x)]


def invert(doc_count, docid, words, df):
    for word, count in words.items():
        arr = [0] * doc_count # TODO: how to construct sparse series directly
        arr[docid] = min(count, 255)
        posting_list = pd.Series(arr, dtype='Sparse[int64]') # TODO: can this work with int8?
        if word not in df:
            df[word] = posting_list
        else:
            df[word] = df[word] + posting_list


def process(docs):
    df = pd.DataFrame()
    for docid, doc in enumerate(docs):
        words = count(tokenize(doc))
        invert(len(docs), docid, words, df)
    return df


def rank(query, index):
    return [(0,0.95),(1,0.6),(2,0.2)] # TODO: implement TF-IDF or BM25


if __name__ == '__main__':
    docs = ['the fat cat sat on the mat', 'hello fat world cat', 'how long is a piece of string on a mat']
    index = process(docs)
    print(index)
    print(index['fat'][0] == 1)
