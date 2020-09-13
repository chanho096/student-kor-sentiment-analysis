#  Naver sentiment movie corpus v1.0 사용
#  https://github.com/e9t/nsmc

import os
from urllib.request import urlretrieve, urlopen

train_data_url = "https://github.com/e9t/nsmc/raw/master/ratings_train.txt"
test_data_url = "https://github.com/e9t/nsmc/raw/master/ratings_test.txt"
corpus_data_dir_name = "corpus"
train_data_name = "ratings_train.txt"
test_data_name = "ratings_test.txt"


def download_corpus_data():
    current_dir = os.getcwd()
    corpus_data_dir = os.path.join(current_dir, corpus_data_dir_name)

    if not os.path.isdir(corpus_data_dir):
        os.makedirs(corpus_data_dir)

    train_data_path = os.path.join(corpus_data_dir, train_data_name)
    test_data_path = os.path.join(corpus_data_dir, test_data_name)

    # download train/test corpus data
    if not os.path.isfile(train_data_path):
        print(f"Download URL: {train_data_url}\n")
        urlretrieve(train_data_url, train_data_path)

    if not os.path.isfile(test_data_path):
        print(f"Download URL: {test_data_url}\n")
        urlretrieve(test_data_url, test_data_path)

    return train_data_path, test_data_path

