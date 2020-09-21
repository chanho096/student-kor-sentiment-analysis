import os
import numpy as np
from urllib.request import urlretrieve

corpus_data_dir_name = "corpus"

#  Naver sentiment movie corpus v1.0
#  https://github.com/e9t/nsmc
train_data_url = "https://github.com/e9t/nsmc/raw/master/ratings_train.txt"
test_data_url = "https://github.com/e9t/nsmc/raw/master/ratings_test.txt"
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
# ---------------------------------------------


#  labeled data for aspect-based sentiment analysis
#  Naver sentiment movie corpus - ratings_test.txt 기반
#  ABSA model validation 에 사용
labeled_corpus_data_name = "labeled_corpus.npy"
labeled_aspect_data_name = "labeled_aspect.npy"


def load_validation_data():
    current_dir = os.getcwd()
    corpus_data_dir = os.path.join(current_dir, corpus_data_dir_name)

    labeled_corpus_data_path = os.path.join(corpus_data_dir, labeled_corpus_data_name)
    labeled_aspect_data_path = os.path.join(corpus_data_dir, labeled_aspect_data_name)
    assert(os.path.isfile(labeled_corpus_data_path))
    assert(os.path.isfile(labeled_aspect_data_path))

    # load npy file
    labeled_corpus_data = np.load(labeled_corpus_data_path)
    labeled_aspect_data = np.load(labeled_aspect_data_path)

    # get corpus list
    corpus_list = labeled_corpus_data.tolist()

    return corpus_list, labeled_aspect_data
# ---------------------------------------------

