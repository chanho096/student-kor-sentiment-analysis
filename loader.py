import os
import numpy as np
import json
from urllib.request import urlretrieve

corpus_data_dir_name = "corpus"

#  Naver sentiment movie corpus v1.0
#  https://github.com/e9t/nsmc
train_data_url = "https://github.com/e9t/nsmc/raw/master/ratings_train.txt"
test_data_url = "https://github.com/e9t/nsmc/raw/master/ratings_test.txt"
train_data_name = "ratings_train.txt"
test_data_name = "ratings_test.txt"


def download_movie_corpus_data():
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


#  국립 국어원, 모두의 말뭉치 - 구문 분석 말뭉치 
#  의존 관계 분석 모델 학습
dp_corpus_data_name = "NXDP1902008051.json"
dp_max_word_length = 32  # 어절 최대 개수


def load_dependency_parsing_data():
    current_dir = os.getcwd()
    corpus_data_dir = os.path.join(current_dir, corpus_data_dir_name)

    dp_corpus_data_path = os.path.join(corpus_data_dir, dp_corpus_data_name)
    assert(os.path.isfile(dp_corpus_data_path))

    # load json file
    with open(dp_corpus_data_path, encoding='UTF8') as file:
        dp_corpus_data = json.load(file)

    corpus_list = []
    dp_label_list = []
    dp_head_list = []

    dp_data_list = dp_corpus_data['document']
    for dp_data_set in dp_data_list:
        for dp_data in dp_data_set['sentence']:
            corpus = dp_data['form']
            dp = dp_data['DP']

            label = []
            head = []
            for word_info in dp:
                label.append(word_info['label'])
                head.append(word_info['head'])

            if len(label) > dp_max_word_length:
                continue

            corpus_list.append(corpus)
            dp_label_list.append(label)
            dp_head_list.append(head)

    return corpus_list, dp_label_list, dp_head_list


# ---------------------------------------------

