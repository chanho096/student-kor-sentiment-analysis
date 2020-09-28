import os
import numpy as np
import json
from urllib.request import urlretrieve

corpus_data_dir_name = "corpus"
analysis_data_dir_name = "analysis"

#  Naver sentiment movie corpus v1.0
#  https://github.com/e9t/nsmc
train_data_url = "https://github.com/e9t/nsmc/raw/master/ratings_train.txt"
test_data_url = "https://github.com/e9t/nsmc/raw/master/ratings_test.txt"
movie_train_data_name = "ratings_train.txt"
movie_test_data_name = "ratings_test.txt"


def get_movie_corpus_data_path():
    current_dir = os.getcwd()
    corpus_data_dir = os.path.join(current_dir, corpus_data_dir_name)

    if not os.path.isdir(corpus_data_dir):
        os.makedirs(corpus_data_dir)

    train_data_path = os.path.join(corpus_data_dir, movie_train_data_name)
    test_data_path = os.path.join(corpus_data_dir, movie_test_data_name)

    # download train/test corpus data
    if not os.path.isfile(train_data_path):
        print(f"Download URL: {train_data_url}")
        urlretrieve(train_data_url, train_data_path)

    if not os.path.isfile(test_data_path):
        print(f"Download URL: {test_data_url}")
        urlretrieve(test_data_url, test_data_path)

    return train_data_path, test_data_path
# ---------------------------------------------


# 한국 경영 학회 - 감성분석을 위한 온라인 상품평 데이터
# http://www.drbr.or.kr/datasets/view/?seq=20
# ABSA 연구 목적 데이터
ABSA_train_data_name = "sentiment_analysis_train.csv"
ABSA_test_data_name = "sentiment_analysis_test.csv"


def get_aspect_based_corpus_data_path():
    current_dir = os.getcwd()
    corpus_data_dir = os.path.join(current_dir, corpus_data_dir_name)

    train_data_path = os.path.join(corpus_data_dir, ABSA_train_data_name)
    test_data_path = os.path.join(corpus_data_dir, ABSA_test_data_name)

    assert(os.path.isfile(train_data_path))
    assert(os.path.isfile(test_data_path))

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


#  다음 영화 (https://movie.daum.net/) 리뷰 데이터에 대한 속성별 감성 분석 결과
#  1138개 영화에 대한 속성 단위 감석 분석 결과
movie_information_data_name = "movie_information.npy"
movie_names_data_name = "movie_names.npy"


def load_movie_analysis_data():
    current_dir = os.getcwd()
    analysis_data_dir = os.path.join(current_dir, analysis_data_dir_name)

    movie_information_data_path = os.path.join(analysis_data_dir, movie_information_data_name)
    movie_names_data_path = os.path.join(analysis_data_dir, movie_names_data_name)
    assert(os.path.isfile(movie_information_data_path))
    assert(os.path.isfile(movie_names_data_path))

    # load numpy file
    movie_information_data = np.load(movie_information_data_path)
    movie_names_data = np.load(movie_names_data_path)

    return movie_information_data, movie_names_data


# ---------------------------------------------

