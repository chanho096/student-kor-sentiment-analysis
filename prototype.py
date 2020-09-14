#  2020 국어 정보 처리 시스템 경진 대회 출품작
#  Team 리프: 영화 리뷰 분석 시스템

from model import ABSAModel
from crawler.utils import MovieCrawler

import logging
import numpy as np
import os
import sys

MOVIE_ASPECT = ["연기", "배우", "스토리", "액션", "감동", "연출", "반전", "사운드", "스케일"]

ABSA_model_path = "ABSA_model.pt"
daum_movie_url = "https://movie.daum.net/main/new#slide-1-0"
demo_url = "https://movie.daum.net/moviedb/main?movieId=2"


def _aspect_mask_to_corpus(corpus_list, opt):
    """
        말뭉치 데이터를 이용하여 ABSA 데이터 생성
    """
    masked_corpus_list = []
    masked_corpus_info = []
    mask = [opt["object_text_0"], opt["object_text_1"]]

    for corpus_idx, corpus in enumerate(corpus_list):
        asp = np.zeros((len(MOVIE_ASPECT), 1), dtype=np.int32)
        for idx, aspect in enumerate(MOVIE_ASPECT):
            asp[idx] = corpus.find(aspect)
        asp[asp != -1] = 1
        asp[asp == -1] = 0

        rnd_asp = np.where(asp == 1)[0]
        idx = 0

        # 홀수개의 aspect 가 존재하는 경우
        if np.sum(asp) % 2 != 0:
            asp_idx = rnd_asp[idx]
            masked_corpus_list.append(corpus.replace(MOVIE_ASPECT[asp_idx], mask[0]))
            masked_corpus_info.append([corpus_idx, asp_idx, -1])
            idx = idx + 1

        # 짝수개의 aspect 를 치환
        while idx < rnd_asp.shape[0]:
            asp_idx_0 = rnd_asp[idx]
            asp_idx_1 = rnd_asp[idx + 1]
            replaced_corpus = corpus.replace(MOVIE_ASPECT[asp_idx_0], mask[0])
            replaced_corpus = replaced_corpus.replace(MOVIE_ASPECT[asp_idx_1], mask[1])
            masked_corpus_list.append(replaced_corpus)
            masked_corpus_info.append([corpus_idx, asp_idx_0, asp_idx_1])
            idx = idx + 2

    return masked_corpus_list, masked_corpus_info


def daum_review_analyze():
    # create movie crawler
    crawler = MovieCrawler()

    # create ABSA model
    model = ABSAModel()
    model.load_kobert()
    model.load_model(ABSA_model_path)

    # input url
    print("\n##### [2020 국어 정보 처리 시스템 경진 대회 출품작]")
    print("##### Aspect-based Sentiment Analysis 를 이용한 영화 리뷰 분석 시스템\n")
    print("DAUM 영화 홈페이지: {}".format(daum_movie_url))

    url = input("영화 메인 URL 입력: ")
    crawl_data = crawler.crawl(url)
    print("영화 리뷰 데이터 크롤링 성공\n")

    print("### 영화 제목: [ {} ]".format(crawl_data[0]))

    # get corpus list
    corpus_list = crawl_data[1]

    # create masked corpus_list
    masked_corpus_list, masked_corpus_info = _aspect_mask_to_corpus(corpus_list, model.opt)

    # create review-aspect matrix
    review_matrix = np.zeros((len(corpus_list), len(MOVIE_ASPECT)), dtype=np.int32)

    # aspect-base sentiment analysis
    sentence_info = model.tokenize(masked_corpus_list)
    _, result_1, result_2 = model.analyze(sentence_info, sa=False, absa=True)
    result_1 = np.argmax(result_1, axis=1)
    result_2 = np.argmax(result_2, axis=1)

    # write review-aspect matrix
    for idx, (review_idx, aspect_1, aspect_2) in enumerate(masked_corpus_info):
        # ABSA Classifier Label: [0:neg, 1:null, 2:pos]
        # Review-Aspect Matrix: [-1:neg, 0:null, 1:pos]
        review_matrix[review_idx][aspect_1] = result_1[idx] - 1
        review_matrix[review_idx][aspect_2] = result_2[idx] - 1

    # aspect-base review analysis
    total_count = np.abs(review_matrix).sum(axis=0)
    pos_count = review_matrix.copy()
    pos_count[pos_count != 1] = 0
    pos_count = np.count_nonzero(pos_count, axis=0)
    ratio = pos_count / total_count

    # review information
    print("### 총 리뷰 개수: {}".format(len(corpus_list)))
    print("### 감성 분석 리뷰 개수: {}".format(np.sum(total_count)))

    # Top 3 aspect information
    asp_rank = np.argsort(total_count)[::-1]
    print("")
    print("### Top 3 Aspect: 영화 리뷰에 가장 많이 발견된 측면에 대하여 감성 분석")
    for i in range(0, 3):
        idx = asp_rank[i]
        print(f"### {i + 1}. {MOVIE_ASPECT[idx]}: {'긍정적' if ratio[idx] > 0.5 else '부정적'} "
              f"({'%0.2f' % (ratio[idx]*100 if ratio[idx] > 0.5 else (1 - ratio[idx])*100)}%)")


if __name__ == '__main__':
    logging.disable(sys.maxsize)
    daum_review_analyze()


