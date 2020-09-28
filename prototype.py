#  2020 국어 정보 처리 시스템 경진 대회 출품작
#  Team 리프: 영화 리뷰 분석 시스템

from masa.model import ABSAModel
from masa.utils import gen_aspect_mask, create_result_matrix
from crawler.utils import MovieCrawler

import loader
import os
import numpy as np

MOVIE_ASPECT = ["연기", "배우", "스토리", "액션", "감정", "연출", "반전", "음악", "규모"]
SIM_WORD_LIST = [["연기", "연극"],
                 ["배우", "캐스팅", "모델"],
                 ["스토리", "이야기", "시나리오", "콘텐츠", "에피소드", "전개"],
                 ["액션", "전투", "싸움"],
                 ["감정", "감성", "심리"],
                 ["연출", "촬영", "편집"],
                 ["반전", "역전", "전환"],
                 ["음악", "노래", "사운드", "음향"],
                 ["규모", "스케일", "크기"]]

ABSA_model_path = "ABSA_model.pt"
daum_movie_url = "https://movie.daum.net/main/new#slide-1-0"


def _console_clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def _load_fail_msg():
    print("### 모델 로드에 실패했습니다.")
    print(f"### 실행 경로에 \'{ABSA_model_path}\' 모델 파일이 필요합니다.")
    input("")


def corpus_analysis(ctx="cuda:0"):
    # create ABSA model
    model = ABSAModel(ctx=ctx)
    model.load_kobert()
    if not model.load_model(ABSA_model_path):
        _load_fail_msg()
        return False

    print("\n##### Aspect-based Sentiment Analysis")
    print("##### 2020-09-28, Team 리프")

    while True:
        print("### Aspect: [", end="")
        for aspect in MOVIE_ASPECT[:-1]:
            print("{}".format(aspect), end=", ")
        print("{}]".format(MOVIE_ASPECT[-1]))
        corpus = input("### 분석 말뭉치 입력: ")
        if corpus == "":
            break

        # create masked corpus
        masked_corpus_list, masked_corpus_info = gen_aspect_mask([corpus], model.opt, SIM_WORD_LIST)
        result_label = np.zeros((len(MOVIE_ASPECT), 1), dtype=np.int32)
        result = np.zeros((len(MOVIE_ASPECT), 1), dtype=np.float)

        if len(masked_corpus_list) > 0:
            # aspect-base sentiment analysis
            sentence_info = model.tokenize(masked_corpus_list)
            _, result_1, result_2 = model.analyze(sentence_info, sa=False, absa=True)

            result_label_1 = np.argmax(result_1, axis=1)
            result_label_2 = np.argmax(result_2, axis=1)
            result_1 = np.max(result_1, axis=1)
            result_2 = np.max(result_2, axis=1)

            # get result
            for idx, (_, aspect_1, aspect_2) in enumerate(masked_corpus_info):
                if aspect_1 != -1:
                    result[aspect_1] = result_1[idx]
                    result_label[aspect_1] = result_label_1[idx] - 1
                if aspect_2 != -1:
                    result[aspect_2] = result_2[idx]
                    result_label[aspect_2] = result_label_2[idx] - 1

        print("\n### 감성 분석 결과")
        if np.sum(np.abs(result_label)) == 0:
            print("검출된 분석 대상 없음.")
        else:
            for asp_idx, aspect in enumerate(MOVIE_ASPECT):
                if result_label[asp_idx] == 1:
                    print(f"{aspect}: 긍정적 ({'%0.2f' % (result[asp_idx] * 100)}%)")
                elif result_label[asp_idx] == -1:
                    print(f"{aspect}: 부정적 ({'%0.2f' % (result[asp_idx] * 100)}%)")

        input()
        print("\n--------------------------------------")


def daum_review_analysis(ctx="cuda:0"):
    # create movie crawler
    crawler = MovieCrawler()

    # create ABSA model
    model = ABSAModel(ctx=ctx)
    model.load_kobert()
    if not model.load_model(ABSA_model_path):
        _load_fail_msg()
        return False

    # input url
    print("\n##### [2020 국어 정보 처리 시스템 경진 대회 출품작]")
    print("##### Aspect-based Sentiment Analysis 를 이용한 영화 리뷰 분석 시스템")
    print("##### 2020-09-28, Team 리프")

    print("\nDAUM 영화 홈페이지: {}".format(daum_movie_url))
    url = input("영화 메인 URL 입력: ")
    print("영화 리뷰 데이터를 가져오는 중...")
    crawl_data = crawler.crawl(url)
    print("영화 리뷰 데이터 크롤링 성공")

    print("\n### 영화 제목: [ {} ]".format(crawl_data[0]))

    # get corpus list
    corpus_list = crawl_data[1]

    # aspect-based review analysis
    review_matrix = model.analyze_quickly(corpus_list, SIM_WORD_LIST)

    # get probability
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
    print("\n\n### Top 3 Aspect: 영화 리뷰에 가장 많이 발견된 속성에 대하여 감성 분석")
    for i in range(0, 3):
        idx = asp_rank[i]
        if 0 <= ratio[idx] <= 1:
            print(f"### {i + 1}. {MOVIE_ASPECT[idx]}: {'긍정적' if ratio[idx] > 0.5 else '부정적'} "
                  f"({'%0.2f' % (ratio[idx] * 100 if ratio[idx] > 0.5 else (1 - ratio[idx]) * 100)}%)")
        else:
            print(f"### {i + 1}. {MOVIE_ASPECT[idx]}: 없음")

    # Target Review
    print("\n\n### Target Review: 관심 있는 속성에 대한 리뷰 분석 결과를 출력")
    print("### Aspect: [", end="")
    for aspect in MOVIE_ASPECT[:-1]:
        print("{}".format(aspect), end=", ")
    print("{}]".format(MOVIE_ASPECT[-1]))

    aspect_idx = -1
    while aspect_idx == -1:
        keyword = input("### 검색 키워드 입력: ")
        
        for idx, aspect in enumerate(MOVIE_ASPECT):
            if aspect == keyword:
                aspect_idx = idx
                break

    pos_reviews_idx = np.where(review_matrix[:, aspect_idx] == 1)[0]
    neg_reviews_idx = np.where(review_matrix[:, aspect_idx] == -1)[0]
    pos_review_count = min(pos_reviews_idx.shape[0], 5)
    neg_review_count = min(neg_reviews_idx.shape[0], 5)
    np.random.shuffle(pos_reviews_idx)
    np.random.shuffle(neg_reviews_idx)

    if pos_review_count < 1:
        print(f"\n### \"{MOVIE_ASPECT[aspect_idx]}\" 관련 긍정적 리뷰가 존재하지 않습니다.")
    else:
        print(f"\n### \"{MOVIE_ASPECT[aspect_idx]}\" 관련 긍정적 리뷰 {pos_review_count}개 분석 결과 출력")
        for i in range(0, pos_review_count):
            print(f"### Review {i + 1}. ({'긍정적' if review_matrix[pos_reviews_idx[i]][aspect_idx] > 0 else '부정적'})"
                  f" \"{corpus_list[pos_reviews_idx[i]]}\"")

    if neg_review_count < 1:
        print(f"\n### \"{MOVIE_ASPECT[aspect_idx]}\" 관련 부정적 리뷰가 존재하지 않습니다.")
    else:
        print(f"\n### \"{MOVIE_ASPECT[aspect_idx]}\" 관련 부정적 리뷰 {neg_review_count}개 분석 결과 출력")
        for i in range(0, neg_review_count):
            print(f"### Review {i + 1}. ({'긍정적' if review_matrix[neg_reviews_idx[i]][aspect_idx] > 0 else '부정적'})"
                  f" \"{corpus_list[neg_reviews_idx[i]]}\"")

    # Custom Aspect
    print("\n\n### Custom Aspect: 사용자가 직접 지시한 단어에 대한 감성 분석 결과 출력")
    custom_aspect = input("### 감성 분석 주제 입력: ")
    print(f"### \"{custom_aspect}\" 감성 분석 실행...")

    # aspect-based review analysis
    rm = model.analyze_quickly(corpus_list, sim_aspects=[[custom_aspect]])

    # get probability
    tc = np.abs(rm).sum(axis=0)[0]
    pc = rm.copy()
    pc[pc != 1] = 0
    pc = np.count_nonzero(pc, axis=0)[0]
    pr = pc / tc

    print(f"\n### \"{custom_aspect}\" 관련 감성 분석 결과")
    print(f"### 연관 리뷰 개수: {tc}")
    if tc > 0:
        print(f"### 감성 지표: {'긍정적' if pr > 0.5 else '부정적'}"
              f" ({'%0.2f' % (pr * 100 if pr > 0.5 else (1 - pr) * 100)}%)")
    else:
        print("감성 지표를 분석할 수 없습니다.")

    trg = np.where(rm != 0)[0]
    np.random.shuffle(trg)
    out_cnt = min(tc, 5)
    print(f"\n### \"{custom_aspect}\" 관련 리뷰 무작위 {out_cnt}개 분석 결과 출력")
    for i in range(0, out_cnt):
        print(f"### Review {i+1}. ({'긍정적' if rm[trg[i]] > 0 else '부정적'})"
              f" \"{corpus_list[trg[i]]}\"")


def movie_recommendation():
    """
        다음 리뷰 데이터를 이용하여 영화 리뷰 추천
    """
    movie_information, movie_names = loader.load_movie_analysis_data()

    print("\n##### [2020 국어 정보 처리 시스템 경진 대회 출품작]")
    print("##### Aspect-based Sentiment Analysis 를 이용한 영화 추천 시스템")
    print("##### 2020-09-28, Team 리프")

    # Target Selection
    print("\n\n### Movie Recommendation: 관심 있는 속성을 기준으로 영화를 추천")
    print("### - 영화 리뷰 데이터 감성 분석 결과를 바탕으로 영화를 추천합니다.")
    print("### Aspect: [", end="")
    for aspect in MOVIE_ASPECT[:-1]:
        print("{}".format(aspect), end=", ")
    print("{}]".format(MOVIE_ASPECT[-1]))

    asp_idx = -1
    while asp_idx == -1:
        keyword = input("### 검색 키워드 입력: ")

        for idx, aspect in enumerate(MOVIE_ASPECT):
            if aspect == keyword:
                asp_idx = idx
                break

    # 속성 리뷰가 10개 이하인 경우 제외
    asp_cnt_threshold = movie_information[:, 0, asp_idx + 1] > 10
    info = movie_information[asp_cnt_threshold]
    names = movie_names[asp_cnt_threshold]

    print(f"\n### 전체 영화 개수: {movie_names.shape[0]}개")
    print(f"### \"{MOVIE_ASPECT[asp_idx]}\" 관련 영화 개수: {names.shape[0]}개")

    # 영화 추천 알고리즘
    data_cnt = info[:, 0, 0]  # 영화 리뷰 개수
    asp_cnt = info[:, 0, asp_idx + 1]  # 속성 리뷰 개수

    x1 = info[:, 1, 0] / data_cnt  # 영화 리뷰 긍정 비율
    x2 = info[:, 1, asp_idx + 1] / asp_cnt  # 속성 리뷰 긍정 비율
    x3 = asp_cnt / np.sum(info[:, 0, 1:], axis=1)  # 속성 리뷰 개수 비율

    # 가중치 산출
    z1 = 0.4 * np.log(x1) / 6
    z2 = 0.4 * x2
    z3 = 0.2 * x3 * 4.5
    sum_of_z = z1 + z2 + z3

    w1 = z1 / sum_of_z
    w2 = z2 / sum_of_z
    w3 = z3 / sum_of_z

    # 영화 추천 점수 계산 ... 가중 조화 평균
    score = 1 / (w1 / x1 + w2 / x2 + w3 / x3)
    rec_idx = np.argsort(score)[::-1]

    print(f"\n### 영화 추천 결과 [영화 리뷰 긍정 비율 / 관심 속성 긍정 비율]")
    # 추천 영화 산출
    for i in range(0, 10):
        idx = rec_idx[i]
        movie_name = names[idx]
        print(f"[{(i + 1)}]: {movie_name} [{'%0.2f' % (x1[idx] * 100)}% / {'%0.2f' % (x2[idx] * 100)}%]")


def model_validation(ctx):
    """
        validation set 을 이용하여 모델 정확도 계산
    """
    # create ABSA model
    model = ABSAModel(ctx=ctx)
    model.load_kobert()
    if not model.load_model(ABSA_model_path):
        _load_fail_msg()
        return False

    print("\n##### [2020 국어 정보 처리 시스템 경진 대회 출품작]")
    print("##### 2020-09-28, Team 리프")

    print("\n### 평가 데이터 분석중...")

    # load validation data
    corpus_list, asp_info = loader.load_validation_data()

    # split counter example
    counter_example = np.abs(asp_info.sum(axis=1)) != (np.abs(asp_info)).sum(axis=1)
    pos_0 = np.where(counter_example)[0]
    pos_1 = np.where(np.logical_not(counter_example))[0]

    # split corpus list
    corpus_list_0 = []
    corpus_list_1 = []
    for idx in range(0, pos_0.shape[0]):
        corpus_list_0.append(corpus_list[pos_0[idx]])
    for idx in range(0, pos_1.shape[0]):
        corpus_list_1.append(corpus_list[pos_1[idx]])

    # split aspect info
    asp_info_0 = asp_info[pos_0, :]
    asp_info_1 = asp_info[pos_1, :]

    split_package = [[corpus_list_0, asp_info_0], [corpus_list_1, asp_info_1]]
    result = []
    total_count = 0
    hit_count = 0
    for (cl, inf) in split_package:
        # analysis
        rm = model.analyze_quickly(cl, SIM_WORD_LIST)

        # 전체 Aspect 정보 개수
        sub_count = np.count_nonzero(inf)

        # 적중 Aspect 정보 개수
        sub_hit_count = np.count_nonzero(inf[inf == rm])

        # 결과 저장
        result.append([sub_hit_count, sub_count])
        total_count = total_count + sub_count
        hit_count = hit_count + sub_hit_count

    result_0 = hit_count / total_count  # 전체 적중률
    result_1 = result[0][0] / result[0][1]  # 대립 사례 적중률
    result_2 = result[1][0] / result[1][1]  # 일치 사례 적중률

    print("### 평가 데이터 분석 완료\n")
    print(f"### 전체 리뷰 개수: {len(corpus_list)}")
    print(f"### 대립/일치 리뷰 개수: {len(corpus_list_0)}, {len(corpus_list_1)}")
    print(f"### 전체 속성 개수: {total_count}")
    print(f"### 긍정/부정 속성 개수: {np.count_nonzero(asp_info[asp_info==1])}, "
          f"{np.count_nonzero(asp_info[asp_info==-1])}\n")

    print(f"### 전체 정확도: {'%0.2f' % (result_0 * 100)}%")
    print(f"### 대립 사례 정확도: {'%0.2f' % (result_1 * 100)}%")
    print(f"### 일치 사례 정확도: {'%0.2f' % (result_2 * 100)}%")


if __name__ == '__main__':
    # logging.disable(sys.maxsize)
    print("### CUDA GPU 프로세서를 사용합니까?")
    print("[A]: Yes")
    print("[B]: No")
    while True:
        key = input("### Select \'A\' ~ \'B\': ")
        if key == 'A' or key == 'B':
            break
    if key == 'A':
        _ctx = "cuda:0"
    else:
        _ctx = "cpu"

    _console_clear()

    print("\n### 분석 종류를 선택하십시오.")
    print("[A]: Corpus Analysis")
    print("[B]: DAUM Moview Review Analysis")
    print("[C]: Movie Recommendation System")
    print("[D]: Model Validation")

    while True:
        key = input("### Select \'A\' ~ \'D\': ")
        if key == 'A' or key == 'B' or key == 'C' or key =='D':
            break

    _console_clear()
    print("")

    if key == 'A':
        corpus_analysis(ctx=_ctx)
    elif key == 'B':
        daum_review_analysis(ctx=_ctx)
    elif key == 'C':
        movie_recommendation()
    elif key == 'D':
        model_validation(ctx=_ctx)



