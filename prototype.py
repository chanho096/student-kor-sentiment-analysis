#  2020 국어 정보 처리 시스템 경진 대회 출품작
#  Team 리프: 영화 리뷰 분석 시스템

from masa.model import ABSAModel
from masa.utils import gen_aspect_mask, create_result_matrix
from crawler.utils import MovieCrawler

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


def corpus_analysis(ctx="cuda:0"):
    # create ABSA model
    model = ABSAModel(ctx=ctx)
    model.load_kobert()
    model.load_model(ABSA_model_path)

    print("\n##### Aspect-based Sentiment Analysis")
    print("##### 2020-09-14, Team 리프")

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
        print("\n--------------------------------------")


def daum_review_analysis(ctx="cuda:0"):
    # create movie crawler
    crawler = MovieCrawler()

    # create ABSA model
    model = ABSAModel(ctx=ctx)
    model.load_kobert()
    model.load_model(ABSA_model_path)

    # input url
    print("\n##### [2020 국어 정보 처리 시스템 경진 대회 출품작]")
    print("##### Aspect-based Sentiment Analysis 를 이용한 영화 리뷰 분석 시스템")
    print("##### 2020-09-14, Team 리프")

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
    print("\n\n### Top 3 Aspect: 영화 리뷰에 가장 많이 발견된 측면에 대하여 감성 분석")
    for i in range(0, 3):
        idx = asp_rank[i]
        print(f"### {i + 1}. {MOVIE_ASPECT[idx]}: {'긍정적' if ratio[idx] > 0.5 else '부정적'} "
              f"({'%0.2f' % (ratio[idx]*100 if ratio[idx] > 0.5 else (1 - ratio[idx])*100)}%)")

    # Target Review
    print("\n\n### Target Review: 관심 있는 측면에 대한 리뷰 분석 결과를 출력")
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

    target_reviews = np.where(review_matrix[:, aspect_idx] != 0)[0]
    np.random.shuffle(target_reviews)
    review_count = min(total_count[aspect_idx], 5)
    print(f"\n### \"{MOVIE_ASPECT[aspect_idx]}\" 관련 리뷰 무작위 {review_count}개 분석 결과 출력")
    for i in range(0, review_count):
        print(f"### Review {i+1}. ({'긍정적' if review_matrix[target_reviews[i]][aspect_idx] > 0 else '부정적'})"
              f" \"{corpus_list[target_reviews[i]]}\"")

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
    print(f"### 감성 지표: {'긍정적' if pr > 0.5 else '부정적'}"
          f" ({'%0.2f' % (pr*100 if pr > 0.5 else (1 - pr)*100)}%)")

    trg = np.where(rm != 0)[0]
    np.random.shuffle(trg)
    out_cnt = min(tc, 5)
    print(f"\n### \"{custom_aspect}\" 관련 리뷰 무작위 {out_cnt}개 분석 결과 출력")
    for i in range(0, out_cnt):
        print(f"### Review {i+1}. ({'긍정적' if rm[trg[i]] > 0 else '부정적'})"
              f" \"{corpus_list[trg[i]]}\"")


if __name__ == '__main__':
    # logging.disable(sys.maxsize)
    print("### CUDA GPU 프로세서를 사용합니까?")
    print("[A]: Yes")
    print("[B]: No")
    while True:
        key = input("### Press \'A\' or \'B\': ")
        if key == 'A' or key == 'B':
            break
    if key == 'A':
        ctx = "cuda:0"
    else:
        ctx = "cpu"

    print("\n### 분석 종류를 선택하십시오.")
    print("[A]: Corpus Analysis")
    print("[B]: DAUM Moview Review Analysis")
    while True:
        key = input("### Press \'A\' or \'B\': ")
        if key == 'A' or key == 'B':
            break

    print("")
    if key == 'A':
        corpus_analysis(ctx=ctx)
    else:
        daum_review_analysis(ctx=ctx)



