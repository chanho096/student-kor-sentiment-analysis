#  2020 국어 정보 처리 시스템 경진 대회 출품작
#  Team 리프: 영화 리뷰 분석 시스템

from model import ABSAModel
from crawler.utils import MovieCrawler


def main():
    crawler = MovieCrawler()
    data = crawler.crawl('https://movie.daum.net/moviedb/main?movieId=2')
    print(data)

    model = ABSAModel()
    model.load_kobert()
    model.load_model("example_ABSA_model.pt")
    corpus_list = ["오늘 밥먹었는데 정말 최고였어요. 근데 영화 대상 진짜 안좋더라구요"]
    sentence_info = model.tokenize(corpus_list)
    result_0, result_1, result_2 = model.analyze(sentence_info, sa=True, absa=True)
    print(result_0)
    print(result_1)
    print(result_2)


if __name__ == '__main__':
    main()
    # ex__sentiment_analysis()
    # ex__ABSA_training()
    # ex__ABSA()