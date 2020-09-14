import model as md


def main():
    model = md.ABSAModel()
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