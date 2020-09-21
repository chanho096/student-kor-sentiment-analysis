import numpy as np


def gen_aspect_mask(corpus_list, opt, sim_aspects):
    """
        말뭉치 데이터를 이용하여 ABSA Model 입력 데이터 생성

        ##### parms info #####
        corpus_list: list type - string set
        sim_aspects: double list type - similar word dictionary
    """

    masked_corpus_list = []
    masked_corpus_info = []
    mask = [opt["object_text_0"], opt["object_text_1"]]

    for corpus_idx, corpus in enumerate(corpus_list):
        asp = np.zeros((len(sim_aspects), 1), dtype=np.int32)

        for idx, aspect_list in enumerate(sim_aspects):
            for aspect in aspect_list:
                asp[idx] = corpus.find(aspect)
                if asp[idx] != -1:
                    break
        asp[asp != -1] = 1
        asp[asp == -1] = 0

        rnd_asp = np.where(asp == 1)[0]
        idx = 0

        # 홀수개의 aspect 가 존재하는 경우
        if np.sum(asp) % 2 != 0:
            asp_idx = rnd_asp[idx]
            
            # 유사어 이용 대치
            replaced_corpus = corpus
            for aspect in sim_aspects[asp_idx]:
                replaced_corpus = replaced_corpus.replace(aspect, mask[0])

            masked_corpus_list.append(replaced_corpus)
            masked_corpus_info.append([corpus_idx, asp_idx, -1])
            idx = idx + 1

        # 짝수개의 aspect 를 치환
        while idx < rnd_asp.shape[0]:
            asp_idx_0 = rnd_asp[idx]
            asp_idx_1 = rnd_asp[idx + 1]
            
            # 유사어 이용 대치
            replaced_corpus = corpus
            for aspect in sim_aspects[asp_idx_0]:
                replaced_corpus = replaced_corpus.replace(aspect, mask[0])
            for aspect in sim_aspects[asp_idx_1]:
                replaced_corpus = replaced_corpus.replace(aspect, mask[1])
                
            masked_corpus_list.append(replaced_corpus)
            masked_corpus_info.append([corpus_idx, asp_idx_0, asp_idx_1])
            idx = idx + 2

    return masked_corpus_list, masked_corpus_info


def create_result_matrix(result_1, result_2, masked_corpus_info,
                         corpus_size, aspect_size):
    """
        ABSA 모델 결과를 요약하는 numpy 행렬 생성

        행: corpus
        열: aspect
        입력으로 주어진 corpus_list, 분석에 사용된 aspects 정보는 별도로 가지고있어야 한다.

        ##### parms info #####
        result_1, result_2: result of 'analyze' method in ABSA model
        masked_corpus_info: masked corpus information of input
        corpus_size: size of input
        aspect_size: size of aspects
    """
    result_1 = np.argmax(result_1, axis=1)
    result_2 = np.argmax(result_2, axis=1)

    result_matrix = np.zeros((corpus_size, aspect_size), dtype=np.int32)

    # write result-aspect matrix
    for corpus_idx, (masked_corpus_idx, aspect_1, aspect_2) in enumerate(masked_corpus_info):
        # ABSA Classifier Label: [0:neg, 1:null, 2:pos]
        # Result Matrix: [-1:neg, 0:null, 1:pos]
        if aspect_1 != -1:
            result_matrix[masked_corpus_idx][aspect_1] = result_1[corpus_idx] - 1
        if aspect_2 != -1:
            result_matrix[masked_corpus_idx][aspect_2] = result_2[corpus_idx] - 1

    return result_matrix

