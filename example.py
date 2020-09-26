import loader
from masa.model import ABSAModel
import masa.model

import gluonnlp as nlp
import torch
import transformers
import numpy as np
import random

ABSA_model_path = "ABSA_model.pt"
MOVIE_ASPECT = ["연기", "배우", "스토리", "액션", "감정", "연출", "반전", "음악", "규모"]
EX_SIM_WORD_LIST = [["연기", "연극"],
                    ["배우", "캐스팅", "모델"],
                    ["스토리", "이야기", "시나리오", "콘텐츠", "에피소드", "전개"],
                    ["액션", "전투", "싸움"],
                    ["감정", "감성", "심리"],
                    ["연출", "촬영", "편집"],
                    ["반전", "역전", "전환"],
                    ["음악", "노래", "사운드", "음향"],
                    ["규모", "스케일", "크기"]]


class BaseModel(torch.nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,  # bert hidden size
                 dr_rate_0=None
                 ):
        super(BaseModel, self).__init__()
        self.bert = bert
        self.dr_rate_0 = dr_rate_0

        self.classifier_0 = torch.nn.Linear(hidden_size, 3)

        if dr_rate_0:
            self.dropout_0 = torch.nn.Dropout(p=dr_rate_0)

    def forward(self, x, segment_ids, attention_mask):
        # bert forward
        _, pooler = self.bert(inputs_embeds=x, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask)

        out_0 = self.dropout_0(pooler) if self.dr_rate_0 else pooler
        out_0 = self.classifier_0(out_0)
        out_0 = torch.nn.functional.softmax(out_0, dim=1)

        return out_0


def _default_train_setup(opt, model, batch_loader):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=opt["learning_rate"])
    loss_function = torch.nn.CrossEntropyLoss()

    t_total = len(batch_loader) * opt["num_epochs"]
    warmup_steps = int(t_total * opt["warmup_ratio"])
    scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer, warmup_steps, t_total)

    return loss_function, optimizer, scheduler,


def _model_validation(ABSA_model):
    """
        validation set 을 이용하여 모델 정확도 계산
    """
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
        rm = ABSA_model.analyze_quickly(cl, EX_SIM_WORD_LIST)

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

    return result_0, result_1, result_2


def _model_validation_for_base(opt, base_model, device, tokenizer):
    """
        validation set 을 이용하여 모델 정확도 계산
    """
    corpus_list, asp_info = loader.load_validation_data()
    transform = nlp.data.BERTSentenceTransform(
        tokenizer, max_seq_length=opt["bert_max_len"], pad=True, pair=True)

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
        # validation input 재생성
        added_cl = []
        result_info = []

        for idx, corpus in enumerate(cl):
            for asp_idx in range(0, inf[idx].shape[0]):
                if inf[idx][asp_idx] == 0:
                    continue

                # find aspect wor
                word_list = EX_SIM_WORD_LIST[asp_idx]
                aspect_word = None
                for word in word_list:
                    if corpus.find(word) != -1:
                        aspect_word = word
                if aspect_word is None:
                    continue

                added_cl.append((corpus, aspect_word))
                result_info.append((asp_idx, inf[idx][asp_idx]))

        # tokenizing
        sentence_info = []
        for idx, added_corpus in enumerate(added_cl):
            sentence = transform(added_corpus)
            sentence_info.append(sentence + result_info[idx])

        # batch loader
        batch_loader = torch.utils.data.DataLoader(sentence_info, batch_size=opt["batch_size"],
                                                   num_workers=0, shuffle=False)

        # analyze
        sub_count = 0
        sub_hit_count = 0

        with torch.no_grad():
            for batch_id, (token_ids, valid_length, segment_ids, _, label) in enumerate(batch_loader):
                # set test batch
                token_ids = token_ids.long().to(device)
                segment_ids = segment_ids.long().to(device)
                valid_length = valid_length

                # get word embedding
                attention_mask = masa.model.gen_attention_mask(token_ids, valid_length)
                word_embedding = base_model.bert.get_input_embeddings()
                x = word_embedding(token_ids)

                # forward propagation
                out_0 = base_model(x, segment_ids, attention_mask)

                # compare
                result_0 = out_0.cpu().numpy()
                result_0 = np.argmax(result_0, axis=1) - 1

                sub_count += result_0.shape[0]
                sub_hit_count += np.count_nonzero(result_0[result_0 == label.numpy()])

        # 결과 저장
        result.append([sub_hit_count, sub_count])
        total_count = total_count + sub_count
        hit_count = hit_count + sub_hit_count

    result_0 = hit_count / total_count  # 전체 적중률
    result_1 = result[0][0] / result[0][1]  # 대립 사례 적중률
    result_2 = result[1][0] / result[1][1]  # 일치 사례 적중률

    return result_0, result_1, result_2


def _load_with_augmentation(dataset, opt=masa.model.DEFAULT_OPTION):
    object_text_0 = opt["object_text_0"]
    object_text_1 = opt["object_text_1"]

    target_dataset = []
    for data in dataset:
        # 길이 64 초과의 문자열 데이터를 제거한다.
        if len(data[0]) > 64:
            continue

        # corpus 에서 object text를 제거한다.
        data[0] = data[0].replace(object_text_0, "", 3)
        data[0] = data[0].replace(object_text_1, "", 3)

        # 수정된 dataset 생성
        target_dataset.append(data)

    dataset = target_dataset

    # --------------- DATA AUGMENTATION FOR ABSA ---------------
    # ----------------------------------------------------------
    # get dataset with data augmentation
    data_list = []
    pos_dataset = []
    neg_dataset = []
    for data in dataset:
        if data[2] == 'positive':
            pos_dataset.append(data)
        else:
            neg_dataset.append(data)

    num_counter_case = len(dataset) * 2

    # random value
    rnd_0 = np.random.uniform(0, 1, len(dataset)) > 0.5
    rnd_1 = np.random.randint(0, len(dataset) - 1, len(dataset))
    rnd_2 = np.random.uniform(0, 1, len(dataset)) > 0.5
    rnd_3 = np.random.uniform(0, 1, num_counter_case) > 0.5
    rnd_4 = np.random.randint(0, 256, num_counter_case)
    rnd_5 = np.random.uniform(0, 1, len(dataset)) > 0.5
    rnd_6 = np.random.randint(0, 256, len(dataset))
    rnd_7 = np.random.randint(0, 256, len(dataset)) > 0.5
    rnd_8 = np.random.randint(0, len(pos_dataset), num_counter_case)
    rnd_9 = np.random.randint(0, len(neg_dataset), num_counter_case)
    rnd_10 = np.random.uniform(0, 1, (num_counter_case, 3)) > 0.5

    # split by list
    for idx, (corpus, aspect, label) in enumerate(list(dataset)):
        # Augmented Data - single
        # single case - 중립 데이터 생성
        # data_list.append([corpus, [1, 1]])
        
        # single case - 대립 데이터 생성
        label_number = 2 if label == "positive" else 0
        if rnd_0[idx]:
            aug_corpus = corpus.replace(aspect, object_text_0, 3)
            aug_label = [label_number, 1]
        else:
            aug_corpus = corpus.replace(aspect, object_text_1, 3)
            aug_label = [1, label_number]
        data_list.append([aug_corpus, aug_label])

        # single case - 일치 데이터 생성
        if rnd_5[idx]:
            if rnd_6[idx] % 4 == 0:
                aug_corpus = corpus.replace(aspect, object_text_0 + ", " + object_text_1, 3)
            elif rnd_6[idx] % 4 == 1:
                aug_corpus = corpus.replace(aspect, object_text_1 + ", " + object_text_0, 3)
            elif rnd_6[idx] % 4 == 2:
                aug_corpus = corpus.replace(aspect, object_text_0 + " " + object_text_1, 3)
            else:
                aug_corpus = corpus.replace(aspect, object_text_1 + ", " + object_text_0, 3)

            aug_label = [label_number, label_number]
            data_list.append([aug_corpus, aug_label])

        # Augmented Data - pair
        rnd_1[idx] = len(dataset) - 1 if rnd_1[idx] == idx else rnd_1[idx]
        corpus_1, aspect_1, label_1 = dataset[rnd_1[idx]]
        label_number_1 = 2 if label_1 == "positive" else 0

        # pair case - 완전 중립 데이터 생성
        # data_list.append([corpus + " " + corpus_1, [1, 1]])

        # pair case - 부분 중립 데이터 생성
        if rnd_7[idx] % 4 == 0:
            aug_corpus = corpus.replace(aspect, object_text_0, 3) + " " + corpus_1
            aug_label = [label_number, 1]
        elif rnd_7[idx] % 4 == 1:
            aug_corpus = corpus + " " + corpus_1.replace(aspect_1, object_text_0, 3)
            aug_label = [label_number_1, 1]
        elif rnd_7[idx] % 4 == 2:
            aug_corpus = corpus.replace(aspect, object_text_1, 3) + " " + corpus_1
            aug_label = [1, label_number]
        else:
            aug_corpus = corpus + " " + corpus_1.replace(aspect_1, object_text_1, 3)
            aug_label = [1, label_number_1]
        # data_list.append([aug_corpus, aug_label])

        # pair case - 대립 데이터 생성
        if rnd_2[idx]:
            aug_text_0 = object_text_0
            aug_text_1 = object_text_1
            aug_label = [label_number, label_number_1]
        else:
            aug_text_0 = object_text_1
            aug_text_1 = object_text_0
            aug_label = [label_number_1, label_number]

        aug_corpus = corpus.replace(aspect, aug_text_0, 3) + " " + corpus_1.replace(aspect_1, aug_text_1, 3)
        data_list.append([aug_corpus, aug_label])

    # Augmented Data - counter pair
    for idx in range(0, num_counter_case):
        pos_corpus, pos_aspect, _ = pos_dataset[rnd_8[idx]]
        neg_corpus, neg_aspect, _ = neg_dataset[rnd_9[idx]]

        pos_label_number = 2
        neg_label_number = 0

        # counter pair case - 대립 데이터 생성
        if rnd_4[idx] % 4 == 0:
            left_corpus = pos_corpus.replace(pos_aspect, object_text_0, 3)
            right_corpus = neg_corpus.replace(neg_aspect, object_text_1, 3)
            aug_label = [pos_label_number, neg_label_number]
        elif rnd_4[idx] % 4 == 1:
            left_corpus = neg_corpus.replace(neg_aspect, object_text_0, 3)
            right_corpus = pos_corpus.replace(pos_aspect, object_text_1, 3)
            aug_label = [neg_label_number, pos_label_number]
        elif rnd_4[idx] % 4 == 2:
            left_corpus = neg_corpus.replace(neg_aspect, object_text_1, 3)
            right_corpus = pos_corpus.replace(pos_aspect, object_text_0, 3)
            aug_label = [pos_label_number, neg_label_number]
        else:
            left_corpus = pos_corpus.replace(pos_aspect, object_text_1, 3)
            right_corpus = neg_corpus.replace(neg_aspect, object_text_0, 3)
            aug_label = [neg_label_number, pos_label_number]

        aug_corpus = left_corpus + " " + right_corpus
        data_list.append([aug_corpus, aug_label])
        
        # counter pair case - 완전 중립 데이터 생성
        '''
        if rnd_3[idx]:
            data_list.append([pos_corpus + " " + neg_corpus, [1, 1]])
        else:
            data_list.append([neg_corpus + " " + pos_corpus, [1, 1]])
        '''

        # counter pair case - 부분 중립 데이터 생성
        # random case 1. 마스크 단어 선택
        if rnd_10[idx, 0]:
            aug_text = object_text_0
            aug_label_idx = 0
        else:
            aug_text = object_text_1
            aug_label_idx = 1

        # random case 2. 긍정/부정 문장 중 중립값 선택
        if rnd_10[idx, 1]:
            null_corpus = pos_corpus
            target_corpus = neg_corpus.replace(neg_aspect, aug_text)
            target_label = 0
        else:
            null_corpus = neg_corpus
            target_corpus = pos_corpus.replace(pos_aspect, aug_text)
            target_label = 2
        
        # random case 3. 중립 문장의 왼쪽/오른쪽 결합 선택
        if rnd_10[idx, 2]:
            aug_corpus = null_corpus + " " + target_corpus
        else:
            aug_corpus = target_corpus + " " + null_corpus
        aug_label = [1, 1]
        aug_label[aug_label_idx] = target_label

        # data_list.append([aug_corpus, aug_label])

    # Augmented Data - counter triple
    short_pos_dataset = []
    short_neg_dataset = []
    short_dataset = []
    for data in dataset:
        # 짧은 문장 데이터만을 추출한다.
        if len(data[0]) > 32:
            continue

        if data[2] == 'positive':
            short_pos_dataset.append(data)
        else:
            short_neg_dataset.append(data)
        short_dataset.append(data)

    num_triple_counter_case = len(short_dataset) * 3
    rnd_pos = np.random.randint(0, len(short_pos_dataset), num_triple_counter_case)
    rnd_neg = np.random.randint(0, len(short_neg_dataset), num_triple_counter_case)
    rnd_null = np.random.randint(0, len(short_dataset), num_triple_counter_case)
    rnd_case_0 = np.random.uniform(0, 1, num_triple_counter_case) > 0.5
    rnd_case_1 = np.random.randint(0, 768, num_triple_counter_case)
    rnd_case_2 = np.random.randint(0, 768, num_triple_counter_case)

    for idx in range(0, num_triple_counter_case):
        if rnd_case_0[idx]:
            neg_text = object_text_0
            pos_text = object_text_1
            aug_label = [0, 2]
        else:
            neg_text = object_text_1
            pos_text = object_text_0
            aug_label = [2, 0]

        pos_corpus, pos_aspect, _ = short_pos_dataset[rnd_pos[idx]]
        neg_corpus, neg_aspect, _ = short_neg_dataset[rnd_neg[idx]]
        null_corpus, _, _ = short_dataset[rnd_null[idx]]
        
        # counter triple case - 완전 중립 데이터 생성
        if rnd_case_1[idx] % 6 == 0:
            aug_corpus = pos_corpus + " " + null_corpus + " " + neg_corpus
        elif rnd_case_1[idx] % 6 == 1:
            aug_corpus = neg_corpus + " " + null_corpus + " " + pos_corpus
        elif rnd_case_1[idx] % 6 == 2:
            aug_corpus = null_corpus + " " + pos_corpus + " " + neg_corpus
        elif rnd_case_1[idx] % 6 == 3:
            aug_corpus = null_corpus + " " + neg_corpus + " " + pos_corpus
        elif rnd_case_1[idx] % 6 == 4:
            aug_corpus = pos_corpus + " " + neg_corpus + " " + null_corpus
        else:
            aug_corpus = neg_corpus + " " + pos_corpus + " " + null_corpus

        # data_list.append([aug_corpus, [1, 1]])

        # counter triple case - 대립 데이터 생성
        pos_corpus = pos_corpus.replace(pos_aspect, pos_text, 3)
        neg_corpus = neg_corpus.replace(neg_aspect, neg_text, 3)

        if rnd_case_2[idx] % 6 == 0:
            aug_corpus = pos_corpus + " " + null_corpus + " " + neg_corpus
        elif rnd_case_2[idx] % 6 == 1:
            aug_corpus = neg_corpus + " " + null_corpus + " " + pos_corpus
        elif rnd_case_2[idx] % 6 == 2:
            aug_corpus = null_corpus + " " + pos_corpus + " " + neg_corpus
        elif rnd_case_2[idx] % 6 == 3:
            aug_corpus = null_corpus + " " + neg_corpus + " " + pos_corpus
        elif rnd_case_2[idx] % 6 == 4:
            aug_corpus = pos_corpus + " " + neg_corpus + " " + null_corpus
        else:
            aug_corpus = neg_corpus + " " + pos_corpus + " " + null_corpus

        data_list.append([aug_corpus, aug_label])

    # random shuffle
    random.shuffle(data_list)

    # get corpus, label list
    corpus_list = []
    label_list = []
    for (corpus, label) in data_list:
        corpus_list.append(corpus)
        label_list.append(np.array(label, dtype=np.int32))

    return corpus_list, label_list


def ex_pre_training(opt=masa.model.DEFAULT_OPTION, ctx="cuda:0"):
    ABSA_model = ABSAModel(ctx=ctx, opt=opt)
    ABSA_model.load_kobert()
    ABSA_model.load_model(model_path=None, dr_rate_0=opt["drop_out_rate"])

    # ---------------------- Data Loader -----------------------
    # ----------------------------------------------------------
    # Naver sentiment movie corpus v1.0
    data_path = loader.get_movie_corpus_data_path()

    batch_loaders = []
    for path in data_path:
        dataset = nlp.data.TSVDataset(path, field_indices=[1, 2], num_discard_samples=1)

        corpus_list = []
        label_list = []
        for (corpus, label) in list(dataset):
            corpus_list.append(corpus)

            # set label array
            label_array = np.array(label, dtype=np.int32)

            # set label list
            label_list.append(label_array)

        sentence_info = ABSA_model.tokenize(corpus_list)
        for idx, tuple_item in enumerate(sentence_info):
            sentence_info[idx] = tuple_item + (label_list[idx],)

        batch_loader = torch.utils.data.DataLoader(sentence_info, batch_size=opt["batch_size"],
                                                   num_workers=0, shuffle=True)
        batch_loaders.append(batch_loader)

    train_batch_loader, test_batch_loader = batch_loaders

    # ----------------- ABSA CLASSIFIER MODEL ------------------
    # ----------------------------------------------------------
    model = ABSA_model.model
    device = ABSA_model.device

    # default setup
    loss_function, optimizer, scheduler = _default_train_setup(opt, model, train_batch_loader)

    # -------------- ABSA CLASSIFIER MODEL TRAIN ---------------
    # ----------------------------------------------------------
    for e in range(opt["num_epochs"]):
        train_accuracy = 0.0
        test_accuracy = 0.0

        # Train Batch
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(train_batch_loader):
            optimizer.zero_grad()

            # set train batch
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)

            # get word embedding
            attention_mask = masa.model.gen_attention_mask(token_ids, valid_length)
            word_embedding = model.bert.get_input_embeddings()
            x = word_embedding(token_ids)

            # forward propagation
            out_0, _, _ = model(x, segment_ids, attention_mask, sa=True, absa=False)

            # backward propagation
            loss = loss_function(out_0, label)
            loss.backward()

            # optimization
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt["max_grad_norm"])
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_accuracy += masa.model.calculate_accuracy(out_0, label)

            if batch_id % opt["log_interval"] == 0:
                print("epoch {} batch id {} loss {} train accuracy {}".format(e + 1, batch_id + 1,
                                                                              loss.data.cpu().numpy(),
                                                                              train_accuracy / (batch_id + 1)))
        print("epoch {} train accuracy {}".format(e + 1, train_accuracy / (batch_id + 1)))

        # Test Batch
        model.eval()
        with torch.no_grad():
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_batch_loader):
                # set test batch
                token_ids = token_ids.long().to(device)
                segment_ids = segment_ids.long().to(device)
                valid_length = valid_length
                label = label.long().to(device)

                # get word embedding
                attention_mask = masa.model.gen_attention_mask(token_ids, valid_length)
                word_embedding = model.bert.get_input_embeddings()
                x = word_embedding(token_ids)

                # forward propagation
                out_0, _, _ = model(x, segment_ids, attention_mask, sa=True, absa=False)

                # test accuracy
                test_accuracy += masa.model.calculate_accuracy(out_0, label)
            print("epoch {} test accuracy {}".format(e + 1, test_accuracy / (batch_id + 1)))

        torch.save(model.state_dict(), f"pre_trained_model_{e+1}.pt")


def ex_base_model_training(opt=masa.model.DEFAULT_OPTION, ctx="cuda:0"):
    ABSA_model = ABSAModel(ctx=ctx)
    ABSA_model.load_kobert()
    ABSA_model.load_model(model_path=ABSA_model_path)

    # ---------------------- LOAD DATASET ----------------------
    # ----------------------------------------------------------
    # load dataset
    data_path = loader.get_aspect_based_corpus_data_path()

    # set batch loader
    batch_loaders = []
    transform = nlp.data.BERTSentenceTransform(
        ABSA_model.bert_tokenizer, max_seq_length=opt["bert_max_len"], pad=True, pair=True)

    for path in data_path:
        dataset = nlp.data.TSVDataset(path, field_indices=[0, 1, 2], num_discard_samples=1)

        sentence_info = []
        for (corpus, aspect, label) in list(dataset):
            sentence = transform((corpus, aspect))

            # set label array
            label = 2 if label == 'positive' else 0
            label_array = np.array(label, dtype=np.int32)

            # get sentence information (model input)
            sentence_info.append(sentence + (label_array, ))

        batch_loader = torch.utils.data.DataLoader(sentence_info, batch_size=opt["batch_size"],
                                                   num_workers=0, shuffle=True)
        batch_loaders.append(batch_loader)

    train_batch_loader, test_batch_loader = batch_loaders

    # ----------------- ABSA CLASSIFIER MODEL ------------------
    # ----------------------------------------------------------
    # aspect-based sentiment analysis model
    device = ABSA_model.device
    model = BaseModel(ABSA_model.bert_model).to(device)

    # default setup
    loss_function, optimizer, scheduler = _default_train_setup(opt, model, train_batch_loader)

    # -------------- ABSA CLASSIFIER MODEL TRAIN ---------------
    # ----------------------------------------------------------
    for e in range(opt["num_epochs"]):
        train_accuracy = 0.0
        test_accuracy = 0.0

        # Train Batch
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(train_batch_loader):
            optimizer.zero_grad()

            # set train batch
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)

            # get word embedding
            attention_mask = masa.model.gen_attention_mask(token_ids, valid_length)
            word_embedding = model.bert.get_input_embeddings()
            x = word_embedding(token_ids)

            # forward propagation
            out_0 = model(x, segment_ids, attention_mask)

            # backward propagation
            loss = loss_function(out_0, label)
            loss.backward()

            # optimization
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt["max_grad_norm"])
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_accuracy += masa.model.calculate_accuracy(out_0, label)

            if batch_id % opt["log_interval"] == 0:
                print("epoch {} batch id {} loss {} train accuracy {}".format(e + 1, batch_id + 1,
                                                                              loss.data.cpu().numpy(),
                                                                              train_accuracy / (batch_id + 1)))
        print("epoch {} train accuracy {}".format(e + 1, train_accuracy / (batch_id + 1)))

        # Test Batch
        model.eval()
        with torch.no_grad():
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_batch_loader):
                # set test batch
                token_ids = token_ids.long().to(device)
                segment_ids = segment_ids.long().to(device)
                valid_length = valid_length
                label = label.long().to(device)

                # get word embedding
                attention_mask = masa.model.gen_attention_mask(token_ids, valid_length)
                word_embedding = model.bert.get_input_embeddings()
                x = word_embedding(token_ids)

                # forward propagation
                out_0 = model(x, segment_ids, attention_mask)

                # test accuracy
                test_accuracy += masa.model.calculate_accuracy(out_0, label)
            print("epoch {} test accuracy {}".format(e + 1, test_accuracy / (batch_id + 1)))

        # Validation
        r1, r2, r3 = _model_validation_for_base(opt, model, device, ABSA_model.bert_tokenizer)
        print(f"movie corpus test: total accuracy: {'%0.2f' % (r1 * 100)}%, "
              f"case_0 accuracy: {'%0.2f' % (r2 * 100)}%, "
              f"case_1 accuracy: {'%0.2f' % (r3 * 100)}%")

        torch.save(model.state_dict(), f"pre_trained_model_{e + 1}.pt")


def ex_cosine_similarity(model_path=ABSA_model_path, ctx="cuda:0"):
    model = masa.model.ABSAModel(ctx)
    model.load_kobert()
    model.load_model(model_path)

    sentence_info = model.tokenize(MOVIE_ASPECT)
    x = model.word_embedding(sentence_info)
    asp = x[:, 1, :]  # LEN * H
    asp_norm = np.linalg.norm(asp, ord=None, axis=1).reshape((1, -1))

    word_list = [char for char in model.vocab.idx_to_token]
    dataloader = torch.utils.data.DataLoader(word_list, batch_size=128, num_workers=0)
    cosine_sim = np.zeros((len(word_list), len(MOVIE_ASPECT)), dtype=np.float)
    idx = 0

    with torch.no_grad():
        for batch_id, words in enumerate(dataloader):
            batch_count = len(words)

            # set test batch
            sentence_info = model.tokenize(words)
            x = model.word_embedding(sentence_info)
            w = x[:, 1, :]  # B * H

            w_norm = np.linalg.norm(w, ord=None, axis=1).reshape((-1, 1))
            sim = np.dot(w, asp.T)  # B * LEN
            sim = np.divide(sim, w_norm)
            sim = np.divide(sim, asp_norm)

            cosine_sim[idx:idx + batch_count, :] = sim
            idx = idx + batch_count

    top_count = 30
    sim_texts = []

    for asp_idx, aspect in enumerate(MOVIE_ASPECT):
        sim_text = [aspect]
        cosine_rank = cosine_sim[:, asp_idx]
        cosine_rank = np.argsort(cosine_rank)[::-1]
        for i in range(0, top_count):
            sim_text.append(model.vocab.idx_to_token[cosine_rank[i]])
        sim_texts.append(sim_text)

    sim_texts = np.array(sim_texts, dtype=np.str)
    np.savetxt("similar_word.txt", sim_texts, fmt="%s", delimiter=" ", encoding='UTF-8')


if __name__ == '__main__':
    ex_base_model_training()
