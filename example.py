import kobert.pytorch_kobert
import kobert.utils
import loader
from masa.model import ABSAModel
import masa.model as md

import gluonnlp as nlp
import torch
import transformers
import numpy as np
import random


ABSA_model_path = "ABSA_model.pt"
result_model_name = "ABSA_model_trained"
EX_SIM_WORD_LIST = [["연기", "연극"],
                    ["배우", "캐스팅", "모델"],
                    ["스토리", "이야기", "시나리오", "콘텐츠", "에피소드", "전개"],
                    ["액션", "전투", "싸움"],
                    ["감정", "감성", "심리"],
                    ["연출", "촬영", "편집"],
                    ["반전", "역전", "전환"],
                    ["음악", "노래", "사운드", "음향"],
                    ["규모", "스케일", "크기"]]


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


def ex__pre_training(opt=md.DEFAULT_OPTION, ctx="cuda:0"):
    device = torch.device(ctx)

    # load bert model
    bert_model, vocab = kobert.pytorch_kobert.get_pytorch_kobert_model()

    # load train / test dataset
    bert_tokenizer = md.get_bert_tokenizer(vocab)

    train_data_path, test_data_path = loader.download_corpus_data()  # Naver sentiment movie corpus v1.0
    dataset_train = md.get_bert_dataset(train_data_path, sentence_idx=1, label_idx=2,
                                        max_len=opt["max_len"], bert_tokenizer=bert_tokenizer)
    dataset_test = md.get_bert_dataset(test_data_path, sentence_idx=1, label_idx=2,
                                       max_len=opt["max_len"], bert_tokenizer=bert_tokenizer)

    # data loader
    train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=opt["batch_size"], num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=opt["batch_size"], num_workers=0)

    # model
    model = md.BERTClassifier(bert_model, dr_rate=opt["drop_out_rate"]).to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=opt["learning_rate"])
    loss_function = torch.nn.CrossEntropyLoss()

    t_total = len(train_dataloader) * opt["num_epochs"]
    warmup_steps = int(t_total * opt["warmup_ratio"])
    scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer, warmup_steps, t_total)

    for e in range(opt["num_epochs"]):
        train_accuracy = 0.0
        test_accuracy = 0.0

        # Train Batch
        model.train()

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(train_dataloader):
            optimizer.zero_grad()

            # set train batch
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)

            # get word embedding
            attention_mask = md.gen_attention_mask(token_ids, valid_length)
            word_embedding = model.bert.get_input_embeddings()
            x = word_embedding(token_ids)

            # forward propagation
            out = model(x, segment_ids, attention_mask)

            # backward propagation
            loss = loss_function(out, label)
            loss.backward()

            # optimization
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt["max_grad_norm"])
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_accuracy += md.calculate_accuracy(out, label)

            if batch_id % opt["log_interval"] == 0:
                print("epoch {} batch id {} loss {} train accuracy {}".format(e + 1, batch_id + 1,
                                                                              loss.data.cpu().numpy(),
                                                                              train_accuracy / (batch_id + 1)))
        print("epoch {} train accuracy {}".format(e + 1, train_accuracy / (batch_id + 1)))

        # Test Batch
        model.eval()
        with torch.no_grad():
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
                # set test batch
                token_ids = token_ids.long().to(device)
                segment_ids = segment_ids.long().to(device)
                valid_length = valid_length
                label = label.long().to(device)

                # get word embedding
                attention_mask = md.gen_attention_mask(token_ids, valid_length)
                word_embedding = model.bert.get_input_embeddings()
                x = word_embedding(token_ids)

                # forward propagation
                out = model(x, segment_ids, attention_mask)

                # test accuracy
                test_accuracy += md.calculate_accuracy(out, label)
            print("epoch {} test accuracy {}".format(e + 1, test_accuracy / (batch_id + 1)))

        torch.save(model.state_dict(), sa_model_path)


def ex__ABSA_training(opt=md.DEFAULT_OPTION, ctx="cuda:0"):
    ABSA_model = ABSAModel(ctx=ctx)
    ABSA_model.load_kobert()
    ABSA_model.load_model(model_path=None)

    random.seed(3)
    np.random.seed(3)

    # load dataset
    total_dataset = nlp.data.TSVDataset("sentiment_dataset.csv", field_indices=[0, 1, 3], num_discard_samples=1)
    train_dataset = total_dataset

    object_text_0 = opt["object_text_0"]
    object_text_1 = opt["object_text_1"]

    # --------------- DATA AUGMENTATION FOR ABSA ---------------
    # ----------------------------------------------------------
    # data list
    train_data_list = []
    data_list = [train_data_list]
    dataset_list = [train_dataset]

    # set train / test data with data augmentation
    for didx, dataset in enumerate(dataset_list):
        pos_dataset = []
        neg_dataset = []
        for data in dataset:
            if data[2] == 'positive':
                pos_dataset.append(data)
            else:
                neg_dataset.append(data)

        # random value
        rnd_0 = np.random.uniform(0, 1, len(dataset)) > 0.5
        rnd_1 = np.random.randint(0, len(dataset) - 1, len(dataset))
        rnd_2 = np.random.uniform(0, 1, len(dataset)) > 0.5
        rnd_3 = np.random.randint(0, len(neg_dataset), len(pos_dataset))
        rnd_4 = np.random.uniform(0, 1, len(pos_dataset)) > 0.5

        # split by list
        for idx, (corpus, aspect, label) in enumerate(list(dataset)):
            # original data
            data_list[didx].append([corpus, [1, 1]])

            # augmented data - single
            label_number = 2 if label == "positive" else 0
            if rnd_0[idx]:
                aug_corpus = corpus.replace(aspect, object_text_0)
                aug_label = [label_number, 1]
            else:
                aug_corpus = corpus.replace(aspect, object_text_1)
                aug_label = [1, label_number]
            data_list[didx].append([aug_corpus, aug_label])

            # augmented data - double
            rnd_1[idx] = len(dataset) - 1 if rnd_1[idx] == idx else rnd_1[idx]
            corpus_1, aspect_1, label_1 = dataset[rnd_1[idx]]
            label_number_1 = 2 if label_1 == "positive" else 0

            if rnd_2[idx]:
                aug_text_0 = object_text_0
                aug_text_1 = object_text_1
            else:
                aug_text_0 = object_text_1
                aug_text_1 = object_text_0

            aug_corpus = corpus.replace(aspect, aug_text_0) + " " + corpus_1.replace(aspect_1, aug_text_1)
            aug_label = [label_number, label_number_1]
            data_list[didx].append([aug_corpus, aug_label])

        # augmented data - counter double
        for idx, (pos_corpus, pos_aspect, _) in enumerate(pos_dataset):
            neg_corpus, neg_aspect, _ = neg_dataset[rnd_3[idx]]
            pos_label_number = 2
            neg_label_number = 0

            if rnd_4[idx]:
                left_corpus = pos_corpus.replace(pos_aspect, object_text_0)
                right_corpus = neg_corpus.replace(neg_aspect, object_text_1)
                aug_label = [pos_label_number, neg_label_number]
            else:
                left_corpus = neg_corpus.replace(neg_aspect, object_text_0)
                right_corpus = pos_corpus.replace(pos_aspect, object_text_1)
                aug_label = [neg_label_number, pos_label_number]
            aug_corpus = left_corpus + " " + right_corpus
            data_list[didx].append([aug_corpus, aug_label])

    # random shuffle
    random.shuffle(train_data_list)

    #
    train_corpus_list = []
    train_label_list = []
    for (corpus, label) in train_data_list:
        train_corpus_list.append(corpus)
        train_label_list.append(np.array(label, dtype=np.int32))

    # ----------------- ABSA CLASSIFIER MODEL ------------------
    # ----------------------------------------------------------

    # create batch loader
    sentence_info = ABSA_model.tokenize(train_corpus_list)

    for idx, tuple_item in enumerate(sentence_info):
        sentence_info[idx] = tuple_item + (train_label_list[idx], )
    batch_loader = torch.utils.data.DataLoader(sentence_info, batch_size=opt["batch_size"], num_workers=0)

    # aspect-based sentiment analysis model
    model = ABSA_model.model

    # optimization - AdamW
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=opt["learning_rate"])

    # loss function - Cross Entropy
    loss_function = torch.nn.CrossEntropyLoss()

    # warmup scheduler
    t_total = len(batch_loader) * opt["num_epochs"]
    warmup_steps = int(t_total * opt["warmup_ratio"])
    scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer, warmup_steps, t_total)

    # -------------- ABSA CLASSIFIER MODEL TRAIN ---------------
    # ----------------------------------------------------------
    for e in range(opt["num_epochs"]):
        train_accuracy = 0.0

        # Train Batch
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(batch_loader):
            optimizer.zero_grad()

            # set train batch
            token_ids = token_ids.long().to(ABSA_model.device)
            segment_ids = segment_ids.long().to(ABSA_model.device)
            valid_length = valid_length
            label = label.long().to(ABSA_model.device)

            # get word embedding
            attention_mask = md.gen_attention_mask(token_ids, valid_length)
            word_embedding = model.bert.get_input_embeddings()
            x = word_embedding(token_ids)

            # forward propagation
            _, out_1, out_2 = model(x, segment_ids, attention_mask, sa=False, absa=True)

            # backward propagation
            loss = loss_function(out_1, label[:, 0]) + loss_function(out_2, label[:, 1])
            loss.backward()

            # optimization
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt["max_grad_norm"])
            optimizer.step()
            scheduler.step()  # Update learning rate schedule

            accuracy_0 = md.calculate_accuracy(out_1, label[:, 0])
            accuracy_1 = md.calculate_accuracy(out_2, label[:, 1])
            train_accuracy += ((accuracy_0 + accuracy_1) / 2)  # Average of correct count

            if batch_id % opt["log_interval"] == 0:
                print("epoch {} batch id {} loss {} train accuracy {}".format(e + 1, batch_id + 1,
                                                                              loss.data.cpu().numpy(),
                                                                              train_accuracy / (batch_id + 1)))
        print("epoch {} train accuracy {}".format(e + 1, train_accuracy / (batch_id + 1)))

        # Validation
        r1, r2, r3 = _model_validation(ABSA_model)
        print(f"total accuracy: {'%0.2f' % r1}%, case_0 accuracy: {'%0.2f' % r2}%, case_1 accuracy: {'%0.2f' % r3}%")

        torch.save(model.state_dict(), result_model_name + f"_{e}.pt")


def ex__ABSA(model_path=ABSA_model_path, opt=md.DEFAULT_OPTION, ctx="cuda:0"):
    device = torch.device(ctx)
    bert_model, vocab = kobert.pytorch_kobert.get_pytorch_kobert_model()

    model = md.ABSAClassifier(bert_model, torch.nn.Linear(768, 2)).to(device)
    model.load_state_dict(torch.load(model_path))

    corpus = [["오늘 밥먹었는데 정말 최고였어요. 근데 영화 대상 진짜 안좋더라구요", [0, 1]]]
    bert_tokenizer = md.get_bert_tokenizer(vocab)
    dataset = md.BERTDataset(corpus, 0, 1, bert_tokenizer, opt["max_len"], pad=True, pair=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0)

    model.eval()
    with torch.no_grad():
        for batch_id, (token_ids, valid_length, segment_ids, _) in enumerate(dataloader):
            # set test batch
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length

            # get word embedding
            attention_mask = md.gen_attention_mask(token_ids, valid_length)
            word_embedding = model.bert.get_input_embeddings()
            x = word_embedding(token_ids)

            # forward propagation
            _, out_1, out_2 = model(x, segment_ids, attention_mask, sa=False, absa=True)
            print(out_1, out_2)


def ex__cosine_similarity(model_path=ABSA_model_path, ctx="cuda:0"):
    model = md.ABSAModel(ctx)
    model.load_kobert()
    model.load_model(model_path)

    MOVIE_ASPECT = ["연기", "배우", "스토리", "액션", "감정", "연출", "반전", "음악", "규모"]
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

            cosine_sim[idx:idx+batch_count, :] = sim
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
    ex__ABSA_training()







