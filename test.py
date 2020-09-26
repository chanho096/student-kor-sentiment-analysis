import loader
import torch
import numpy as np
import random

import kobert.pytorch_kobert
import masa.model
import gluonnlp as nlp
import transformers

DP_LABEL_SYNTAX_DICT = {'NULL': 0,  # Pad
                        'NP': 1,  # 체언
                        'VP': 2,  # 용언
                        'AP': 3,  # 부사구
                        'VNP': 4,  # 긍정 지정사구
                        'DP': 5,  # 관형사구
                        'IP': 6,  # 감탄사구
                        'X': 7,  # 의사 구
                        'L': 8,  # 부호
                        'R': 9  # 부호
                        }

DP_LABEL_FUNCTION_DICT = {'NULL': 0,  # Pad
                          'SBJ': 1,  # 주어
                          'OBJ': 2,  # 목적어
                          'MOD': 3,  # 관형어 (체언 수식어)
                          'AJT': 4,  # 부사어 (용언 수식어)
                          'CMP': 5,  # 보어
                          'CNJ': 6  # 접속어
                          }

DP_MAX_WORD_LENGTH = loader.dp_max_word_length
DP_LABEL_SYNTAX_LENGTH = len(DP_LABEL_SYNTAX_DICT)
DP_LABEL_FUNCTION_LENGTH = len(DP_LABEL_FUNCTION_DICT)

dp_model_path = "DP_model.pt"


class MultipleSoftmax(torch.nn.Module):
    def __init__(self,
                 num_hidden,
                 num_layers,
                 num_classes,
                 dr_rate=None):
        super(MultipleSoftmax, self).__init__()

        self.classifiers = torch.nn.ModuleList([torch.nn.Linear(num_hidden, num_classes) for _ in range(0, num_layers)])

        self.dr_rate = dr_rate
        if dr_rate:
            self.dropouts = [torch.nn.Dropout(p=dr_rate) for _ in range(0, num_layers)]

    def forward(self, input_tensor):
        """
              x: input tensor - shape (batch_size, num_hidden, num_layers)
        """
        result = []
        hidden = []
        for idx, classifier in enumerate(self.classifiers):
            x = input_tensor[:, :, idx]
            out = self.dropouts[idx](x) if self.dr_rate else x
            out = classifier(out)
            hidden.append(out)

            out = torch.nn.functional.softmax(out, dim=1)
            result.append(out)

        return result, hidden


class SimpleDP(torch.nn.Module):
    def __init__(self,
                 bert,
                 num_hiddens=768,
                 num_classes_0=DP_LABEL_SYNTAX_LENGTH,
                 num_classes_1=DP_LABEL_FUNCTION_LENGTH,
                 dr_rate_0=None,
                 ):
        super(SimpleDP, self).__init__()
        self.bert = bert
        self.dr_rate_0 = dr_rate_0

        self.word_classifier = torch.nn.Linear(in_features=64, out_features=32)
        self.multiple_softmax_0 = MultipleSoftmax(num_hiddens, 32, num_classes_0)
        self.multiple_softmax_1 = MultipleSoftmax(num_hiddens, 32, num_classes_1)

        if dr_rate_0:
            self.dropout_0 = torch.nn.Dropout(p=dr_rate_0)
            self.dropout_1 = torch.nn.Dropout(p=dr_rate_0)

    def forward(self, x, segment_ids, attention_mask):
        # bert forward
        encoder_out, pooler = self.bert(inputs_embeds=x, token_type_ids=segment_ids.long(),
                                        attention_mask=attention_mask)

        out = torch.transpose(encoder_out, 1, 2)
        out = self.dropout_0(out) if self.dr_rate_0 else out
        word_vectors = self.word_classifier(out)
        word_vectors = torch.tanh(word_vectors)

        result_0, hidden_0 = self.multiple_softmax_0(word_vectors)
        result_1, hidden_1 = self.multiple_softmax_1(word_vectors)

        return pooler, word_vectors, (result_0, hidden_0), (result_1, hidden_1)


class DPSA(torch.nn.Module):
    def __init__(self,
                 simple_dp,
                 num_hiddens=768,
                 dr_rate_0=None,
                 dr_rate_1=None
                 ):
        super(DPSA, self).__init__()
        self.simple_dp = simple_dp
        self.dr_rate_0 = dr_rate_0
        self.dr_rate_1 = dr_rate_1

        self.classifier_0 = torch.nn.Linear(32, 1)
        self.classifier_1 = torch.nn.Linear(num_hiddens*2 + DP_LABEL_SYNTAX_LENGTH + DP_LABEL_FUNCTION_LENGTH, 2)
        self.classifier_tmp = torch.nn.Linear(768, 2)

        if dr_rate_0:
            self.dropout_0 = torch.nn.Dropout(p=dr_rate_0)
        if dr_rate_1:
            self.dropout_1 = torch.nn.Dropout(p=dr_rate_1)

    def forward(self, x, segment_ids, attention_mask):
        # bert forward
        pooler, word_vectors, (result_0, hidden_0), (result_1, hidden_1) = \
            self.simple_dp(x, segment_ids, attention_mask)

        """
        result_stack_0 = torch.stack(result_0, dim=2)
        result_stack_1 = torch.stack(result_1, dim=2)
        dp_result = torch.cat((word_vectors, result_stack_0, result_stack_1), dim=1)

        dp_out = self.dropout_0(dp_result) if self.dr_rate_0 else dp_result
        dp_out = self.classifier_0(dp_out)  # pooled dp_output
        dp_out = torch.squeeze(dp_out, dim=2)
        dp_out = torch.tanh(dp_out)
        """
        out_0 = self.dropout_1(pooler) if self.dr_rate_1 else pooler
        out_0 = self.classifier_tmp(out_0)
        out_0 = torch.nn.functional.softmax(out_0, dim=1)

        return out_0

        classifier_input = torch.cat((pooler, dp_out), dim=1)
        out = self.dropout_1(classifier_input) if self.dr_rate_1 else classifier_input
        out = self.classifier_1(out)
        out = torch.nn.functional.softmax(out, dim=1)

        return out


def dp_label_encoder(label_list):
    labels = []

    for label_keys in label_list:
        label_syn = np.zeros((loader.dp_max_word_length, ), dtype=np.int32)
        label_fun = np.zeros((loader.dp_max_word_length, ), dtype=np.int32)
        for idx, label_key in enumerate(label_keys):
            split_key = label_key.split('_')
            key_syn = split_key[0]
            key_fun = split_key[1] if len(split_key) > 1 else 'NULL'

            label_syn[idx] = DP_LABEL_SYNTAX_DICT[key_syn]
            label_fun[idx] = DP_LABEL_FUNCTION_DICT[key_fun]

        labels.append([label_syn, label_fun])

    return labels


def ex_dp_training(ctx="cuda:0", opt=masa.model.DEFAULT_OPTION):
    # ------------------------------------------
    # load data
    corpus_list, label_list, head_list = loader.load_dependency_parsing_data()
    label_list = dp_label_encoder(label_list)

    # ------------------------------------------
    # set model
    bert_model, vocab = kobert.pytorch_kobert.get_pytorch_kobert_model()
    bert_tokenizer = masa.model.get_bert_tokenizer(vocab)

    # tokenizing
    transform = nlp.data.BERTSentenceTransform(
        bert_tokenizer, max_seq_length=opt["max_len"], pad=True, pair=False)

    sentence_info = []
    for corpus in corpus_list:
        sentence = transform([corpus])
        """
            ##### Transform Result #####
            sentence[0]: token_ids, numpy array
            sentence[1]: valid_length, array type scalar
            sentence[2]: segment_ids, numpy array
        """
        sentence_info.append(sentence)

    for idx, tuple_item in enumerate(sentence_info):
        sentence_info[idx] = tuple_item + (label_list[idx][0], label_list[idx][1], )

    random.shuffle(sentence_info)
    validation_rate = 0.1
    val_idx = int(len(sentence_info) * validation_rate)

    test_batch_loader = torch.utils.data.DataLoader(sentence_info[:val_idx], batch_size=opt["batch_size"],
                                                    num_workers=0, shuffle=True)
    train_batch_loader = torch.utils.data.DataLoader(sentence_info[val_idx:], batch_size=opt["batch_size"],
                                                     num_workers=0, shuffle=True)

    # ------------------------------------------
    # model
    device = torch.device(ctx)
    model = SimpleDP(bert_model).to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=opt["learning_rate"])
    lf = torch.nn.CrossEntropyLoss()

    t_total = len(train_batch_loader) * opt["num_epochs"]
    warmup_steps = int(t_total * opt["warmup_ratio"])
    scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer, warmup_steps, t_total)

    # ------------------------------------------
    # train / test
    for e in range(opt["num_epochs"]):
        train_accuracy = 0.0
        test_accuracy = 0.0

        # Train Batch
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label_0, label_1) in enumerate(train_batch_loader):
            optimizer.zero_grad()

            # set train batch
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label_0 = label_0.long().to(device)
            label_1 = label_1.long().to(device)

            # get word embedding
            attention_mask = masa.model.gen_attention_mask(token_ids, valid_length)
            word_embedding = model.bert.get_input_embeddings()
            x = word_embedding(token_ids)

            # forward propagation
            _, word_vector, (result_0, _), (result_1, _) = model(x, segment_ids, attention_mask)

            # backward propagation
            loss = lf(result_0[0], label_0[:, 0])
            for idx, result in enumerate(result_0[1:]):
                loss = loss + lf(result, label_0[:, idx + 1])
            for idx, result in enumerate(result_1):
                loss = loss + lf(result, label_1[:, idx])
            loss.backward()

            # optimization
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt["max_grad_norm"])
            optimizer.step()
            scheduler.step()  # Update learning rate schedule

            batch_accuracy = 0
            for idx, result in enumerate(result_0):
                batch_accuracy += masa.model.calculate_accuracy(result, label_0[:, idx])
            for idx, result in enumerate(result_1):
                batch_accuracy += masa.model.calculate_accuracy(result, label_1[:, idx])
            batch_accuracy = batch_accuracy / (len(result_0) + len(result_1))
            train_accuracy += batch_accuracy

            if batch_id % opt["log_interval"] == 0:
                print("epoch {} batch id {} loss {} train accuracy {}".format(e + 1, batch_id + 1,
                                                                              loss.data.cpu().numpy(),
                                                                              train_accuracy / (batch_id + 1)))
        print("epoch {} train accuracy {}".format(e + 1, train_accuracy / (batch_id + 1)))

        # Test Batch
        model.eval()
        with torch.no_grad():
            for batch_id, (token_ids, valid_length, segment_ids, label_0, label_1) in enumerate(test_batch_loader):
                # set test batch
                token_ids = token_ids.long().to(device)
                segment_ids = segment_ids.long().to(device)
                valid_length = valid_length
                label_0 = label_0.long().to(device)
                label_1 = label_1.long().to(device)

                # get word embedding
                attention_mask = masa.model.gen_attention_mask(token_ids, valid_length)
                word_embedding = model.bert.get_input_embeddings()
                x = word_embedding(token_ids)

                # forward propagation
                _, word_vector, (result_0, _), (result_1, _) = model(x, segment_ids, attention_mask)

                # backward propagation
                loss = lf(result_0[0], label_0[:, 0])
                for idx, result in enumerate(result_0[1:]):
                    loss = loss + lf(result, label_0[:, idx + 1])
                for idx, result in enumerate(result_1):
                    loss = loss + lf(result, label_1[:, idx])

                # test accuracy
                batch_accuracy = 0
                for idx, result in enumerate(result_0):
                    batch_accuracy += masa.model.calculate_accuracy(result, label_0[:, idx])
                for idx, result in enumerate(result_1):
                    batch_accuracy += masa.model.calculate_accuracy(result, label_1[:, idx])
                batch_accuracy = batch_accuracy / (len(result_0) + len(result_1))
                test_accuracy += batch_accuracy

            print("epoch {} test accuracy {}".format(e + 1, test_accuracy / (batch_id + 1)))

        torch.save(model.state_dict(), f"pre_trained_model_{e + 1}.pt")


def ex_pre_training(opt=masa.model.DEFAULT_OPTION, ctx="cuda:0"):
    # ------------------------------------------
    # load bert
    bert_model, vocab = kobert.pytorch_kobert.get_pytorch_kobert_model()
    bert_tokenizer = masa.model.get_bert_tokenizer(vocab)
    transform = nlp.data.BERTSentenceTransform(
        bert_tokenizer, max_seq_length=opt["max_len"], pad=True, pair=False)

    # ---------------------- Data Loader -----------------------
    # ----------------------------------------------------------
    # Naver sentiment movie corpus v1.0
    data_path = loader.download_movie_corpus_data()

    batch_loaders = []
    for path in data_path:
        dataset = nlp.data.TSVDataset(path, field_indices=[1, 2], num_discard_samples=1)

        corpus_list = []
        label_list = []
        sentence_info = []
        for (corpus, label) in list(dataset):
            corpus_list.append(corpus)

            # set label array
            label_array = np.array(label, dtype=np.int32)

            # set label list
            label_list.append(label_array)

            # tokenizing
            sentence = transform([corpus])
            sentence_info.append(sentence)

        for idx, tuple_item in enumerate(sentence_info):
            sentence_info[idx] = tuple_item + (label_list[idx],)

        batch_loader = torch.utils.data.DataLoader(sentence_info, batch_size=opt["batch_size"], num_workers=0)
        batch_loaders.append(batch_loader)

    train_batch_loader, test_batch_loader = batch_loaders

    # ----------------- ABSA CLASSIFIER MODEL ------------------
    # ----------------------------------------------------------
    device = torch.device(ctx)

    simple_dp_model = SimpleDP(bert_model).to(device)
    # simple_dp_model.load_state_dict(torch.load(dp_model_path, map_location=device))
    model = DPSA(simple_dp_model, dr_rate_0=0.2, dr_rate_1=0.2).to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=opt["learning"
                                                                        "_rate"])
    lf = torch.nn.CrossEntropyLoss()

    t_total = len(train_batch_loader) * opt["num_epochs"]
    warmup_steps = int(t_total * opt["warmup_ratio"])
    scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer, warmup_steps, t_total)

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
            word_embedding = simple_dp_model.bert.get_input_embeddings()
            x = word_embedding(token_ids)

            # forward propagation
            out = model(x, segment_ids, attention_mask)

            # backward propagation
            loss = lf(out, label)
            loss.backward()

            # optimization
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt["max_grad_norm"])
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_accuracy += masa.model.calculate_accuracy(out, label)

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
                word_embedding = simple_dp_model.bert.get_input_embeddings()
                x = word_embedding(token_ids)

                # forward propagation
                out = model(x, segment_ids, attention_mask)

                # test accuracy
                test_accuracy += masa.model.calculate_accuracy(out, label)
            print("epoch {} test accuracy {}".format(e + 1, test_accuracy / (batch_id + 1)))

        torch.save(model.state_dict(), f"pre_trained_model_{e+1}.pt")


if __name__ == '__main__':
    ex_pre_training()





