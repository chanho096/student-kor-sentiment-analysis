import loader
import torch
import numpy as np

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
        for idx, classifier in enumerate(self.classifiers):
            x = input_tensor[:, :, idx]
            out = self.dropouts[idx](x) if self.dr_rate else x
            out = classifier(out)
            out = torch.nn.functional.softmax(out)
            result.append(out)

        return result


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
        word_vectors = torch.nn.functional.tanh(word_vectors)

        result_0 = self.multiple_softmax_0(word_vectors)
        result_1 = self.multiple_softmax_1(word_vectors)

        return word_vectors, result_0, result_1


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

    batch_loader = torch.utils.data.DataLoader(sentence_info, batch_size=opt["batch_size"],
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

    t_total = len(batch_loader) * opt["num_epochs"]
    warmup_steps = int(t_total * opt["warmup_ratio"])
    scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer, warmup_steps, t_total)

    # ------------------------------------------
    # train / test
    for e in range(opt["num_epochs"]):
        train_accuracy = 0.0
        test_accuracy = 0.0

        # Train Batch
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label_0, label_1) in enumerate(batch_loader):
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
            word_vector, result_0, result_1 = model(x, segment_ids, attention_mask)

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

        torch.save(model.state_dict(), f"pre_trained_model_{e + 1}.pt")


if __name__ == '__main__':
    """
    a, b, c = loader.load_dependency_parsing_data()

    label_type = []
    for ll in b:
        for l in ll:
            if l not in label_type:
                label_type.append(l)

    print(label_type)
    """
    ex_dp_training()





