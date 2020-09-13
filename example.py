import kobert.pytorch_kobert
import kobert.utils
import loader
import model as md

import torch
import transformers

example_model_path = "example_model.pt"


def ex__training(opt=md.DEFAULT_OPTION, ctx="cuda:0"):
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
            x.retain_grad()
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

        torch.save(model.state_dict(), example_model_path)


def ex__sentiment_analysis():
    _, corpus_path = loader.download_corpus_data()  # Naver sentiment movie corpus v1.0
    md.sentiment_analysis(example_model_path, corpus_path, sentence_idx=1, label_idx=2)


if __name__ == '__main__':
    # ex__training()
    ex__sentiment_analysis()



