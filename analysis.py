import os
import json
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

from datasets import _rocstories, _race, _arc, NUM2CHOI

def rocstories(data_dir, pred_path, log_path):
    preds = pd.read_csv(pred_path, delimiter='\t')['prediction'].values.tolist()
    _, _, _, labels = _rocstories(os.path.join(data_dir, 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'))
    test_accuracy = accuracy_score(labels, preds)*100.
    logs = [json.loads(line) for line in open(log_path)][1:]
    best_validation_index = np.argmax([log['va_acc'] for log in logs])
    valid_accuracy = logs[best_validation_index]['va_acc']
    print('ROCStories Valid Accuracy: %.2f'%(valid_accuracy))
    print('ROCStories Test Accuracy:  %.2f'%(test_accuracy))

def racem(data_dir, pred_path, log_path):
    preds = pd.read_csv(pred_path, delimiter='\t')['prediction'].values.tolist()
    *_, labels = _race(os.path.join(data_dir, "test", "middle"))
    labels = [NUM2CHOI[l] for l in labels]
    test_accuracy = accuracy_score(labels, preds)*100.
    logs = [json.loads(line) for line in open(log_path)][1:]
    best_validation_index = np.argmax([log['va_acc'] for log in logs])
    valid_accuracy = logs[best_validation_index]['va_acc']
    print('RACE-M Valid Accuracy: %.2f'%(valid_accuracy))
    print('RACE-M Test Accuracy:  %.2f'%(test_accuracy))

def race(data_dir, pred_path, log_path):
    preds = pd.read_csv(pred_path, delimiter='\t')['prediction'].values.tolist()
    *_, labels = _race(os.path.join(data_dir, "test", "high"))
    labels = [NUM2CHOI[l] for l in labels]
    test_accuracy = accuracy_score(labels, preds)*100.
    dirname, filename = os.path.split(pred_path)
    dev_preds = pd.read_csv(os.path.join(dirname, f"dev-{filename}"), delimiter='\t')['prediction'].values.tolist()
    *_, dlabels = _race(os.path.join(data_dir, "dev", "high"))
    dlabels = [NUM2CHOI[l] for l in dlabels]
    valid_accuracy = accuracy_score(dlabels, dev_preds)*100.
    #logs = [json.loads(line) for line in open(log_path)][1:]
    #best_validation_index = np.argmax([log['va_acc'] for log in logs])
    #valid_accuracy = logs[best_validation_index]['va_acc']
    print('RACE Valid Accuracy: %.2f'%(valid_accuracy))
    print('RACE Test Accuracy:  %.2f'%(test_accuracy))

def arc_easy(data_dir, pred_path, log_path):
    preds = pd.read_csv(pred_path, delimiter='\t')['prediction'].values.tolist()
    *_, labels = _arc(os.path.join(data_dir, "ARC-Easy", "ARC-Easy-Test.jsonl"))
    labels = [NUM2CHOI[l] for l in labels]
    test_accuracy = accuracy_score(labels, preds)*100.
    logs = [json.loads(line) for line in open(log_path)][1:]
    best_validation_index = np.argmax([log['va_acc'] for log in logs])
    valid_accuracy = logs[best_validation_index]['va_acc']
    print('ARC-Easy Valid Accuracy: %.2f'%(valid_accuracy))
    print('ARC-Easy Test Accuracy:  %.2f'%(test_accuracy))

def arc_challenge(data_dir, pred_path, log_path):
    preds = pd.read_csv(pred_path, delimiter='\t')['prediction'].values.tolist()
    *_, labels = _arc(os.path.join(data_dir, "ARC-Challenge", "ARC-Challenge-Test.jsonl"))
    labels = [NUM2CHOI[l] for l in labels]
    test_accuracy = accuracy_score(labels, preds)*100.
    logs = [json.loads(line) for line in open(log_path)][1:]
    best_validation_index = np.argmax([log['va_acc'] for log in logs])
    valid_accuracy = logs[best_validation_index]['va_acc']
    print('ARC-Challenge Valid Accuracy: %.2f'%(valid_accuracy))
    print('ARC-Challenge Test Accuracy:  %.2f'%(test_accuracy))

from utils import encode_dataset

def transform(art, ques, c1, c2, c3, c4, encoder):
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    clf_token = encoder['_classify_']
    transformed_dataset1 = []
    transformed_dataset2 = []
    transformed_dataset3 = []
    transformed_dataset4 = []
    for i, (x1, q, x2, x3, x4, x5), in enumerate(zip(art, ques, c1, c2, c3, c4)):
        x12 = [start]+x1[:469]+q[:23]+[delimiter]+x2[:17]+[clf_token]
        x13 = [start]+x1[:469]+q[:23]+[delimiter]+x3[:17]+[clf_token]
        x14 = [start]+x1[:469]+q[:23]+[delimiter]+x4[:17]+[clf_token]
        x15 = [start]+x1[:469]+q[:23]+[delimiter]+x5[:17]+[clf_token]
        transformed_dataset1.append(x12)
        transformed_dataset2.append(x13)
        transformed_dataset3.append(x14)
        transformed_dataset4.append(x15)

    return transformed_dataset1, \
           transformed_dataset2, \
           transformed_dataset3, \
           transformed_dataset4

def decode_dataset(*splits, decoder):
    decoded_splits = []
    for split in splits[0]:
        fields = []
        for field in split:
            field = decoder.decode_string(field)
            fields.append(field)
        decoded_splits.append(fields)
    return decoded_splits

#we add a encoder argument to pass the encoder in
def ea_race(data_dir, pred_path, log_path, encoder):
    encoder.decoder[encoder.encoder['_start_']] = '_start_</w>'
    encoder.decoder[encoder.encoder['_delimiter_']] = '_delimiter_</w>'
    encoder.decoder[encoder.encoder['_classify_']] = '_classify_</w>'
    preds = pd.read_csv(pred_path, delimiter='\t')['prediction'].values.tolist()
    passage, question, choice1, choice2, choice3, choice4, labels = \
         _race(os.path.join(data_dir, "test", "high"))
    labels = [NUM2CHOI[l] for l in labels]

    ed = encode_dataset(((passage, question, choice1, choice2, choice3, choice4),), encoder=encoder)[0]
    td = transform(*ed, encoder=encoder.encoder)
    dc1, dc2, dc3, dc4 = decode_dataset((td,), decoder=encoder)[0]

    f = open("result.tsv", "w+")
    header = ["passage",
              "question",
              "choice1",
              "choice2",
              "choice3",
              "choice4",
              "transform_c1",
              "transform_c2",
              "transform_c3",
              "transform_c4",
              "label",
              "prediction"]
    f.write('\t'.join(header))
    f.write('\n')

    for p, q, c1, c2, c3, c4, d1, d2, d3, d4, l, pred in zip(passage,
                                                             question,
                                                             choice1,
                                                             choice2,
                                                             choice3,
                                                             choice4,
                                                             dc1,
                                                             dc2,
                                                             dc3,
                                                             dc4,
                                                             labels,
                                                             preds):
        if l != pred:
            sample = [repr(p),
                      repr(q),
                      repr(c1),
                      repr(c2),
                      repr(c3),
                      repr(c4),
                      repr(d1),
                      repr(d2),
                      repr(d3),
                      repr(d4),
                      repr(l),
                      repr(pred)]
            f.write("\t".join(sample))
            f.write('\n')

    f.close()
