import os
import pandas as pd
from datasets import _race, NUM2CHOI
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
def ea_race(data_dir, pred_path, log_path, text_encoder): 
    #adding decode token for the 3 special token
    text_encoder.decoder[text_encoder.encoder['_start_']] = '_start_</w>'
    text_encoder.decoder[text_encoder.encoder['_delimiter_']] = '_delimiter_</w>'
    text_encoder.decoder[text_encoder.encoder['_classify_']] = '_classify_</w>'

    preds = pd.read_csv(pred_path, delimiter='\t')['prediction'].values.tolist()
    passage, question, choice1, choice2, choice3, choice4, labels = \
         _race(os.path.join(data_dir, "test", "high"))
    labels = [NUM2CHOI[l] for l in labels]

    ed = encode_dataset(((passage, question, choice1, choice2, choice3, choice4),), encoder=text_encoder)[0]
    td = transform(*ed, encoder=text_encoder.encoder)
    dc1, dc2, dc3, dc4 = decode_dataset((td,), decoder=text_encoder)[0]

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

    ftxt = open("result.txt", "w+")
    ctxt = 1
    for p, q, c1, c2, c3, c4, d1, d2, d3, d4, l, pred in zip(passage, question, choice1, choice2, choice3, choice4, dc1, dc2, dc3, dc4, labels, preds):
        if l != pred:
           ftxt.write(f"sample{ctxt}\n")
           ftxt.write(f"passage: {repr(p)}\n")
           ftxt.write(f"question: {repr(q)}\n")
           ftxt.write(f"choice1: {repr(c1)}\n")
           ftxt.write(f"choice2: {repr(c2)}\n")
           ftxt.write(f"choice3: {repr(c3)}\n")
           ftxt.write(f"choice4: {repr(c4)}\n")
           ftxt.write(f"decode input1: {repr(d1)}\n")
           ftxt.write(f"decode input2: {repr(d2)}\n")
           ftxt.write(f"decode input3: {repr(d3)}\n")
           ftxt.write(f"decode input4: {repr(d4)}\n")
           ftxt.write(f"label: {repr(l)}\n")
           ftxt.write(f"prediction: {repr(pred)}\n")
           ftxt.write("\n")
           ctxt+=1

    ftxt.close()
    f.close()
