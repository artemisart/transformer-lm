import os
import csv
import numpy as np
from pathlib import Path

from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

seed = 3535999445

def imdb(path=Path("data/aclImdb/")):
    import pickle

    try:
        return pickle.load((path / "train-test.p").open("rb"))
    except FileNotFoundError:
        pass

    CLASSES = ["neg", "pos", "unsup"]
    def get_texts(path):
        texts,labels = [],[]
        for idx,label in tqdm(enumerate(CLASSES)):
            for fname in tqdm((path/label).glob('*.txt'), leave=False):
                texts.append(fname.read_text())
                labels.append(idx)
        return texts, np.asarray(labels)
    
    trXY = get_texts(path / "train")
    teXY = get_texts(path / "test")
    data = (trXY, teXY)
    pickle.dump(data, (path / "train-test.p").open("wb"))
    return data


def _rocstories(path):
    """Returns 4 lists :
        st: input sentences
        ct1: first answer
        ct2: second answer
        y: index of the good answer
    """
    with open(path, encoding='utf_8') as f:
        f = csv.reader(f)
        st = []
        ct1 = []
        ct2 = []
        y = []
        for i, line in enumerate(tqdm(list(f), ncols=80, leave=False)):
            if i > 0:
                s = ' '.join(line[1:5])
                c1 = line[5]
                c2 = line[6]
                st.append(s)
                ct1.append(c1)
                ct2.append(c2)
                y.append(int(line[-1])-1)
        return st, ct1, ct2, y

def rocstories(data_dir, n_train=1497, n_valid=374):
    storys, comps1, comps2, ys = _rocstories(os.path.join(data_dir, 'cloze_test_val__spring2016 - cloze_test_ALL_val.csv'))
    teX1, teX2, teX3, _ = _rocstories(os.path.join(data_dir, 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'))
    tr_storys, va_storys, tr_comps1, va_comps1, tr_comps2, va_comps2, tr_ys, va_ys = train_test_split(storys, comps1, comps2, ys, test_size=n_valid, random_state=seed)
    trX1, trX2, trX3 = [], [], []
    trY = []
    for s, c1, c2, y in zip(tr_storys, tr_comps1, tr_comps2, tr_ys):
        trX1.append(s)
        trX2.append(c1)
        trX3.append(c2)
        trY.append(y)

    vaX1, vaX2, vaX3 = [], [], []
    vaY = []
    for s, c1, c2, y in zip(va_storys, va_comps1, va_comps2, va_ys):
        vaX1.append(s)
        vaX2.append(c1)
        vaX3.append(c2)
        vaY.append(y)
    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)
    return (trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3)
