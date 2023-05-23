import json
import codecs
import jsonlines

import numpy as np
import matplotlib.pyplot as plt


json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


def jsonlines_load(f):
    with jsonlines.open(f) as reader:
        data = [sample for sample in reader]
    return data


def jsonlines_dump(d, f):
    with jsonlines.open(f, mode='w') as writer:
        writer.write_all(d)


def sort_dict(_dict, reverse=True):
    if len(_dict) == 0:
        return {}
    _list = [(k, v) for k, v in _dict.items()]
    if reverse:
        if isinstance(_list[0][1], dict):
            _list.sort(key=lambda x: (-x[1]['cnt'], x[0]))
        else:
            _list.sort(key=lambda x: (-x[1], x[0]))
    else:
        if isinstance(_list[0][1], dict):
            _list.sort(key=lambda x: (x[1]['cnt'], x[0]))
        else:
            _list.sort(key=lambda x: (x[1], x[0]))
    return {x[0]: x[1] for x in _list}


def load_vocab(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read().strip().split('\n')
    data = {l.split('\t')[0]: int(l.split('\t')[-1]) for l in data}
    return sort_dict(data)


def dump_vocab(vocab, filename, _type='w'):
    with open(filename, _type, encoding='utf-8') as f:
        f.write('\n'.join([f'{k}\t\t\t\t\t\t{v}' for k,v in vocab.items()]))
    

def draw_scatter(X, Y):
    x = np.asarray(X)
    y = np.asarray(Y)

    color_list = [0.75, 0.25, 0.95, 0.5]
    colors = []
    for _x in X:
        if _x < 20: colors.append(color_list[0])
        elif _x < 40: colors.append(color_list[1])
        elif _x < 60: colors.append(color_list[2])
        else: colors.append(color_list[3])
    colors = np.asarray(colors)

    plt.scatter(x, y, s=5, c=colors, alpha=0.8)
    plt.show()