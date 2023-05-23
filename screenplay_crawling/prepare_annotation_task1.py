import string
import random
import collections

from tqdm import tqdm
from math import floor, ceil
from nltk import sent_tokenize

from utils import json_load, json_dump

random.seed(100)


def calculate_time(times):
    times = times.split(':')
    _time = float(times[0]) * 60 + float(times[1])
    return _time


def same_char(chars1, chars2):
    chars1 = [x.lower() for x in chars1]
    chars2 = [x.lower() for x in chars2]
    bad_cases = [0, 0]
    both = list(set(chars1 + chars2))
    for c in chars1:
        if all(c not in x and x not in c for x in chars2):
            bad_cases[0] += 1
    for c in chars2:
        if all(c not in x and x not in c for x in chars1):
            bad_cases[1] += 1
    bad_cases[0] = 1 - bad_cases[0] / max(len(chars1), 1)
    bad_cases[1] = 1 - bad_cases[1] / max(len(chars2), 1)
    _max = max(bad_cases)
    _min = min(bad_cases)

    if _min >= 0.75 and _max == 1 and len(both) < 5:
        return 1
    else:
        return (_max + _min) / 2


def translate_event(ev, chrct):
    sents = sent_tokenize(ev)
    sent = []
    for s in sents:
        if not len(sent) and chrct.lower() not in s.lower(): continue
        if chrct.lower() in s.lower():
            sent.append(s.strip())
        elif len(sent):
            sent.append(s.strip())
        if len(sent) > 1:
            break
    sent = ' '.join(sent).lstrip('(').rstrip(')').rstrip(string.punctuation)

    if sent.startswith('[['):
        speaker = sent.split(']]')[0].split('[[')[-1].strip()
        content = ':'.join(sent.split(':')[1:]).replace('(recorded)', '').strip()
        sent = f'{speaker} says: {content}'

    return sent


def str_time(_time, _ceil=True):
    _time = ceil(_time) if _ceil else floor(_time)
    seconds = '{:02d}'.format(int(_time % 60))
    minutes = '{:02d}'.format(int(_time // 60))
    return f'{minutes}:{seconds}'


TO_IGNORE = ['officers', 'officer', 'manager', 'waiter', 'waitress']


if __name__ == '__main__':
    raw = json_load('preprocessed/v0_csi_corpus.json')

    cnt = [[], 0]

    # re-distribute the datapoints for user-job assignment
    samples = []
    _id = 0
    for data in raw:
        sample = {
            'id': _id,
            'episode': data['episode'],
            'title': data['title'],
            'annot_cnt': 0,
            'annot': []
        }
        annots = []
        sid = 0
        for script in data['scripts']:
            chars = collections.defaultdict(int)
            for i, c in enumerate(script['selected_characters']):
                if any(
                    x.lower() in c.lower() for x in script['selected_characters'][:i] \
                    + script['selected_characters'][i+1:]
                ): continue
                if c.lower() in TO_IGNORE: continue
                for v in script['characters']:
                    if c.lower() in v['form'].lower():
                        chars[c] += v['cnt']
            # requirement 1: characters appearing more than once
            selected = [k for k,v in chars.items() if v > 1]
            annot = {
                'sid': sid,
                'raw_sid': script['raw_bid'],
                'start_time': str_time(script['start_time'], _ceil=False),
                'end_time': str_time(script['end_time']),
                'script': '\n'.join([x['text'] for x in script['content']]),
                'characters': [],
                'test': []
            }
            # requirement 2: the script is long enough
            if len(script['content']) < 3: continue
            for c in selected:
                sents = [
                    (i, ct['text']) for i, ct in enumerate(script['content']) \
                    if c.lower() in ct['text'].lower()
                ]
                if len(sents) > 2:
                    # randomly select 2 sentences, one from the 1st half, another from the 2nd half
                    sents = [
                        random.choice(sents[:len(sents) // 2 + 1]), 
                        random.choice(sents[len(sents) // 2:])
                    ]
                for sent in sents:
                    i, sent = sent
                    # requirement 3: randomly pick a character's name to start with
                    chars_set = [k for k in chars if k.lower() in sent.lower()]
                    # requirement 4: pick sentences with prop of 50% each time
                    if random.random() < 0.5: continue
                    if i < len(script['content']) // 2 and i != len(script['content']) - 1:
                        annot['characters'].append(c)
                        annot['test'].append([
                            "What is probably caused/enabled by the event that",
                            translate_event(sent, c),
                            random.choice(chars_set),
                        ])
                    elif i >= len(script['content']) // 2 and i != 0:
                        annot['characters'].append(c)
                        annot['test'].append([
                            "What probably caused/enabled the event that",
                            translate_event(sent, c),
                            random.choice(chars_set),
                        ])
            if len(annot['characters']):
                annots.append(annot)
                sid += 1
                cnt[0] += [len(annot['test'])]
                cnt[1] += 1
                if len(cnt[0]) > 800:
                    import ipdb; ipdb.set_trace()
        sample['annot'] = annots
        sample['annot_cnt'] = len(annots)
        samples.append(sample)
        _id += 1

    print('{} samples / {} clips'.format(sum(cnt[0]), cnt[1]))

    annotators = {i: {} for i in range(15)}
    _id = 0
    for eps in samples:
        for ant in eps['annot']:
            for idx in range(len(ant['test'])):
                a_id = _id % 15
                if random.random() < 0.2: continue
                if eps['id'] not in annotators[a_id]:
                    annotators[a_id][eps['id']] = {
                        k: eps[k] 
                        for k in ['id', 'episode', 'title']
                    }
                    annotators[a_id][eps['id']]['annot_cnt'] = 0
                    annotators[a_id][eps['id']]['annot'] = []
                annotators[a_id][eps['id']]['annot_cnt'] += 1
                annotators[a_id][eps['id']]['annot'].append({
                    'sid': len(annotators[a_id][eps['id']]['annot']),
                    'raw_sid': ant['raw_sid'],
                    'start_time': ant['start_time'],
                    'end_time': ant['end_time'],
                    'script': ant['script'],
                    'characters': [ant['characters'][idx]],
                    'test': [ant['test'][idx]],
                })
                _id += 1

    new_samples = []
    _id = 0
    for k, v in annotators.items():
        for vv in v.values():
            vv['id'] = _id
            new_samples.append(vv)
            _id += 1

    for k, v in annotators.items():
        print(sum(vv['annot_cnt'] for vv in v.values()))

    json_dump(new_samples, 'preprocessed/task1/v01_input_task1.json')