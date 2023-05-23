import collections
import random
import string
from tqdm import tqdm
from ipdb import set_trace

from utils import json_load, json_dump, jsonlines_load, jsonlines_dump

from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords

random.seed(100)

stop_words = stopwords.words('english')

AGES = {
    '1': (0, 'Infant or Child (<12)'),
    '2': (1, 'Child, Teenager or Youth (10~25)'),
    '3': (2, 'Youth or Middle-Age (25~55)'),
    '4': (3, 'Middle/Old-Age (>50)'),
    '0': (4, 'N.A.'),
}

EMOS = {
    '1': (0, 'Anger'), '2': (1, 'Boredom'), '3': (2, 'Calmness'),
    '4': (3, 'Disgust'), '5': (4, 'Doubt'), '6': (5, 'Entrancement'),
    '7': (6, 'Fear'), '8': (7, 'Interest'), '9': (8, 'Joy'),
    '10': (9, 'Sadness'), '11': (10, 'Shame'), '12': (11, 'Surprise'), 
    '13': (12, 'Sympathy'),
}

def my_word_tokenize(text):
    raw_tokens = word_tokenize(text)
    tokens, pair = [], ''
    for t in raw_tokens:
        if not pair:
            pair += t
        elif t == '.' and pair.lower() in ['det', 'dr', 'mrs', 'miss', 'ms', 'mr', 'prof', 'sgt', 'lt', 'b', 'd.a', 'n.d', 'c', 'j.j', 'm', 'noc', 'p.d', 'psych', 'r']:
            tokens.append(f'{pair}.')
            pair = ''
        else:
            tokens += [pair, t]
            pair = ''
    if pair: tokens.append(pair)
    return tokens

def get_time(s, e):
    def _time(x):
        x = x.split(':')
        return int(x[0]) * 60 + float(x[1])
    
    return _time(e) - _time(s)

def word_overlap(a, s):
    aa = set([t.lower() for t in word_tokenize(a)])
    ss = [t.lower() for t in word_tokenize(s)]

    ratio = [0, 0]
    overlap = []
    for t in aa:
        if t in stop_words: continue
        if t in string.punctuation: continue
        ratio[1] += 1
        if t.lower() in ss:
            ratio[0] += 1
            overlap.append(t)
    
    return ratio[0] / max(1, ratio[1])

def check_annotator_progress(annots):
    users = collections.defaultdict(int)
    for annot in annots:
        users[annot['user']] += 1
    cnt = [(k, v) for k, v in users.items()]
    cnt.sort(key=lambda x: x[1])
    min_num, max_num = 100, 180
    idx1 = [i for i,x in enumerate(cnt) if x[1] >= min_num][0]
    idx2 = [i for i,x in enumerate(cnt) if x[1] >= max_num][0]
    print('* {} annotators in total'.format(len(users)))
    print('* {} annotators complete all: {}'.format(len(users) - idx2, [x[0] for x in cnt[idx2:]]))
    print('* {} annotators complete min.: {}'.format(idx2 - idx1, [x[0] for x in cnt[idx1:idx2]]))
    print('* {} annotators unfinished yet: {}'.format(idx1, [x for x in cnt[:idx1]]))
    return users

def get_sample_dict(samples):
    sample_dict, _id = {}, 0
    for s in tqdm(samples):
        for a in s['annot']:
            for i in range(len(a['test'])):
                sample_dict[_id] = {
                    'id': s['id'],
                    'episode': s['episode'],
                    'title': s['title'],
                    'annot_cnt': s['annot_cnt'],
                    'annot': {
                        'sid': a['sid'],
                        'raw_sid': a['raw_sid'],
                        'start_time': a['start_time'],
                        'end_time': a['end_time'],
                        'script': a['script'],
                        'character': a['characters'][i],
                        'test': a['test'][i],
                    }
                }
                _id += 1
    return sample_dict

def merge_input_output(sample_dict, annotations):
    merged_dict, frame_error = {}, 0
    for annot in tqdm(annotations):
        if len(annot['frames'][0]) <= int(annot['frames'][1][-1]):
            frame_error += 1
        user = annot['user']
        if user == 'yuxi': continue
        _id = annot['ID']
        sample = sample_dict[annot['ID']]
        
        frames = annot['frames']
        properties = annot['properties']
        emotions = [(EMOS[x], int(x)) for x in properties[2][:-1] if x != '0']
        if '0' in properties[2][:-1]:
            emotions.append(((13, properties[2][-1]), 0))
        
        if _id not in merged_dict:
            merged_dict[_id] = {
                'id': _id,
                'episode': sample['episode'],
                'title': sample['title'],
                'annot_cnt': 0,
                'annot': [],
            }
        merged_dict[_id]['annot_cnt'] += 1
        merged_dict[_id]['annot'].append({
            k: v for k, v in sample['annot'].items()
        })
        merged_dict[_id]['annot'][-1].update({
            'annotator': user,
            'age': (AGES[properties[0]], int(properties[0])),
            'role': properties[1],
            'emotions': emotions,
            'frames': tuple(':'.join(x) for x in frames[0] if x[0]),
            'checked_frames': tuple(int(x) for x in frames[1]),
            'causality': properties[-1],
        })
    return merged_dict, frame_error

def load_progress(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        progress = f.read().strip().split('\n')
    status = {}
    for line in progress:
        line = line.strip().rstrip(',').split(',')
        user = line[2].strip().lower().replace(' ', '')
        status[user] = {
            'before': int(line[1].strip()) if line[1].strip() else -1,
            'current': int(line[0].strip()) if line[0].strip() else -1,
        }
    return status

def pre_check_to_reject(merged_dict, cur_status):
    status = load_progress('preprocessed/task1/mid_correct.txt')
    
    Q5_tuples = collections.defaultdict(list)   # question & answer
    overlap, ans_lens = [], []
    frm_cnt, dur = [[], []], []   # frames & duration
    
    to_record = jsonlines_load('preprocessed/task1/mid_correct.json')
    to_record = {x['annotator']:x for x in to_record}
    
    for _id, sample in merged_dict.items():
        merged_dict[_id]['annot'].sort(key=lambda x: (x['start_time'], x['end_time']))
        sample['annot'].sort(key=lambda x: (x['start_time'], x['end_time']))
        for annot in sample['annot']:
            # Q5 progress
            annotator = annot['annotator']
            progress = status[annotator]
            progress['current'] = cur_status[annotator]
            if progress['before'] > -2:
                season = int(sample['episode'].strip('S').split('E')[0])
                Q5_tuples[annotator].append({
                    'data': (
                        'effect' if 'caused/enabled by' in annot['test'][0] else 'cause',
                        annot['test'][1],
                        annot['causality'],
                        annot['test'][2],
                    ),
                    'script': annot['script'],
                    'video': 'https://csivideosdata.s3.ap-southeast-1.amazonaws.com/season{}/{}.mp4'.format(season, sample['episode']),
                })
            overlap.append(word_overlap(annot['causality'], annot['script']))
            ans_lens.append((len(word_tokenize(annot['causality'])), annot['causality']))
            
            dur.append(get_time(annot['start_time'], annot['end_time']))
            frm_cnt[0].append(len(annot['frames']))
            frm_cnt[1].append(len(annot['checked_frames']))
    
    for k, v in Q5_tuples.items():
        if k in to_record: 
            to_record[k]['current'] = max(status[k]['current'], to_record[k]['current'])
            d = {
                'annotator': k,
                'paid': 0,
                'checked': True,
            }
            d.update(to_record[k])
            to_record[k] = d
            continue
        to_record[k] = {
            'annotator': k,
            'paid': 0,
            'checked': False,
            'current': status[k]['current'],
            'before': status[k]['before'] if status[k]['before'] > 0 else '',
        }
        # all(vv['data'][2].lower().startswith(vv['data'][3].lower()) for vv in v)
        to_record[k]['start_txt'] = 'Y' if all(vv['data'][2].lower().startswith(vv['data'][3].lower()) for vv in v) else '+'
        to_record[k]['all_cause'] = 'Y'
        to_record[k]['PS'] = '-1,-1'
        print(k, len(v))
    
    # import ipdb; ipdb.set_trace()
    
    jsonlines_dump(list(to_record.values()), 'preprocessed/task1/mid_correct.json')

def check_to_reject(merged_dict):
    pre_cause = "What probably has caused/enabled the event that"
    pre_effect = "What probably will be the effect/result of the event that"
    
    status = jsonlines_load('preprocessed/task1/mid_correct.json')
    status = {x['annotator']:x for x in status}
    
    record_cnt = {k:0 for k in status}
    for _id, sample in merged_dict.items():
        sample['annot'].sort(key=lambda x: (x['start_time'], x['end_time']))
        for annot in sample['annot']:
            annotator = annot['annotator']
            progress = status[annotator]
            
            ## cause / effect
            all_cause = (progress['all_cause'], progress['PS'].split(',')[1:])
            threshold = int(all_cause[1][0].rstrip(string.punctuation))
            Q, A = annot['test'], annot['causality'].strip()
            Q[0] = pre_cause if Q[0] == "What probably caused/enabled the event that" else pre_effect
            if all_cause[0].count('Y'):
                if threshold < 0 or record_cnt[annotator] <= threshold:
                    Q[0] = pre_cause
            
            ## Q5 answer
            start_txt = (progress['start_txt'], progress['PS'].split(',')[0])
            threshold = int(start_txt[1].rstrip(string.punctuation))
            if start_txt[0] == '+':
                if threshold < 0 or record_cnt[annotator] <= threshold:
                    A = f'{Q[2]} {A}' if not A.startswith(Q[2]) else A
            elif start_txt[0].count('pronoun'):
                tokens = my_word_tokenize(A)
                tags = pos_tag(tokens)
                pronouns = [(idx, pair) for idx, pair in enumerate(tags) if pair[1] == 'PRP']
                if pronouns and not pronouns[0][0]:
                    prp_dict = collections.defaultdict(list)
                    for idx, pair in pronouns:
                        pair = (pair[0].lower(), pair[1])
                        prp_dict[pair] += [idx]
                    # if len(prp_dict) != 1:
                    #     set_trace()
                    for i in list(prp_dict.values())[0]:
                        tokens[i] = annot['character']    # Q[2]
                    A = ' '.join(tokens)
            if A[0].islower():
                A = A[0].upper() + A[1:]
            
            annot['causality'] = A
            annot['test'] = Q
            record_cnt[annotator] += 1

if __name__ == '__main__':
    annotations = jsonlines_load('preprocessed/task1/v01_output_task1.json')
    # to check the progress of each annotator
    cur_status = check_annotator_progress(annotations)
    
    samples = json_load('preprocessed/task1/v01_input_task1.json')
    # to index each annotation sample in samples by ID
    sample_dict = get_sample_dict(samples)
    
    # merge the input and output of annotation
    merged_dict, frame_error = merge_input_output(sample_dict, annotations)

    # sort the annotations, and check:
    # 1) overlap rate
    # 2) Q5 answer length
    # 3) #frames v.s. #checked-frames
    # 4) durations of clips
    check_to_reject(merged_dict)
    
    merged_lst = list(merged_dict.values())
    merged_lst.sort(key=lambda x: (x['episode'], x['annot'][0]['start_time']))
    
    set_trace()
    
    json_dump(merged_lst, 'preprocessed/task1/v01_merged_task1.json')
    