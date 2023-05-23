from tqdm import tqdm
from math import floor, ceil

from utils import json_load, json_dump, jsonlines_load, draw_scatter


def calculate_time(times):
    _time = float(times[0]) * 60 + float(times[1])
    return _time


def same_char_rate(chars1, chars2):
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


if __name__ == '__main__':
    raw = json_load('raw_data/full_csi_data_100wc.json')

    ##=== no. of blocks in pilot ===##
    cnt = [0, 0]
    for k, v in raw.items():
        if 'S01' not in k: continue
        if int(k[-2:]) > 15: continue
        cnt[1] += 1
        cnt[0] += len(v['scripts'])
    print('{} blocks / {} episodes'.format(cnt[0], cnt[1]))

    task0_out = jsonlines_load('preprocessed/task0/full_output_task0.json')
    episodes = list(raw.keys())

    ##=== merge the annotation results ===##
    labeled, available_chars = [], []
    for annot in tqdm(task0_out):
        i, j = annot['ID'], annot['cur_id']
        labeled.append(episodes[i])

        # update the selected character list
        _chas = raw[episodes[i]]['scripts'][j].get('selected_characters', [])
        _chas += list(set(annot['selected_characters']))
        raw[episodes[i]]['scripts'][j]['selected_characters'] = _chas

        # update the timestamps of each clip
        s = raw[episodes[i]]['scripts'][j].get('start_time', 10000)
        e = raw[episodes[i]]['scripts'][j].get('end_time', -1)
        ss = floor(calculate_time(annot['start_time']))
        ee = ceil(calculate_time(annot['end_time']))
        raw[episodes[i]]['scripts'][j]['start_time'] = min(ss, s)
        raw[episodes[i]]['scripts'][j]['end_time'] = max(ee, e)

        if abs(s - ss) > 5 and s < 10000:
            import ipdb; ipdb.set_trace()
        if abs(e - ee) > 5 and e >= 0:
            import ipdb; ipdb.set_trace()

        # update the data list
        tmp_episode = {'episode': episodes[i]}
        tmp_episode.update(raw[episodes[i]])
        raw[episodes[i]] = tmp_episode

        # count the availbale characters
        char_dict = {x['form']: x['cnt'] for x in raw[episodes[i]]['scripts'][j]['characters']}
        for c in annot['selected_characters']:
            if c in char_dict:
                available_chars.append(char_dict[c])
            else:
                available_chars.append(0)
    
    ##=== only pick the annotated samples ===##
    labeled = list(set(labeled))
    labeled.sort()
    samples = [raw[k] for k in labeled]

    ##=== generate new combinations of blocks ===##
    cnt = [0, 15]
    for i, sample in enumerate(samples):
        # get the adjacent block pairs which share most of characters
        same_set = []
        for j, script in enumerate(sample['scripts'][1:]):
            prev_script = sample['scripts'][j]

            _same_char = same_char_rate(script['selected_characters'], prev_script['selected_characters'])
            if _same_char != 1: continue
            # elif _same_char > 0 and _same_char < 0.75:
            #     import ipdb; ipdb.set_trace()
            # if sum([script['word_cnt'], prev_script['word_cnt']]) > 150:
            #     continue
            # if script['end_time'] - prev_script['start_time'] > 90:
            #     continue
            same_set.append([j, j + 1])
        # merge pairs into sets
        for k, pair in enumerate(same_set):
            for kk in range(k):
                if len(same_set[kk]) > 2: continue
                if any(x in same_set[kk] for x in pair):
                    same_set[kk] += [x for x in pair if x not in same_set[kk]]
                    same_set[kk].sort()
                    same_set[k] = []
                    break
        # generate new blocks based on the sets
        for _set in same_set:
            if not _set: continue
            word_cnt, content = 0, []
            characters = {}
            selected_characters = []
            start_time, end_time = 10000, -1
            for k in _set:
                script = sample['scripts'][k]
                word_cnt += script['word_cnt']
                content += script['content']
                for c in script['characters']:
                    if c['form'] in characters:
                        characters[c['form']]['cnt'] += c['cnt']
                    else:
                        characters[c['form']] = c
                selected_characters += [x.upper() for x in script['selected_characters'] if x.upper() not in selected_characters]
                start_time = min(start_time, script['start_time'])
                end_time = max(end_time, script['end_time'])
            if end_time - start_time > 100: continue
            if len(selected_characters) > 5: continue
            if word_cnt > 150: continue
            sample['scripts'].append({
                'word_cnt': word_cnt,
                'content': content,
                'raw_bid': -1,
                'characters': list(characters.values()),
                'selected_characters': selected_characters,
                'start_time': start_time,
                'end_time': end_time,
            })
        samples[i] = sample
        cnt[0] += len(sample['scripts'])

    print('{} blocks / {} episodes'.format(cnt[0], cnt[1]))

    import ipdb; ipdb.set_trace()
    json_dump(samples, 'preprocessed/v1_csi_corpus.json')