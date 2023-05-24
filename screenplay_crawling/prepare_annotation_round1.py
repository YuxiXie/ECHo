from tqdm import tqdm
from utils import json_load, json_dump

from nltk import sent_tokenize


if __name__ == '__main__':
    data = json_load("raw_data/full_csi_data_100wc.json")
    
    to_annot, idx, word_cnt, char_cnt = [], 0, [], []
    for k, v in tqdm(data.items()):
        samples = []
        for bid, _block in enumerate(v['scripts']):
            sents = [x['text'] for x in _block['content']]
            if len(_block['content']) < 2:
                sents = sent_tokenize(_block['content'][0]['text'])
                if len(sents) > 2:
                    sents = [sents[0], ' '.join(sents[1:-1]), sents[-1]]
                elif len(sents) > 1:
                    sents = [sents[0], sents[-1]]
                else:
                    sents = [sents[0], '']
            
            samples.append({
                'bid': bid,
                'script': '\n'.join([
                    s.strip() for s in sents
                ]),
                'characters': [x['form'] for x in _block['characters']]
            })
            cnt_c = [c['cnt'] for c in _block['characters']]
            if sum(cnt_c) > 0:
                word_cnt.append(_block['word_cnt'])
            char_cnt.append(len(cnt_c))

        to_annot.append({
            'id': idx,
            'episode': k, 
            'title': v['title'],
            'annot_cnt': len(samples),
            'annot': samples
        })
        idx += 1
    
    word_cnt.sort()
    print('- Total number of episodes: {}'.format(len(data)))
    print('- Total number of samples to annotate: {}'.format(len(word_cnt)))
    print('- Avg. word count: {:.2f}'.format(sum(word_cnt) / len(word_cnt)))
    print('- 95% word count: {}'.format(word_cnt[int(len(word_cnt) * 0.95)]))
    print('- 5% word count: {}'.format(word_cnt[int(len(word_cnt) * 0.05)]))
    print('- # > 100: {}'.format(len([x for x in word_cnt if x > 100])))
    print('- # < 20: {}'.format(len([x for x in word_cnt if x < 20])))

    json_dump(to_annot, "preprocessed/full_input_round1.json")
