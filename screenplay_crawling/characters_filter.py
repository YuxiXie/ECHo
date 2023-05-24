import regex
import string
from tqdm import tqdm

from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words += ["'s", "'ll", "'m", "'re", "'t", "n't", "'n", "'ve", "'d"]

from constants import NON_HUMAN_SPEAKERS, IGNORE_SPEAKERS, SPECIAL_CHARS
_NON_HUMAN_SPEAKERS = [x.lower() for x in NON_HUMAN_SPEAKERS]

from utils import json_load, json_dump, sort_dict, load_vocab, dump_vocab
character_list = load_vocab('vocab/character_list.txt')


def bracket_content_1(tokens):
    assert tokens.count('(') <= 1, ' '.join(tokens)
    spans, cur_span = [], []
    for token in tokens:
        if token == '(':
            if len(cur_span):
                spans.append(' '.join(cur_span))
                cur_span = []
            cur_span.append(token)
        elif token == ')':
            cur_span.append(token)
            spans.append(' '.join(cur_span))
            cur_span = []
        else:
            cur_span.append(token)
    if len(cur_span):
        spans.append(' '.join(cur_span))
    
    spans = [s.strip('()').strip() for s in spans]
    return spans


def bracket_content_2(text):
    spans, cur_span = [], []
    for word in text.strip().split():
        if '[' in word:
            if len(cur_span):
                spans.append(' '.join(cur_span))
                cur_span = []
            cur_span.append(word)
        elif ']' in word:
            cur_span.append(word)
            spans.append(' '.join(cur_span))
            cur_span = []
        else:
            cur_span.append(word)
    if len(cur_span):
        spans.append(' '.join(cur_span))
    
    return '\t'.join([s for s in spans if s.count('[') + s.count(']')])


def clean_speaker_txt(speaker, min_cha=3, max_cha=50, double_check=False):

    def _check_valid(txt, txts=[]):
        length_valid = len(txt) >= min_cha and len(txt) <= max_cha
        
        # filter the non-speaker list
        not_valid = any(txt.lower() == x for x in _NON_HUMAN_SPEAKERS + IGNORE_SPEAKERS)
        not_valid = not_valid or any(f' {x} ' in ' {} '.format(txt.lower()) for x in SPECIAL_CHARS)
        not_valid = not_valid or any(x in txts for x in SPECIAL_CHARS)
        # filter non-words
        not_valid = not_valid or regex.match(r'^[^a-zA-Z]$', txt)
        not_valid = not_valid or not regex.search(r'[aeiouy]+', txt.lower())
        not_valid = not_valid or txt.lower().startswith('int.') or ' int.' in txt.lower()
        # filter non-names
        not_valid = not_valid or txt.lower() in stop_words
        
        return length_valid and not not_valid

    def _comma_split(txt):
        txt_list = regex.split(r', ', txt)
        if all(not regex.match(r'^[a-zA-Z]+$', x) \
            and x.lower() not in character_list \
            for x in txt_list):
            txt_list = [txt]
        return txt_list
    
    def _period_split(txt):
        txt_list = regex.split(r'\. ', txt)
        if all(not regex.match(r'^[a-zA-Z]+$', x) \
            and x.lower() not in character_list \
            for x in txt_list):
            txt_list = [txt]
        else:
            for i, s in enumerate(txt_list[:-1]):
                if s.lower() in ['dr', 'det', 'mr', 'mrs', 'prof', 'sgt', 'lt', 'd.a', 'b', 'j.j', 'p.d', 'c', 'd', 'l', 'm']:
                    txt_list[i] = s + '. ' + txt_list[i + 1]
                    txt_list[i + 1] = ''
            txt_list = [s for s in txt_list if s]
        return txt_list

    tokens = word_tokenize(speaker)
    is_valid = _check_valid(speaker, txts=[x.lower() for x in tokens])
    if not is_valid:
        return False, []

    if not double_check and ('/' in speaker or ' & ' in speaker or ' and ' in speaker.lower()) \
        and speaker.lower() not in ['man (light tan suit & tie)']:
        # when the text span contains more than one speakers
        if ' and ' in speaker.lower():
            speaker = regex.split(r' [aA][nN][dD] ', speaker)
        elif regex.match(r'^[a-zA-Z\s\-\./#0-9]+$', speaker):
            speaker = speaker.split('/')
        elif regex.match(r'^[a-zA-Z\s\-\.&#0-9]+$', speaker):
            speaker = speaker.split(' & ')
        elif regex.search(r'[\(\)]+', speaker):
            if '/' in speaker:
                spans = [x for x in bracket_content_1(tokens) if x.count('/')]
                speaker = spans[0].split('/')
            else:
                spans = [x for x in bracket_content_1(tokens) if x.count('&')]
                speaker = spans[0].split(' & ')
        else:
            return False, []
    elif double_check:
        # specific processing on those from narrations
        if ']' in speaker:
            speaker = speaker.split(']')[-1].strip()
        if speaker.strip() and len(speaker.split()[-1]) > 3:
            speaker = speaker.rstrip('.,;:')
        if speaker.lower().endswith("'s"):
            speaker = speaker[:-2]
        
        if speaker.lower() in character_list:
            speaker = [speaker]
        else:
            speaker = speaker.strip(string.punctuation)
            speaker_list = []
            if regex.match(r'^[a-zA-Z]+$', speaker):
                speaker_list = [speaker]
            elif regex.match(r'^[a-zA-Z\s]+/[a-zA-Z\s]+$', speaker):
                # w1/w2
                speaker_list = regex.split(r'/', speaker)
            elif ' and ' in speaker.lower():
                speaker_list = regex.split(r' [aA][nN][dD] ', speaker)
            elif regex.match(r'^[a-zA-Z\s,]+$', speaker):
                # w1, w2
                speaker_list = _comma_split(speaker)
            elif regex.match(r'^[a-zA-Z\s\.]+$', speaker):
                # w1. w2
                speaker_list = _period_split(speaker)
            elif regex.match(r'^[a-zA-Z\s\.,]+$', speaker):
                speaker_list = _comma_split(speaker)
            elif '. ' in speaker:
                speaker_list = _period_split(speaker)
            else:
                speaker_list = [speaker]
            
            speaker = [s for s in speaker_list if s.lower() in character_list]
    else:
        speaker = [speaker.strip(',;:')]

    speakers = []
    for s in speaker:
        s = s.strip()
        if not len(s): continue

        if len(s.split()[-1]) > 3: s = s.rstrip('.')
        
        if _check_valid(s): speakers += [s]
    
    return is_valid, speakers


def get_characters(_block):
    characters = {}
    for line in _block:
        if line['type'] == 'utterance':
            speaker = line['text'].split(']]')[0].split('[[')[-1]
            # filtering
            is_valid, speakers = clean_speaker_txt(speaker)
            if not is_valid: continue
            # add characters
            for speaker in speakers:
                if speaker.lower() not in characters:
                    characters[speaker.lower()] = {'cnt': 0, 'form': speaker}
                characters[speaker.lower()]['cnt'] += 1
        else:
            capitalized = bracket_content_2(line['text'])
            tokens = line['text'].lstrip('([').rstrip(')]').strip().split()
            i = 0
            while i < len(tokens):
                # find the capitalized text-spans
                speaker = []
                for j in range(i, len(tokens)):
                    if tokens[j].isupper(): speaker.append(tokens[j])
                    else: break
                if len(speaker) == 0 or ' '.join(speaker) in capitalized: i += 1
                else:
                    i += len(speaker)
                    speaker = ' '.join(speaker)
                    # filtering
                    is_valid, speakers = clean_speaker_txt(speaker, double_check=True)
                    if not is_valid: continue
                    # add characters
                    for speaker in speakers:
                        if speaker.lower() not in characters:
                            characters[speaker.lower()] = {'cnt': 0, 'form': speaker}
                        characters[speaker.lower()]['cnt'] += 1
    return sort_dict(characters)


if __name__ == '__main__':
    DATA_PATH = "raw_data/full_csi_data_100wc.json"
    data = json_load(DATA_PATH)

    for k, v in tqdm(data.items()):
        for bid, _block in enumerate(v['scripts']):
            characters = get_characters(_block['content'])
            data[k]['scripts'][bid]['characters'] = list(characters.values())

    json_dump(data, DATA_PATH)