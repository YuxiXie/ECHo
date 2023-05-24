import os
import regex
import string

from tqdm import tqdm
from lxml import etree
from statistics import stdev

from utils import json_dump
from constants import START_TERMS, NON_HUMAN_SPEAKERS


OUTFILENAME = "raw_data/full_csi_data_100wc.json"

def _is_others(sent):
    is_equal = sent.count('[') == sent.count(']')
    is_others = is_equal and regex.match(r'^\[.+\]$', sent)

    return is_others

def _is_complete_narration(sent, contain_others=False):
    is_equal = sent.count('(') == sent.count(')') and len(sent) > 2
    is_complete = is_equal and regex.match(r'^\(.+\)$', sent)
    
    return is_complete or (contain_others and _is_others(sent))

def _is_utterance(sent, max_s=10, max_c=20):
    if ':' in sent:
        _sent = sent.split(':')
        speaker, content = _sent[0], ':'.join(_sent[1:])
    elif ';' in sent:
        _sent = sent.split(';')
        speaker, content = _sent[0], ';'.join(_sent[1:])
    else:
        speaker, idx = [], 0
        for i, t in enumerate(sent.split()):
            if t.isupper() and t not in ['I', 'A']:
                speaker += [t]; idx = i + 1
            else:
                break
        speaker = ' '.join(speaker)
        content = ' '.join(sent.split()[idx:]) if idx < len(sent.split()) else ''
    
    speaker, content = speaker.strip(), content.strip()
    is_speaker = len(speaker.split()) < max_s and len(speaker) > 2 and len(speaker) < max_c \
        and all(x[0].isupper() for x in speaker.split('(')[0].strip().split()) \
        and speaker not in NON_HUMAN_SPEAKERS and not regex.match(r'^[\(\[\{]', speaker) \
        and '[[' not in speaker and '[[' not in speaker
    is_content = len(content.split()) * len(content) > 0

    is_uttr = is_speaker and is_content
    return is_uttr, speaker, content


def parse_html(text, _id):
    '''
    Parse the html file to get the transcripts of each episode
    PS: blocks are divided by <hr/>
    '''
    html = etree.fromstring(text, parser=etree.HTMLParser())
    
    # get the episode's title
    title_line = html.xpath('//meta[@property="og:title"]')[0].get('content')
    title = title_line.split(' - CSI: Crime Scene Investigation Transcripts - ')[0].strip()
    title = title.split(' - ')[-1].strip()
    
    # get the main content in "postbody"
    content = html.xpath('//div[@class="postbody"]')
    if len(content) != 1:
        import ipdb; ipdb.set_trace()
    
    blocks, cur_block = [], []
    for child in content[0].getchildren():
        if child.tag == 'p':    
            # lines of transcripts are contained in <p>
            text = [txt for txt in child.itertext()]

            # start a new block
            if ' '.join(text).startswith('CUT TO:') or ' '.join(text).startswith('(COMMERCIAL BREAK)'):
                if len(cur_block) > 0:
                    blocks.append(cur_block)
                    cur_block = []
                continue

            if not len(''.join(text).strip()): continue     # skip the space    
            text_dict = [{'text': txt, 'tag': 'none'} for txt in text]
        
            grandchildren = child.getchildren()
            # prepare for <br/> replacing
            br_splits = []  
            for x in bytes.decode(etree.tostring(child))[:-1].split('<br/>'):
                x = regex.sub(r'<[/a-zA-Z]+>', '', x)
                xx = [t for t in etree.HTML(x).itertext()]
                if len(xx) > 1: import ipdb; ipdb.set_trace()
                xx = xx[0]
                for t in x:
                    if t == '\n': xx = '\n' + xx
                    else: break
                br_splits.append(xx)
            # add grandchildren's tags
            tmp_idx, br_idx = 0, 0
            for grandchild in grandchildren:
                if grandchild.text:
                    cur_idx = text[tmp_idx:].index(grandchild.text) + tmp_idx
                    text_dict[cur_idx]['tag'] = grandchild.tag
                    tmp_idx = cur_idx + 1
                else:
                    assert grandchild.tag == 'br', "we only consider this tag"
                    # locate and add the '\n' text
                    prev, nxt = br_splits[br_idx], br_splits[br_idx + 1]
                    for i, txt in enumerate(text[tmp_idx:]):
                        cur_idx = i + tmp_idx
                        if prev and prev.endswith(txt) and \
                            (not nxt or nxt.startswith(text[cur_idx + 1])):
                            text_dict[cur_idx]['text'] += '\n'
                            break
                        elif nxt and nxt.startswith(txt) and \
                            (not prev or prev.endswith(text[cur_idx - 1])):
                            text_dict[cur_idx]['text'] = '\n' + txt
                            break
                    tmp_idx = cur_idx + 1
                    br_idx += 1
            
            # append the lines
            line = []
            for i, txt in enumerate(text_dict):
                if txt['tag'] == 'strong':                    
                    if i != 0 and not regex.match(r'\s', line[-1][-1]) and not line[-1].endswith('WO'):
                        if line[-1].endswith('/'):  # merge all speakers
                            line[-1] = '[[{}'.format(line[-1])
                            txt['text'] = '{}]]'.format(txt['text'])
                    else:
                        txt['text'] = '[[{}]]'.format(txt['text'])
                line.append(txt['text'])
            if len(line) > 0: 
                cur_block.append(line)
        elif child.tag == 'hr':
            # start a new block
            if len(cur_block) > 2:
                blocks.append(cur_block)
                cur_block = []
    if len(cur_block) > 0:
        blocks.append(cur_block)

    return title, blocks


def assign_block(to_assign, n_blocks=1):
    '''
    DP Algorithm to Merge N Lines of Scripts as M Blocks 
    with the Lowest Variance of word-cnt
    
    return: list of indexes for all blocks
    '''
    N, M = len(to_assign), n_blocks
    D = [x['cnt'] for x in to_assign]

    if M == 1: 
        return [[i for i in range(N)]]

    # P[(n, i)] - the split point (index) the split lines[: i + 1] into n blocks
    P = {}  
    for i in range(1, N):
        V_list = [(j, stdev([sum(D[: j + 1]), sum(D[j + 1: i + 1])])) for j in range(i)]
        V_list.sort(key=lambda x: x[1])
        P[(2, i)] = V_list[0][0]
    
    if M == 2: 
        return [[i for i in range(P[2, N - 1] + 1)], 
                [i for i in range(P[2, N - 1] + 1, N)]]
    
    for n in range(3, M + 1):
        for i in range(n - 1, N):
            V_list = []
            for j in range(n - 2, i):
                s, e = j, i
                cnt_list = [sum(D[s + 1: e + 1])]
                for k in range(1, n - 1):
                    e = s
                    s = P[(n - k, s)]
                    cnt_list += [sum(D[s + 1: e + 1])]
                cnt_list += [sum(D[: s + 1])]
                V_list.append((j, stdev(cnt_list)))
            V_list.sort(key=lambda x: x[1])
            P[(n, i)] = V_list[0][0]
    
    paths, s = [N - 1], N - 1
    for k in range(M - 1):
        s = P[(n - k, s)]
        paths.append(s)
    paths.append(-1)
    paths.sort()

    return [[i for i in range(paths[idx] + 1, paths[idx + 1] + 1)] for idx in range(M)]


def process_block(_block, bid, max_word_cnt=200):
    '''
    Process Blocks of Transcripts:
    1) Filter - remove meaningless lines;
    2) Merge - merge lines which are related;
    3) Categorize - label the type of text for each line;
    4) Split - split the blocks which are too long
    '''
    ##=== Filter and Merge Sentences ===##
    lines, cur_line = [], ''

    def _update_lines(sent, cur_line):
        if cur_line:
            lines.append(cur_line)
        cur_line = sent
        return lines, cur_line

    for line in _block:
        line = ''.join(line).strip()    # join to text
        if line.lower() == 'end' or line.isupper(): continue      # filter the meaningless contents
        
        line = [x.strip() for x in line.split('\n') if x.strip()]

        for sent in line:
            # filter the lines of movie terminology
            if not sent.startswith('[['):
                if sent.startswith('[') and sent.endswith(']') and sent.count('[') == sent.count(']'):
                    continue
                if any(sent.lstrip('([').lower().startswith(x) for x in START_TERMS):
                    tmp_sent = ':'.join(sent.split(':')[1:]).strip()
                    if not tmp_sent or any(tmp_sent.lstrip('([').lower().startswith(x) for x in START_TERMS):
                        continue
                    # leave the available descriptions/narrations
                    _to_add = ''
                    for t in sent:
                        if t == '(': _to_add += '('
                        elif t == '[': _to_add += '['
                        else: break
                    sent = _to_add + tmp_sent
            if len(sent.split()) == 1 and len(line) == 1:
                continue    # filter those which are too short
            
            # merge the sentences to update the lines
            if sent.startswith('[[') or _is_complete_narration(sent):
                lines, cur_line = _update_lines(sent, cur_line)
            else:
                is_uttr, speaker, content = _is_utterance(sent)
                if not regex.match(r'^[\(\[\{]', sent) and is_uttr and \
                    not (speaker.startswith('[') or speaker.endswith(']')):
                    sent = '[[{}]]: {}'.format(speaker, content)
                    lines, cur_line = _update_lines(sent, cur_line)
                elif _is_complete_narration(cur_line) or len((cur_line + f' {sent}').strip().split()) > max_word_cnt: 
                    lines, cur_line = _update_lines(sent, cur_line)
                else:
                    cur_line += f' {sent}'
    lines, cur_line = _update_lines('', cur_line)

    ##=== Categorize Lines with Tags ===##
    lines_dict, uncomplete = [], False

    def _update_uncomplete(sent):
        uncomplete = sent.count('(') > sent.count(')')
        return uncomplete

    for line in lines:
        if not line.startswith('[[') and (line.isupper() or regex.match(r'^\[.*\]$', line) or \
            any(line.lstrip(string.punctuation).lower().startswith(x) for x in START_TERMS)): 
            continue    # filter the meaningless ones
        # label the type tags
        if line.startswith('[['):
            lines_dict.append({'type': 'utterance', 'text': line})
        elif line.startswith('(') and (line.count('(') > line.count(')') or line.endswith(')')):
            if uncomplete and len(lines_dict) and lines_dict[-1]['type'] == 'narration' and \
                not (_is_complete_narration(line) and len((lines_dict[-1]['text'] + f' {line}').split()) > 30):
                lines_dict[-1]['text'] += f' {line}'
            else:
                lines_dict.append({'type': 'narration', 'text': line})
            uncomplete = _update_uncomplete(lines_dict[-1]['text'])
        else:
            is_uttr, speaker, content = _is_utterance(line)
            if is_uttr and not (speaker.startswith('[')) or speaker.endswith(']'):
                line = '[[{}]] {}'.format(speaker, content)
                lines_dict.append({'type': 'utterance', 'text': line})
            else:
                lines_dict.append({'type': 'others', 'text': line})
    
    if len(lines_dict) == 0:
        return []
    
    ##=== Split the Blocks which are Too Long ===##
    text = ' '.join([x['text'] for x in lines_dict])
    total_word_cnt = len(text.split())
    n_blocks = max(1, (total_word_cnt + max_word_cnt - 1) // max_word_cnt)
    to_assign = [{'type': x['type'], 'cnt': len(x['text'].split())} for x in lines_dict]

    indexes = assign_block(to_assign, n_blocks=n_blocks)

    blocks = []
    for idx in indexes:
        blocks.append({
            'word_cnt': sum(to_assign[i]['cnt'] for i in idx), 
            'content': [lines_dict[i] for i in idx],
            'raw_bid': bid, 
        })

    return blocks


if __name__ == '__main__':    
    ##=== Load and filter the transcripts ===##
    g = os.walk('raw_data/html_data')
    corpus = {}
    for path, _, file_list in g:  
        for filename in tqdm(file_list):
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
                content = f.read()

            e_id = filename.split('.')[0]
            title, content = parse_html(content, e_id)

            if len(content) > 1:    # filter those which don't contain narrations
                corpus[e_id] = {'title': title, 'content': content}

    ##=== Filter and save contents ===##
    max_word_cnt = int(OUTFILENAME.split('.')[0].split('_')[-1].strip('wc'))

    updated_corpus = {}
    for e_id, episode in tqdm(corpus.items()):
        title, blocks = episode['title'], episode['content']
        
        processed_blocks = [process_block(_block, bid, max_word_cnt=max_word_cnt) for bid, _block in enumerate(blocks)]
        processed_blocks = [xx for x in processed_blocks for xx in x]

        updated_corpus[e_id] = {
            'title': title,
            'n_blocks': len(processed_blocks),
            'scripts': processed_blocks,
        }

    json_dump(updated_corpus, OUTFILENAME)
