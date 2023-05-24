import random
import openai
import jsonlines

from tqdm import tqdm
from time import sleep, time
from utils import json_load, json_dump, jsonlines_load, jsonlines_dump
from tenacity import wait_random_exponential, stop_after_attempt, retry


## path to the input .jsonl file (each line is an instance)
INPUTFILE = ''
OUTPUTFILE = ''

USE_CHATGPT = False
PROMPT_BATCH_SIZE = 8   # PS: ChatGPT only accepts batch-size = 1

INPUTFILE_CAP = {'R': '', 'E': ''}
TOM_TYPE = ''   # blip , / llm
INPUTFILE_TOM = {'R': '', 'E': ''}


with open('./keys/my_keys.txt', 'r', encoding='utf-8') as f:
    text = f.read().strip().split('\n')
KEYS = [line.strip() for line in text]

num_key = len(KEYS)
print(f'({num_key} keys in total).')
KEYS_DICT = {k: None for k in KEYS}

ERRORS = [
    'Rate limit reached for default-code-davinci-002', 
    'Request timed out'
]


def get_internal(keys_used, k, cur_time):
    t = 6e2    # TODO: magic number
    if keys_used[k] is not None:
        t = cur_time - keys_used[k]
    return t

def select_key(keys_used, keys, all_keys=None):
    if all_keys is not None and random.random() < 0.05:     # TODO: magic number
        newkeys = [(k, get_internal(keys_used, k, time())) for k in all_keys]
        newkeys = [k[0] for k in newkeys if k[1] > 6e2]    # TODO: magic number
        if len(newkeys):
            keys = list(set(keys + random.sample(newkeys, min(10, len(newkeys)))))    # TODO: magic number
    
    key_and_time = [(k, get_internal(keys_used, k, time())) for k in keys]
    _key = sorted(key_and_time, key=lambda x:-x[1])[0][0]
    keys_used[_key] = time()
    return _key

@retry(wait=wait_random_exponential(min=5, max=1000), stop=stop_after_attempt(128))
def _generate_prompt(prompt, max_tokens=128, temperature=0.0, 
                     top_p=1, n=1, logprobs=1, key=[], use_chatgpt=True):
    _prompt = prompt[0] if use_chatgpt else prompt
    
    st = time()
    if use_chatgpt:
        rst = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=_prompt,
            api_key=select_key(KEYS_DICT, key, all_keys=KEYS),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stop=['\n\n'],
        )
    else:
        rst = openai.Completion.create(
            engine='text-davinci-003',
            prompt=_prompt,
            api_key=select_key(KEYS_DICT, key, all_keys=KEYS),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stop=['\n\n'],
            logprobs=logprobs,
        )
    dur = time() - st
    print(f'@_generate_prompt: {dur} seconds')
    return [rst] if use_chatgpt else rst

def _openai_prompt_batch(prompt, max_tokens=256, temperature=0.0, top_p=1, n=1, logprobs=1, 
                        key=[], min_to_wait=3, max_to_wait=10, sleep_time=5, use_chatgpt=True):
    if use_chatgpt:
        prompt = [[
            {'role': 'system', 'content': 'You are a helpful assistant that answer questions of human-centric reasoning, where the human-centric information is elicited as Theory of Mind.'},
            {'role': 'user', 'content': p},
        ] for p in prompt]
        assert len(prompt) == 1
    else:
        prompt = [f'{p}\n\nQuestion: ' for p in prompt]
    
    start_time = time()
    result, cost = None, sleep_time
    
    try:
        result = _generate_prompt(prompt, max_tokens=max_tokens, temperature=temperature, 
                                  top_p=top_p, n=n, logprobs=logprobs, key=key, use_chatgpt=use_chatgpt)
        cost = time() - start_time
    except Exception as e:
        print(f'***code API error***', str(e))
        sleep(sleep_time)
    
    dur = time() - start_time
    to_wait = sleep_time
    if to_wait < min_to_wait: to_wait = random.uniform(to_wait, max_to_wait) 
    _sleep_time = to_wait - int(dur) if dur < to_wait - min_to_wait \
        else random.uniform(min_to_wait, max(min_to_wait + 1, sleep_time))
    print(f'@generation: {cost} seconds (+ sleep {_sleep_time} seconds)')
    sleep(_sleep_time)
    
    return result

def _construct_prompt(script, character, role, emotion, event1, desc):    # add the line of discriptions
    template = "Read the subscript of a crime-drama plot:\n{}\n\n" \
        + "For reference, there are textual descriptions for a squence of keyframes of {} in the plot video:\n{}\n\n" \
        + "About the identity of {}, we know that:\n{}\n\nAbout the emotional traits of {}, we know that:\n{}" \
        + "\n\nQuestion: {}\nAnswer: {} "

    return template.format(script, character, desc,
                           character, role, character, emotion, 
                           ' '.join(event1[:-1]) + '?', event1[-1])


def _get_desc(line, fid_list):
    captions = {x: [] for x in fid_list}
    for x in fid_list:
        desc = line[f'{x}-desc'].strip().split(' / ')
        add = line[f'{x}']
        captions[x] = list(set(desc + add))
    to_return = {}
    for i, v in enumerate(captions.values()): to_return[f'Frame {i + 1}'] = v
    return to_return


def get_data(raw_data, ToM_dict, txt_descriptions):
    prompts = []
    for dt in tqdm(raw_data):
        idx = (dt['ID'], dt['aid'], dt['character'],)
        fid_list = [str(frm['fid']) for frm in dt['frames'] if frm['checked']]
        frm_id = idx + ('-'.join(fid_list),)
        role = ToM_dict['role'][frm_id]
        emotion = ToM_dict['emotion'][frm_id]
        raw_desc = (txt_descriptions['role'][frm_id], txt_descriptions['emotion'][frm_id])
        desc = {k:[] for k in raw_desc[0]}
        for k, vr in raw_desc[0].items():
            ve = raw_desc[1][k]
            desc[k] = '; '.join(list(set(vr + ve)))
        # prompt = _construct_prompt(dt['script'], dt['character'], role, emotion, dt['event1'],
        #                            '\n'.join([f'{k}: {v}' for k, v in desc.items()]))
        role = f"{dt['character']} is {dt['identity']['role']} in their {dt['identity']['age']}"
        emotion = "{} may feel {}".format(dt['character'], '; '.join(dt['identity']['emotions']))
        prompt = _construct_prompt(dt['script'], dt['character'], role, emotion, dt['event1'],
                                   '\n'.join([f'{k}: {v}' for k, v in desc.items()]))
        prompts.append({'idx': frm_id, 'prompt': prompt, 'event1': dt['event1'], 'event2': dt['event2'],})
    return prompts


def extract_rst(rst, use_chatgpt=True):
    return rst['choices'][0]['message']['content'] if use_chatgpt else rst['text']


def prompt_in_batch(prompt_list, outputfile, keys, 
                    batch_size=1, use_chatgpt=True):
    indexes, prompts = [p['idx'] for p in prompt_list], [p['prompt'] for p in prompt_list]
    results = []
    num_batch = (len(prompt_list) + batch_size - 1) // batch_size
    for i in tqdm(range(num_batch)):
        inputs = prompts[i * batch_size: (i + 1) * batch_size]
        rst = None
        while rst is None:
            rst = _openai_prompt_batch(inputs, max_tokens=128, temperature=0.0, top_p=1, n=1, logprobs=1, 
                                       key=keys, min_to_wait=3, max_to_wait=10, sleep_time=5, use_chatgpt=use_chatgpt)
        rst = [extract_rst(r, use_chatgpt=use_chatgpt) for r in (rst if use_chatgpt else rst['choices'])]
        for cur_idx in range(len(inputs)):
            idx = cur_idx + i * batch_size
            prompt_item = prompt_list[idx]
            prompt_item['pred_event2'] = rst[cur_idx]
            with jsonlines.open(outputfile, mode='a') as writer:
                writer.write(prompt_item)
        results += rst
    return results


if __name__ == '__main__':
    tail = f'_{TOM_TYPE}'
    data = jsonlines_load(INPUTFILE)
    keys = KEYS[:]
    
    ToM_dict = {'role': {}, 'emotion': {}}
    for line in jsonlines_load(INPUTFILE_TOM['R']):
        if len(line) <= 1: continue
        idx = tuple(line['ID']) + ('-'.join(list(line.keys())[1:]),)
        ToM_dict['role'][idx] = '; '.join(list(set([x for xx in list(line.values())[1:] for x in xx])))
    for line in jsonlines_load(INPUTFILE_TOM['E']):
        if len(line) <= 1: continue
        idx = tuple(line['ID']) + ('-'.join(list(line.keys())[1:]),)
        ToM_dict['emotion'][idx] = '; '.join(list(set([x for xx in list(line.values())[1:] for x in xx])))
    
    ##=== construct caption set ===##
    txt_descriptions = {k: jsonlines_load(INPUTFILE_CAP[k]) for k in INPUTFILE_CAP}
    for k, v in txt_descriptions.items():
        new_v = {}
        for line in v:
            if len(line) <= 1: continue
            if 'ID' not in line: continue
            _id = tuple(line['ID'])
            fid_list = [int(kk) for kk in line if kk.isdigit()]
            frm_id = _id + ('-'.join([str(x) for x in fid_list]),)
            new_v[frm_id] = _get_desc(line, fid_list)
        txt_descriptions[k] = new_v
    
    prompt_list = get_data(data, ToM_dict, txt_descriptions)
    mtype = 'chatgpt' if USE_CHATGPT else 'base'
    results = prompt_in_batch(prompt_list, f'{OUTPUTFILE}{mtype}{tail}.json',
                              keys, batch_size=PROMPT_BATCH_SIZE, 
                              use_chatgpt=USE_CHATGPT)
