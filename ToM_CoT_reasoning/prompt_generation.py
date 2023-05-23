import random
import openai
import jsonlines

from tqdm import tqdm
from time import sleep, time
from utils import json_load, json_dump, jsonlines_load, jsonlines_dump
from tenacity import wait_random_exponential, stop_after_attempt, retry

parallel = False
n_jobs = 4


_s, _e = 0, 10
print(f'Using keys {_s} to {_e - 1} ...')
with open('./keys/my_keys.txt', 'r', encoding='utf-8') as f:
    text = f.read().strip().split('\n')
KEYS = [line.strip().split('----')[2] for line in text[_s:_e]]

num_key = len(KEYS)
if num_key:
    if parallel:
        num_key = n_jobs * (num_key // n_jobs)
    KEYS = random.sample(KEYS, min(len(KEYS), num_key))
print('({} keys in total).'.format(len(KEYS)))
KEYS_DICT = {k: None for k in KEYS}

ERRORS = ['Rate limit reached for default-code-davinci-002', 'Request timed out']


def get_internal(keys_used, k, cur_time):
    t = 6e2    # TODO: magic number
    if keys_used[k] is not None:
        t = cur_time - keys_used[k]
    return t

def select_key(keys_used, keys, all_keys=None):
    if all_keys is not None and random.random() < 0.1:     # TODO: magic number
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
            {'role': 'system', 'content': 'You are a helpful assistant that generates **specific** questions for visual question answering to extract visual information as required. Note that you will be only given the subscript of the video clip to generate the questions without knowing the visual information.'},
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

def _construct_prompt(ptype, script, character):
    '''
    ptype: role, emotion
    '''
    if ptype == "role":
        ptype_desc = "identity (specific characteristics including role and occupation)"
        ptype_details = "appearance details"
    elif ptype == 'emotion':
        ptype_desc = "emotional traits"
        ptype_details = "facial expressions"
        
    template = "Read the subscript of a crime-drama plot below:\n{}\n\nTo know the " \
        + ptype_desc + " of {}, generate a question (for visual question answering) to extract required information (about their " \
        + ptype_details + ") in the corresponding video clip. Note that you should infer what content the clip may contain based on above information."

    return template.format(script, character)


def get_data(raw_data):
    data_dict = {}
    prompts = []
    for dt in raw_data:
        _id = (dt['ID'], dt['character'])
        if _id in data_dict: continue
        data_dict[_id] = dt
        data_dict[_id]['prompts'] = {
            'role': _construct_prompt('role', dt['script'], dt['character']),
            'emotion': _construct_prompt('emotion', dt['script'], dt['character']),
        }
        for k, p in data_dict[_id]['prompts'].items():
            prompts.append({
                'index': len(prompts), 'prompt': p, 
                'type': k, 'ID': _id,
            })
    return data_dict, {p['index']: p for p in prompts}


def extract_rst(rst, use_chatgpt=True):
    return rst['choices'][0]['message']['content'] if use_chatgpt else rst['text']


def prompt_in_batch(prompt_list, outputfile, keys, batch_size=1, use_chatgpt=True):
    promptkeys = list(prompt_list.keys())
    prompts = [p['prompt'] for p in prompt_list.values()]
    num_batch = (len(prompt_list) + batch_size - 1) // batch_size
    
    results = []
    for i in tqdm(range(num_batch)):
        inputs = prompts[i * batch_size: (i + 1) * batch_size]
        rst = None
        while rst is None:
            rst = _openai_prompt_batch(inputs, max_tokens=128, temperature=0.0, top_p=1, n=1, logprobs=1, 
                                       key=keys, min_to_wait=3, max_to_wait=10, sleep_time=5, use_chatgpt=use_chatgpt)
        rst = [extract_rst(r, use_chatgpt=use_chatgpt) for r in (rst if use_chatgpt else rst['choices'])]
        for idx in range(i * batch_size, (i + 1) * batch_size):
            prompt_list[promptkeys[idx]]['question'] = rst[idx - i * batch_size]
            with jsonlines.open(outputfile, mode='a') as writer:
                writer.write(prompt_list[promptkeys[idx]])
        results += rst
    return results


if __name__ == '__main__':
    dtype = 'all'
    to_split = False
    use_chatgpt = False
    batch_size = 1
    
    tail = ''
    if to_split:
        idx = 0
        stepwise, kstep = 1050, 250
        sid, eid = int(idx * stepwise), int((idx + 1) * stepwise)
        data = jsonlines_load(f'./dataset/{dtype}/v01_{dtype}_all.json')[sid:eid]
        keys = KEYS[idx * kstep: (idx + 1) * kstep]
        tail += f'_{idx}'
    else:
        data = jsonlines_load(f'./dataset/{dtype}/v01_{dtype}_all.json')
        keys = KEYS[:]
    
    data_dict, prompt_list = get_data(data)
    mtype = 'chatgpt' if use_chatgpt else 'base'
    results = prompt_in_batch(prompt_list, f'./dataset/{dtype}/v01_{dtype}_questions_{mtype}.json', keys, batch_size=batch_size, use_chatgpt=use_chatgpt)
    

