import torch
import requests
import jsonlines
from PIL import Image
from tqdm import tqdm

from lavis.models import load_model_and_preprocess
from utils import json_load, json_dump, jsonlines_load, jsonlines_dump

def load_image(img_url=None, img_path=None):
    if img_url:
        return Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    elif img_path:
        return Image.open(img_path).convert("RGB")
    assert False, "Invalid Image Input"

def load_model(name, mtype, device="cpu"):
    '''
    blip2_t5: pretrain_flant5xxl
              pretrain_flant5xl
              caption_coco_flant5x
    blip2_opt: pretrain_opt2.7b
               pretrain_opt6.7b
               caption_coco_opt2.7b
               caption_coco_opt6.7b
    blip_vqa: vqav2
    blip_caption: large_coco
                  base_coco
    blip2_feature_extractor: pretrain
    blip2_image_text_matching: pretrain
                               coco
    blip2_vicuna_instruct: vicuna7b
    '''
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name=name, model_type=mtype, is_eval=True, device=device
    )
    
    return {
        'model': model, 'vis': vis_processors, 'txt': txt_processors,
    }
    
def instructed_generation(img, model_dict, prompt="", device="cpu"):
    '''
    prompt examples:
        (1) Question: which city is this? Answer:
        (2) Question: which city is this? Answer: singapore. Question: why? Answer:
    name & model-types:
        blip2_t5: 
            pretrain_flant5xxl
            pretrain_flant5xl
            caption_coco_flant5x
        blip2_opt: 
            pretrain_opt2.7b
            pretrain_opt6.7b
            caption_coco_opt2.7b
            caption_coco_opt6.7b
    '''
    image = model_dict["vis"]["eval"](img).unsqueeze(0).to(device)
    
    inputs = {"image": image, "prompt": prompt} if prompt else {"image": image}
    captions = []
    # generate caption using beam search
    captions += model_dict["model"].generate(inputs, num_beams=5,
                                             max_length=64, min_length=8,
                                             repetition_penalty=2.0,
                                             length_penalty=1.0)
    # generate multiple captions using nucleus sampling
    captions += model_dict["model"].generate(inputs, use_nucleus_sampling=True, 
                                             max_length=64, min_length=8,
                                             top_p=0.9, temperature=1,
                                             repetition_penalty=2.0,
                                             length_penalty=1.0,
                                             num_captions=3)
    
    return captions

def _extract_ans(raw):
    ans = []
    for a in raw:
        for aa in a.strip().split(' / '):
            if aa not in ans: ans.append(aa)
    return ' / '.join(ans)

def captioning(data, outputfile, model_dict, vl_option=0, prompt=None, llm_questions=None, llm_answers=None):
    captions = [{'prompt': prompt}]
    with jsonlines.open(outputfile, mode='a') as writer:
        writer.write(captions[0])
    
    for dt in tqdm(data):
        cpt = {'ID': (dt['ID'], dt['aid'], dt['character'])}
        for frm in tqdm(dt['frames'], disable=True):
            if not frm['checked']: continue
            img_path = f'../frames/{dt["episode"]}/{frm["fid"]}.jpg'
            img = load_image(img_path=img_path)
            if prompt:
                complete_prompt = prompt
                if vl_option == 1:
                    complete_prompt = prompt.format(dt['script'], dt['character'])
                elif vl_option == 2:
                    fid = cpt['ID'] + (frm['fid'],)
                    _type = 'role' if 'role' in outputfile else 'facial'
                    try: qu, desc = llm_questions[fid][_type]
                    except: qu = desc = ''
                    cpt.update({f"{frm['fid']}-qu": qu, f"{frm['fid']}-desc": desc})
                    complete_prompt = prompt.format(dt['script'], desc.strip(), qu.strip())
                elif vl_option == 3:
                    fid = cpt['ID'] + (frm['fid'],)
                    _type = 'role' if 'role' in outputfile else 'emotion'
                    fans = llm_answers[_type][fid]
                    complete_prompt = prompt.format(dt['script'], dt['character'], _extract_ans(fans[0]), dt['character'])
                frm_cpt = instructed_generation(img, model_dict, prompt=f"{complete_prompt}\nAnswer:", device=device)
            else:
                frm_cpt = instructed_generation(img, model_dict, device=device)
            # import ipdb; ipdb.set_trace()
            cpt[frm['fid']] = frm_cpt
        with jsonlines.open(outputfile, mode='a') as writer:
            writer.write(cpt)
        captions.append(cpt)

def _extract_frmcap(_prompt):
    return _prompt.split('For reference, there are brief descriptions of one of the video keyframes of ')[1].split('\n')[1].strip()

if __name__ == '__main__':
    dtype = 'full'
    vl_option = 2   # 0 (no subscript), 1 (default vqa), 2 (llm-raised vqa), 3 (llm-enhanced vqa)
    gpu = 7
    to_split = True
    
    prompts = {}
    llm_questions, llm_answers = None, None
    if vl_option == 0:
        ##=== no subscript ===##
        tail = '_basic_wotxt'
        prompts = {
            # 'app': 'Describe the appearance of the person in details. You may include specific features that indicate their age, role, or identity.',
            'app': 'Question: What is the probable identity (age and role) of the person? Justify your answer using their detailed appearance in the image.',
            # 'emo': 'Describe the facial expression of the person in details. You may include further inference on their emotional traits.',
            'emo': 'Question: What are the probable emotional traits of the person? Justify your answer using their facial expressions in the image.',
        }
    elif vl_option == 1:
        ##=== default vqa ===##
        tail = '_basic'
        prompts = {
            'app': 'Read the subscript of the plot as follows:\n{}\n\nQuestion: What is the probable identity (age and role) of the person {}? Justify your answer using their detailed appearance in the image.',
            'emo': 'Read the subscript of the plot as follows:\n{}\n\nQuestion: What are the probable emotional traits of the person {}? Justify your answer using their facial expressions in the image.',
        }
    elif vl_option == 2:
        ##=== llm-raised vqa ===##
        tail = '_llm_vqa_basee'
        prompts = {
            'app': 'Read the subscript of the plot as follows:\n{}\n\nFor reference, we know from the frame that {}\n\nQuestion: {}',
            'emo': 'Read the subscript of the plot as follows:\n{}\n\nFor reference, we know from the frame that {}\n\nQuestion: {}',
        }
        llm_questions = {}
        for line in jsonlines_load(f'./dataset/{dtype}/questions/v01_{dtype}_questions_base.json'):
            _id = tuple(line['idx'])
            if _id not in llm_questions: llm_questions[_id] = {}
            llm_questions[_id][line['type']] = (line['question'], _extract_frmcap(line['prompt']))
    elif vl_option == 3:
        ##=== llm-enhanced vqa ===##
        tail = '_cot'
        prompts = {
            'app': 'Read the subscript of the plot as follows:\n{}\n\nSpecifically, we know about {} that {}\n\nQuestion: What is the probable identity (age and role) of the person {}?',
            'emo': 'Read the subscript of the plot as follows:\n{}\n\nSpecifically, we know about {} that {}\n\nQuestion: What are the probable emotional traits of the person {}?',
        }
        llm_answers = {'role': {}, 'emotion': {},}
        for line in jsonlines_load(f'./dataset/{dtype}/role_task1/v01_{dtype}_llm_vqa.json')[1:]:
            _id = tuple(line['ID'])
            fid_list = [int(k) for k in line if k.isdigit()]
            for fid in fid_list:
                idx = _id + (fid,)
                llm_answers['role'][idx] = (line[str(fid)], line[f'{fid}-qu'], line[f'{fid}-desc'])
        for line in jsonlines_load(f'./dataset/{dtype}/emotion_task2/v01_{dtype}_llm_vqa.json')[1:]:
            _id = tuple(line['ID'])
            fid_list = [int(k) for k in line if k.isdigit()]
            for fid in fid_list:
                idx = _id + (fid,)
                llm_answers['emotion'][idx] = (line[str(fid)], line[f'{fid}-qu'], line[f'{fid}-desc'])
    
    if to_split:
        idx = gpu - 4
        stepwise = 1050
        sid, eid = int(idx * stepwise), int((idx + 1) * stepwise)
        data = jsonlines_load(f'./dataset/{dtype}/v01_{dtype}_all.json')[sid:eid]
        tail += f'_{idx}'
    else:
        data = jsonlines_load(f'./dataset/{dtype}/v01_{dtype}_all.json')
    
    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else "cpu"
    model_dict = load_model("blip2_t5", "pretrain_flant5xl", device=device)
    
    if not vl_option:
        ## general captioning (no subscript input) ##
        outputfile = f'./dataset/{dtype}/captions/v01_{dtype}_captions_general_wotxt.json'
        captioning(data, outputfile, model_dict)
    
    outputfile = f'./dataset/{dtype}/role_task1/v01_{dtype}{tail}.json'
    captioning(data, outputfile, model_dict, vl_option=vl_option, prompt=prompts['app'], 
               llm_questions=llm_questions, llm_answers=llm_answers)
    
    outputfile = f'./dataset/{dtype}/emotion_task2/v01_{dtype}{tail}.json'
    captioning(data, outputfile, model_dict, vl_option=vl_option, prompt=prompts['emo'], 
               llm_questions=llm_questions, llm_answers=llm_answers)
        