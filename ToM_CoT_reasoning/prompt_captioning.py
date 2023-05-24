import torch
import requests
import jsonlines
from PIL import Image
from tqdm import tqdm

from lavis.models import load_model_and_preprocess
from utils import json_load, json_dump, jsonlines_load, jsonlines_dump


## path to the input .jsonl file (each line is an instance)
INPUTFILE = ''
OUTPUTFILE = ''

## choices:
#      1) default: general, appearance-specific, facial-specific captioning
#      2) vqa: answer LLM-raised questions
#      3) inference
PROMPT_TYPE = ''

## LLM questions & MM answers
INPUTFILE_LLMQ = ''
INPUTFILE_LLMA = {'R': '', 'E': ''}


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


def captioning(data, outputfile, model_dict, 
               option='default', prompt=None, 
               llm_questions=None, vqa_answers=None):
    with jsonlines.open(outputfile, mode='a') as writer:
        writer.write({'prompt': prompt})
    
    for dt in tqdm(data):
        cpt = {'ID': (dt['ID'], dt['aid'], dt['character'])}
        for frm in tqdm(dt['frames'], disable=True):
            if not frm['checked']: continue
            img_path = f'../frames/{dt["episode"]}/{frm["fid"]}.jpg'
            img = load_image(img_path=img_path)
            if prompt is not None:
                complete_prompt = prompt
                if option == 'vqa':
                    fid = cpt['ID'] + (frm['fid'],)
                    _type = 'role' if 'role' in outputfile else 'facial'
                    try: qu, desc = llm_questions[fid][_type]
                    except: qu = desc = ''
                    cpt.update({f"{frm['fid']}-qu": qu, f"{frm['fid']}-desc": desc})
                    complete_prompt = prompt.format(dt['script'], desc.strip(), qu.strip())
                elif option == 'inference':
                    fid = cpt['ID'] + (frm['fid'],)
                    _type = 'role' if 'role' in outputfile else 'emotion'
                    fans = vqa_answers[_type][fid]
                    complete_prompt = prompt.format(dt['script'], dt['character'], _extract_ans(fans[0]), dt['character'])
                frm_cpt = instructed_generation(img, model_dict, prompt=f"{complete_prompt}\nAnswer:", device=device)
            else:
                frm_cpt = instructed_generation(img, model_dict, device=device)
            cpt[frm['fid']] = frm_cpt
        with jsonlines.open(outputfile, mode='a') as writer:
            writer.write(cpt)


def _extract_frmcap(_prompt):
    return _prompt.split('For reference, there are brief descriptions of one of the video keyframes of ')[1].split('\n')[1].strip()


if __name__ == '__main__':
    gpu_id = 0
    
    data = jsonlines_load(INPUTFILE)
    device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else "cpu"
    
    prompts, llm_questions, vqa_answers = {}, None, None
    if PROMPT_TYPE == 'default':
        tail = '_basic'
        prompts = {
            'R': 'Describe the appearance of the person in details. You may include specific features that indicate their age, role, or identity.',
            'E': 'Describe the facial expression of the person in details. You may include further inference on their emotional traits.',
        }
    elif PROMPT_TYPE == 'vqa':
        tail = '_vqa'
        prompts = {
            'R': 'Read the screenplay of the plot (with the input frame) as follows:\n{}\n\nQuestion: {}',
            'E': 'Read the screenplay of the plot (with the input frame) as follows:\n{}\n\nQuestion: {}',
        }
        llm_questions = {}
        for line in jsonlines_load(INPUTFILE_LLMQ):
            _id = tuple(line['idx'])
            if _id not in llm_questions: llm_questions[_id] = {}
            llm_questions[_id][line['type']] = (line['question'], _extract_frmcap(line['prompt']))
    elif PROMPT_TYPE == 'inference':
        tail = '_tomcot'
        prompts = {
            'R': 'Read the screenplay of the plot (with the input frame) as follows:\n{}\n\nSpecifically, we know about {} that {}\n\nQuestion: What is the probable identity (age and role) of the person {}?',
            'E': 'Read the subscript of the plot (with the input frame) as follows:\n{}\n\nSpecifically, we know about {} that {}\n\nQuestion: What are the probable emotional traits of the person {}?',
        }
        vqa_answers = {'R': {}, 'E': {},}
        for line in jsonlines_load(INPUTFILE_LLMA['R']):
            if len(line) <= 1: continue
            _id = tuple(line['ID'])
            fid_list = [int(k) for k in line if k.isdigit()]
            for fid in fid_list:
                idx = _id + (fid,)
                vqa_answers['R'][idx] = (line[str(fid)], line[f'{fid}-qu'], line[f'{fid}-desc'])
        for line in jsonlines_load(INPUTFILE_LLMA['E']):
            if len(line) <= 1: continue
            _id = tuple(line['ID'])
            fid_list = [int(k) for k in line if k.isdigit()]
            for fid in fid_list:
                idx = _id + (fid,)
                vqa_answers['E'][idx] = (line[str(fid)], line[f'{fid}-qu'], line[f'{fid}-desc'])
        
    model_dict = load_model("blip2_t5", "pretrain_flant5xl", device=device)
    
    if PROMPT_TYPE == 'default':
        captioning(data, f'{OUTPUTFILE}{tail}_general.json', model_dict)
    
    outputfile = f'{OUTPUTFILE}{tail}_role.json'
    captioning(data, outputfile, model_dict, option=PROMPT_TYPE, prompt=prompts['R'], 
               llm_questions=llm_questions, llm_answers=vqa_answers)
    
    outputfile = f'{OUTPUTFILE}{tail}_emotion.json'
    captioning(data, outputfile, model_dict, option=PROMPT_TYPE, prompt=prompts['E'], 
               llm_questions=llm_questions, vqa_answers=vqa_answers)
        