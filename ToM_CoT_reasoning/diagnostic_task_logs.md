## Diagnostic Tasks

**PS**: We tried to generate video captions by concatenating a sequence of frames, but BLIP-2 cannot appropriately consider them sequentially in a clip. 

### Task 1. role identification

**Justification.** `role` usually indicates both individual and relational/interactional information of humans in social scenarios. It is important to understand and infer the probable reasons and outcomes of human behaviors.

#### CoT reasoning
* **task description.** to determine the role (_e.g._, age, occupation) of a specified character
* **input/output formulation.**
    - <u>vanilla</u>: frame + subscript $\rightarrow$ role (BLIP-2 VQA)
        * &#10004; ablation: w/o textual input (subscript)
        * &#10004; main experiment
    - <u>LLM-enhanced CoT</u>:
        * &#10004; (appearance) characteristics recognition (BLIP-2 VQA)
        * &#10004; visual question generation (LLM generation)
        * &#10004; specific visual information extraction (BLIP-2 VQA) as `LLM-enhanced CoT`
        * &#10004; role identification with `LLM-enhanced CoT` (~~BLIP-2 VQA~~ LLM generation)
* **justification.** for the task itself, the CoT framework demonstrates a way to prompt both language and multimodal models to conduct human-centric reasoning in the zero-shot case.


### Task 2. emotional trait prediction

**Justification.** `emotion` usually indicates the mental states of humans. It closely connects to the intents, thoughts, and actions.

#### CoT reasoning
* **task description.** to determine the emotional trait(s) of a specified character
* **input/output formulation.**
    - <u>vanilla</u>: frame + subscript $\rightarrow$ emotion (BLIP-2 VQA)
        * &#10004; ablation: w/o textual input (subscript)
        * &#10004; main experiment
    - <u>LLM-enhanced CoT</u>:
        * &#10004; facial expression recognition (BLIP-2 VQA)
        * &#10004; visual question generation (LLM generation)
        * &#10004; specific visual information extraction (BLIP-2 VQA) as `LLM-enhanced CoT`
        * &#10004; emotion prediction with `LLM-enhanced CoT` (~~BLIP-2 VQA~~ LLM generation)
* **justification.** for the task itself, the CoT framework demonstrates a way to prompt both language and multimodal models to conduct human-centric reasoning in the zero-shot case.


### Task 3. event causality inference

**Justification.** **TODO**

#### CoT reasoning
* **task description.** to predict the cause/effect following ToM-enhanced reasoning of specified events
* **input/output formulation.**
    - <u>vanilla</u>: frames + subscript $\rightarrow$ causality inference (BLIP-2 VQA)
        * &#10004; checked frames
        * &#10004; all frames
    - <u>LLM-enhanced CoT</u>:
        * &#10004; basic and human-centric information extraction (**Task 1 & 2**)
        * inference (w/ prompt) using
            * &#10004; BLIP-2 VQA
            * &#10004; LLM generation

