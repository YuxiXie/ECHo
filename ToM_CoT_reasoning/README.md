## Framework of Theory-of-Mind (ToM) Enhanced Chain-of-Thought (CoT) Reasoning

We provide the code to prompt the Large Multimodal and Language Models iteratively to conduct ToM-enhanced CoT reasoning.

### Prompting Multimodal Models

```bash
python prompt_captioning.py
```
The current code simply demonstrates how to prompt the model to conduct captioning, visual question answering, and prompted inference on the tasks of `role identification` and `emotion recognition`. 
We will release the comprehensive implementation of the `event causality inference` pipeline soon.

<sub><sup>This script is adapted from the examples from [LAVIS repo](https://github.com/salesforce/LAVIS)</sup></sub>.

### Prompting LLMs

1. To raise question for further visual information extraction.
```bash
python prompt_generation.py
```

2. To conduct event causality inference
```bash
```
