# V-RoAst: Visual Road Assessment with Vision–Language Models (VLMs)

> Can VLMs act as iRAP road safety assessors in a zero-shot setting?

[arXiv](https://arxiv.org/abs/2408.10872) • ThaiRAP dataset (images + labels) • Zero-shot VQA pipeline • CNN baselines (VGG/ResNet)

---

## Overview
Road safety assessments are costly and data-hungry, especially in LMICs. V-RoAst is a zero-shot Visual Question Answering framework that uses general-purpose Vision–Language Models (e.g., Gemini-1.5-Flash, GPT-4o-mini) to classify 52 iRAP attributes from street-level imagery. We release:
- An open ThaiRAP dataset (2,037 images, 519 segments) with expert-coded iRAP labels.
- Prompt templates and code to query VLMs for attribute classification.
- CNN baselines (VGG/ResNet) for comparison.

## Key Contributions
- **Open VLM Benchmark** for iRAP-style attributes.
- **Prompt engineering framework** (system + user prompts with local context).
- **Zero-shot performance analysis**, incl. unseen classes.
- **Automatic star-rating demo** using crowdsourced Mapillary imagery.

---



## Abstract
Road safety assessments are critical yet costly, especially in Low- and Middle-Income Countries (LMICs), where most roads remain unrated. Traditional methods require expert annotation and training data, while supervised learning-based approaches struggle to generalise across regions. In this paper, we introduce V-RoAst, a zero-shot Visual Question Answering (VQA) framework using VisionLanguage Models (VLMs) to classify road safety attributes defined by the iRAP standard. We introduce the first opensource dataset from ThaiRAP, consisting of over 2,000 curated street-level images from Thailand annotated for this task. We evaluate Gemini-1.5-flash and GPT-4o-mini on this dataset and benchmark their performance against VGGNet and ResNet baselines. While VLMs underperform on spatial awareness, they generalise well to unseen classes and offer flexible prompt-based reasoning without retraining. Our results show that VLMs can serve as automatic road assessment tools when integrated with complementary data. This work is the first to explore VLMs for zero-shot infrastructure risk assessment and opens new directions for automatic, low-cost road safety mapping. Code and dataset:https://github.com/PongNJ/V-RoAst.

## Installation

### Step 1: Experimental Platform 🛠️

- **OpenAI**: We used OpenAI version 1.40.3. Find the documentation [here](https://platform.openai.com/docs/overview). 

- **Google Gemini**: We used google-generativeai version 0.7.2. Find the documentation [here](https://ai.google.dev/gemini-api/docs).

- **Mapillary API**: Access the documentation [here](https://www.mapillary.com/developer/api-documentation).

### Step 2: V-RoAst Installation (This will be available later)

```bash
git clone https://github.com/PongNJ/V-RoAst.git
```

## ThaiRAP Dataset 📂

Please download ThaiRAP dataset from [(google drive)](https://drive.google.com/drive/folders/1FoAoAQ3oRg0nHIBLGLx61lpmaxrI-0BI?usp=sharing) or [(ucl rdr)](https://rdr.ucl.ac.uk/articles/figure/V-RoAst_A_New_Dataset_for_Visual_Road_Assessment/26520787?file=48277894) and upload all images to the ./image/ThaiRAP/ directory.

The ThaiRAP dataset combines street images with road attributes, stored in a CSV file, as shown in the structure below:


### ThaiRAP Structure:

```
├─V-RoAst
│  ├─image
│  │  ├─ThaiRAP
│  │  │  ├─1.jpg
│  │  │  ├─2.jpg
│  │  │  ├─...
│  │  │  └─2037.jpg
│  └─Validation.csv
│
```
### ThaiRAP Location and Sample of Images in a 100-m road segment
<img src="figure/ThaiRAP_location_and_samples.png" alt="ThaiRAP location and samples" width="500"/>

### ThaiRAP Attribute Distribution
![ThaiRAP Attribute Distribution](./figure/bar_graph.png)

### Framework of V-RoAst for visual road assessment
![Framework of V-RoAst for visual road assessment](figure/V-RoAst_framework.png)

![ext Prompts from Framework of V-RoAst](figure/text_promt.png)



## Citation

If you use this dataset or refer to our work, please cite:

```bibtex
@misc{jongwiriyanurak2025vroastvisualroadassessment,
      title={V-RoAst: Visual Road Assessment. Can VLM be a Road Safety Assessor Using the iRAP Standard?}, 
      author={Natchapon Jongwiriyanurak and Zichao Zeng and June Moh Goo and Xinglei Wang and Ilya Ilyankou and Kerkritt Sriroongvikrai and Nicola Christie and Meihui Wang and Huanfa Chen and James Haworth},
      year={2025},
      eprint={2408.10872},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.10872}, 
}
```




