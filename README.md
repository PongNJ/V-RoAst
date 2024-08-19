# V-RoAst: A New Dataset for Visual Road Assessment

This repo uses for AAAI 2025 Artificial Intelligence for Social Impact Track Submission

## Abstract
Road traffic crashes cause millions of deaths annually and have a significant economic impact, particularly in low- and middle-income countries (LMICs). This paper presents an approach using Vision Language Models (VLMs) for road safety assessment, overcoming the limitations of traditional Convolutional Neural Networks (CNNs). We introduce a new task ,V-RoAst (Visual question answering for Road Assessment), with a real-world dataset. Our approach optimizes prompt engineering and evaluates advanced VLMs, including Gemini-1.5-flash and GPT-4o-mini. The models effectively examine attributes for road assessment. Using crowdsourced imagery from Mapillary, our scalable solution influentially estimates road safety levels. In addition, this approach is designed for local stakeholders who lack resources, as it does not require training data. It offers a cost-effective and automated methods for global road safety assessments, potentially saving lives and reducing economic burdens.

## Installation

### Step 1: Experimental Platform

The model of OpenAI GPT we used is XXXXXX. Find the documentation [here](https://xxxxxx). 

The model of Google Gemini we used is XXXXXX. Find the documentation [here](https://xxxxxx). 


### Step 2: V-RoAst

```bash
git clone https://github.com/xxxxxxxxx.git
```

## ThaiRAP

Please download ThaiRAP [here](https://xxxxxx). 

The structure of ThaiRAP dataset combining street images with road attributes has shown below. Road attributes are saved in a csv file.

### ThaiRAP Structure:

```
├─FI-London
│  ├─Image
│  │  ├─1.jpg
│  │  ├─2.jpg
│  │  ├─...
│  │  └─xxxx.jpg
│  └─XXXX.csv
```

