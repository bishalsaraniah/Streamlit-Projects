# Crop-Stat: A Crop Disease Detecction and Treatment Recommendation system using Gemini AI

## Overview
**Crop-Stat** is an AI-powered system designed to detect crop diseases from plant leaf images and provide accurate treatment recommendations. It leverages deep learning models trained with **TensorFlow** and integrates **Google’s Gemini AI** for generating treatment suggestions. The user interface is built with **Streamlit**, enabling an interactive and simple web experience.

## Features
- **Gemini API Integration**: Generates treatment suggestion to prevent crop loss.

- **Streamlit Interface**: Provides a simple, interactive web app to upload or capture an image to get the disease name.

## Project Structure

- `tensorflow` : Used for training image datasets. Datasets are taken from `kaggle.com`.

- `streamlit.py` : Source code is written here and deployed at `https://streamlit.io`.

- `google-colab` : It reqquires high computation power, memory and hardware. Google service platform provides everything to train our dataset for a free trial. Change runtime-type to `v2-8 TPU`.

- `ipynb` : Contains the model training code in ipynb extension.

## Example Workflow

1. **Input**: Upload or capture of the plant leaf image.
  
2. **Processing**: The image will go through Convolution Neural Network (CNN) for preprocessing, data augmentation, labelling, maxpooling and classification layering.

3. **Output**: The result will showcase crop name, crop disease, and treatment suggestions.

## API Key
- **Google Gemini**: Create your API key from `https://ai.google.dev/gemini-api/docs/api-key`.


## Dataset
- Download the dataset from the given link `https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset`.

- Extract the Dataset and make sure to delete `valid folder` as we only need the `train folder`. Path : `New Plant Diseases Dataset/train/`.

- Group all the images according to the folder structure given below:

```
─── dataset
    ├── Apple
    │   └── all apple images
    ├── Cherry
    │   └── all cherry images
    ├── Corn
    │   └── all corn images
    ├── Grape
    │   └── all grape images
    ├── Peach
    │   └── all peach images
    ├── Pepper
    │   └── all pepper images
    ├── Potato
    │   └── all potato images
    ├── Strawberry
    │   └── all strawberry images
    └── Tomato
        └── all tomato images
```

- Download the repository `https://github.com/bishalsaraniah/Streamlit-Projects/tree/main/Crop%20Stat` and mode the dataset folder inside it.

- Install all the dependencies via `pip` using the `requirements.txt` file.

- Use Python Version `3.12.8` to avoid any unnecessary errors.

## Dependencies

- `numpy` : An open-source python library used for scientific programming in the field of Engineering, Mathematics, Data Science etc.

- `google-generativeai` : A subset of Google Gemini AI for generating treatment recommendation solutions to prevent any crop losses in the future.

- `reportlab` : A Python library used for creating PDF documents and graphics. It offers a variety of features like PDF generation, graphics, customization and has a vector graphics library. 

- `tensorflow` : An open-source machine learning platform developed by Google, primarily used for deep learning and machine learning tasks. It has user-friendly interface and also suitable for traditional machine learning.

- `streamlit`: An open-source Python library for creating interactive web applications for data science and machine learning. It allows developers to share data apps using python. Requires minimal web development knowledge.

- `python-dotenv` : It is a Python library that simplifies the management of environment variables. It reads key-value pairs from a . env file and translates them into the environment variables available to the Python application.

- `pillow` : A powerful Python library used for image processing, supports a wide variety of image formats, such as JPEG, PNG, GIF etc. It offers a wide range of functionalities for image manipulation, customization etc.

## Create and activate a virtual environment:


1. **Creating .env file**
```
GEMINI_API_KEY="your_gemini_api_key"
```

2. **Creating Virtual Environment**
```python
python -m venv .venv
```

3. **Activating Virtual Environment**
```
.venv\Scripts\activate 

```

**4. Requirements**

```
Use Python Version 3.12.8 for development server and production server
```

5. **Project Structure**

```
project-root/
│
├── .venv
├── dataset
│   ├── Apple
│   │   └── all apple images
│   ├── Cherry
│   │   └── all cherry images
│   ├── Corn
│   │   └── all corn images
│   ├── Grape
│   │   └── all grape images
│   ├── Peach
│   │   └── all peach images
│   ├── Pepper
│   │   └── all pepper images
│   ├── Potato
│   │   └── all potato images
│   ├── Strawberry
│   │   └── all strawberry images
│   └── Tomato
│       └── all tomato images
├── ipynb
│   ├── apple_disease_classification_model.ipynb
│   ├── cherry_disease_classification_model.ipynb
│   ├── corn_disease_classification_model.ipynb
│   ├── grape_disease_classification_model.ipynb
│   ├── peach_disease_classification_model.ipynb
│   ├── pepper_disease_classification_model.ipynb
│   ├── potato_disease_classification_model.ipynb
│   ├── strawberry_disease_classification_model.ipynb
│   └── tomato_disease_classification_model.ipynb
├── model
│   ├── apple.h5
│   ├── cherry.h5
│   ├── corn.h5
│   ├── grape.h5
│   ├── peach.h5
│   ├── pepper.h5
│   ├── potato.h5
│   ├── strawberry.h5
│   └── tomato.h5
├── test_images
│   ├── apple black rot
│   ├── apple black rot 2
│   ├── apple black rot 3
│   ├── cherry powdery mildew
│   ├── cherry powdery mildew 2
│   ├── potato early blight
│   ├── potato early blight 2
│   ├── potato early blight 3
│   ├── potato early blight 4
│   └── tomato late blight
├── .env
├── .gitignore
├── class_indices.json
├── example.env
├── main.py
├── Readme.md
├── requirements.txt
└── streamlit.py
```