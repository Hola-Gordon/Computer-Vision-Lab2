# Computer-Vision-Lab2
# Texture Classifier

A machine learning application that classifies textures (stone, brick, wood) using GLCM and LBP feature extraction methods.

## Demo

Try it out: [Texture Classifier Demo](https://yourusername-texture-classifier.hf.space)

![Screenshot](screenshot.png)

## Overview

This application classifies texture images into three categories:
- Stone
- Brick
- Wood

It uses two different texture analysis methods:
- Gray Level Co-occurrence Matrix (GLCM) with SVM
- Local Binary Patterns (LBP) with Random Forest

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/texture-classifier.git
cd texture-classifier

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

## Usage

1. Upload a texture image
2. Select classification method (GLCM or LBP)
3. Click "Classify"
4. View results and model performance metrics

## Project Structure

```
texture-classifier/
├── data/               # Dataset folders
├── src/                # Source code
├── trained_models/     # Saved models
├── app.py              # Main application
├── requirements.txt    # Dependencies
└── README.md
```

## Requirements

- Python 3.6+
- OpenCV
- scikit-image
- scikit-learn
- Gradio
