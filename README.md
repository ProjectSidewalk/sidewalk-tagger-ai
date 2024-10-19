# Dino Trainer

This Jupyter notebook trains a classification model using DinoV2 to predict tags or severity for Project Sidewalk label types.

## Overview

The notebook uses a pre-trained DinoV2 vision transformer model and fine-tunes it on Project Sidewalk image data to classify accessibility issues in street-level imagery. Key features include:

- Customizable training for different label types (e.g. curb ramps, surface problems)
- Support for multi-label classification (tags) or regression (severity)
- Data augmentation and preprocessing pipeline
- Training loop with checkpointing and early stopping
- Visualization of training metrics

## Requirements

- PyTorch
- torchvision  
- matplotlib
- pandas
- scikit-learn
- comet_ml (optional, for experiment tracking)

The DinoV2 pre-trained weights should be downloaded separately.

## Usage

1. Set the desired configuration parameters in the notebook:
   - `label_type`: The type of label to train on (e.g. 'curbramp', 'surfaceproblem')
   - `c12n_category`: Classification category ('TAGS' or 'SEVERITY')
   - `base_model_size`: Size of DinoV2 model to use ('small', 'base', 'large', 'giant')

2. Ensure the training data is in the expected directory structure.

3. Run all cells in the notebook to train the model.

4. The best model will be saved based on validation accuracy.

## Customization

The notebook can be adapted for different datasets by modifying:

- The data loading and preprocessing steps
- The model architecture (e.g. classification head)
- Training hyperparameters

## License

MIT License

Copyright (c) [year] [fullname]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
