# Sidewalk Tagger AI

> **Alex (Xinlei) Liu\*, Kevin Wu\*, Minchu Kulkarni, Michael Sugstad, Peyton Rapo, Jeremy Freiburger, Maryam Hosseini, Chu Li, Jon E. Froehlich**
> 
> We examine the feasibility of using deep learning to infer 33 classes of sidewalk accessibility conditions in pre-cropped streetscape images, including bumpy, brick/cobblestone, cracks, height diference (uplifts), narrow, uneven/slanted, pole, and sign. We present two experiments: frst, a comparison between two state-of-the-art computer vision models, Meta’s DINOv2 and OpenAI’s CLIP-ViT, on a cleaned dataset of ∼24k images; second, an examination of a larger but noisier crowdsourced dataset (∼87k images) on the best performing model from Experiment 1. Though preliminary, Experiment 1 shows that certain sidewalk conditions can be identifed with high precision and recall, such as missing tactile warnings on curb ramps and grass grown on sidewalks, while Experiment 2 demonstrates that larger but noisier training data can have a detrimental efect on performance. We contribute an open dataset and classifcation benchmarks to advance this important area.

![Teaser Image](docs/figure-teaser.png)

## Datasets

This repository contains:
- Code to train and evaluate multi-label classification models with DINOv2 model and OpenAI ViT-CLIP models as base models for predicting Project Sidewalk's tags to assess sidewalk accessibility conditions.
- Links to two datasets: 
  - [Cleaned Dataset (Dataset 1)](to be added)
  - [Uncleaned Dataset (Dataset 2)](to be added)

For each dataset, we provide:
- A directory containing the images and the corresponding tags information in a CSV file organized by label type.
- The CSV file contains the image names, normalized X and Y coordinates of the label points, and their corresponding multi-hot encoded labels information.


## Code

There are two main notebooks to train DINOv2 and CLIP-ViT models:
- `notebooks/dino-trainer.ipynb`: Train DINOv2 model.
- `notebooks/clip-vit-trainer.ipynb`: Train CLIP-ViT model.

Both notebooks use a pretrained model and fine-tune it on the Project Sidewalk dataset.

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

The DinoV2 pre-trained weights should be downloaded separately.

## Usage

1. Set the desired configuration parameters in the notebook:
   - `label_type`: The type of label to train on (e.g. 'curbramp', 'surfaceproblem')
   - `c12n_category`: Classification category ('TAGS' or 'SEVERITY')
   - `base_model_size`: Size of DinoV2 model to use ('small', 'base', 'large', 'giant')

2. Ensure the training data is in the expected directory structure.

3. Run all cells in the notebook to train the model.

4. The best model will be saved based on training accuracy and loss.


<!-- ## Acknowledgements -->

<!-- This repository is partially based on [Diffusers](https://github.com/huggingface/diffusers) and [Collage Diffusion](https://github.com/VSAnimator/collage-diffusion). -->
