import json

import torch
import os
from pathlib import Path
import pandas as pd
from PIL import Image
from torchvision import datasets, transforms
import numpy as np
from torch import nn, optim
from copy import deepcopy
import sys
from visualize import draw_confusion_matrices

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dinov2.models.vision_transformer import vit_small, vit_base, vit_large, vit_giant2


class ResizeAndPad:
    def __init__(self, target_size, multiple):
        self.target_size = target_size
        self.multiple = multiple

    def __call__(self, img):
        # Resize the image
        img = transforms.Resize(self.target_size)(img)

        # Calculate padding
        pad_width = (self.multiple - img.width % self.multiple) % self.multiple
        pad_height = (self.multiple - img.height % self.multiple) % self.multiple

        # Apply padding
        img = transforms.Pad(
            (pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2))(img)

        return img


class DinoVisionTransformerClassifier(nn.Module):

    def __init__(self, model_size="small", nc=0):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.model_size = model_size

        if nc == 0:
            print("Number of classes must be greater than 0")
            exit(1)

        # loading a model with registers
        n_register_tokens = 4

        if model_size == "small":
            model = vit_small(patch_size=14,
                              img_size=526,
                              init_values=1.0,
                              num_register_tokens=n_register_tokens,
                              block_chunks=0)
            self.embedding_size = 384
            self.number_of_heads = 6

        elif model_size == "base":
            model = vit_base(patch_size=14,
                             img_size=526,
                             init_values=1.0,
                             num_register_tokens=n_register_tokens,
                             block_chunks=0)
            self.embedding_size = 768
            self.number_of_heads = 12

        elif model_size == "large":
            model = vit_large(patch_size=14,
                              img_size=526,
                              init_values=1.0,
                              num_register_tokens=n_register_tokens,
                              block_chunks=0)
            self.embedding_size = 1024
            self.number_of_heads = 16

        elif model_size == "giant":
            model = vit_giant2(patch_size=14,
                               img_size=526,
                               init_values=1.0,
                               num_register_tokens=n_register_tokens,
                               block_chunks=0)
            self.embedding_size = 1536
            self.number_of_heads = 24

        # Download pre-trained weights and place locally as-needed:
        # - small: https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth
        # - base:  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
        # - large: https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth
        # - giant: https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth
        model.load_state_dict(torch.load(Path('{}/../dinov2_vitb14_reg4_pretrain.pth'.format(local_directory))))

        self.transformer = deepcopy(model)

        self.classifier = nn.Sequential(nn.Linear(self.embedding_size, 256), nn.ReLU(), nn.Linear(256, nc))

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x


def get_labels_ref_for_run(inference_set_dir):
    csv_file_path = os.path.join(inference_set_dir, '_classes.csv')
    label_data = pd.read_csv(csv_file_path)

    # get the header row
    header_row = label_data.columns.tolist()

    # get the index of 'validated_by' column
    validated_by_index = header_row.index('validated_by')

    # get everything after 'validated_by' column
    labels_ref_for_run = header_row[validated_by_index + 1:]

    # update c12n_category_offset
    global c12n_category_offset
    c12n_category_offset = validated_by_index + 1

    return labels_ref_for_run


image_dimension = 256

# This is what DinoV2 sees
target_size = (image_dimension, image_dimension)


confidence_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1]


# enum to track the classification categories
C12N_CATEGORIES = {
    'TAGS': 'tags',
    'SEVERITY': 'severity',
}

# ------------------------------
# all the parameters to be customized for the run
# ------------------------------
label_type = 'surfaceproblem'
c12n_category = C12N_CATEGORIES['TAGS']
inference_set_dir_name = 'test'

# temporarily skipping the cities with messy data
skip_cities = ['cdmx', 'spgg', 'newberg', 'columbus']

dataset_dirname = 'crops-' + label_type + '-' + c12n_category  # example: crops-surfaceproblem-tags-archive
# dataset_dirname = 'crops-' + label_type + '-' + c12n_category + '-validated'  # example: crops-surfaceproblem-tags-archive
dataset_dir_path = '../datasets/' + dataset_dirname  # example: ../datasets/crops-surfaceproblem-tags-archive

inference_dataset_dir = Path(dataset_dir_path + "/" + inference_set_dir_name)

model_name = 'cls-b-' + label_type + '-' + c12n_category + '-best.pth'
# model_name = 'cls-b-obstacle-tags-masked-best.pth'
# ------------------------------

local_directory = os.getcwd()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if torch.cuda.is_available():
    print("GPU available")
else:
    print("GPU not available")

data_transforms = {
    "train": transforms.Compose(
        [
            ResizeAndPad(target_size, 14),
            # transforms.RandomRotation(360),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "inference": transforms.Compose([ResizeAndPad(target_size, 14),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                     ])
}

c12n_category_offset = 8


# it's okay if the csv contains more filenames than the images in the directory
# we will only load the images that are present in the directory and query the csv for labels
def images_loader(dir_path, batch_size, imgsz, transform):
    file_path = os.path.join(dir_path, '_classes.csv')
    label_data = pd.read_csv(file_path)

    filenames = []
    images = []
    labels = []

    fs = os.listdir(dir_path)
    count = 0

    for filename in fs:
        if filename.endswith(".png") or filename.endswith(".jpg"):

            # skip the cities that are in the list
            city = filename.split('-')[1]
            if city in skip_cities:
                continue


            img = Image.open(os.path.join(dir_path, filename))
            img = img.convert('RGB')
            if img is not None:

                count += 1

                print(str(count) + '/' + str(len(fs)) + ' | Loading image: {}'.format(filename))

                img = img.resize((imgsz, imgsz))
                img = data_transforms[transform](img)

                labels_for_image = label_data.query('filename == @filename')

                if len(labels_for_image) == 0:
                    continue

                if labels_for_image['label_type_validation'].values[0] != 'agree':
                    print('Disagreed or unsure label: ' + filename)
                    continue

                images.append(torch.tensor(np.array([img], dtype=np.float32), requires_grad=True))
                labels.append(
                    torch.tensor(np.array([labels_for_image.values[0][c12n_category_offset:]], dtype=np.float32), requires_grad=True))
                filenames.append(filename)

    return images, labels, filenames


def data_loader(dir_path, batch_size, imgsz, transform):
    images, labels, filenames = images_loader(dir_path, batch_size, imgsz, transform)

    return list(zip(images, labels, filenames))


# -----------------------------------------------------------------


serverity_labels = ['s-1', 's-2', 's-3', 's-4', 's-5']

# todo this needs a better name!
labels_ref_for_run = get_labels_ref_for_run(inference_dataset_dir)


all_n_incorrect_predictions_to_filenames = {}
all_category_to_true_positive_counts = {}
all_category_to_false_positive_counts = {}
all_category_to_false_negative_counts = {}
all_category_to_true_negative_counts = {}

all_category_to_prediction_stats = {}
all_category_to_prediction_details = {}


images_and_labels = data_loader(inference_dataset_dir, 1, image_dimension, 'inference')

# -----------------------------------------------------------------
def inference_on_validation_data(inference_model, confidence_level=0.5):

    # we track these for each confidence level
    n_incorrect_predictions_to_filenames = {}
    category_to_true_positive_counts = {}
    category_to_false_positive_counts = {}
    category_to_false_negative_counts = {}

    category_to_prediction_stats = {}
    category_to_prediction_details = {}

    for img_label_filename in images_and_labels:

        img_tensor, labels, filename = img_label_filename

        input_tensor = img_tensor.to(device)
        labels_tensor = labels.to(device)

        # run model on input image data
        with torch.no_grad():
            embeddings = inference_model.transformer(input_tensor)
            x = inference_model.transformer.norm(embeddings)
            output_tensor = inference_model.classifier(x)

            # Convert outputs to probabilities using sigmoid
            if c12n_category == C12N_CATEGORIES['SEVERITY']:
                probabilities = torch.softmax(output_tensor, dim=1)
            elif c12n_category == C12N_CATEGORIES['TAGS']:
                probabilities = torch.sigmoid(output_tensor)


            # Convert probabilities to predicted classes
            predicted_classes = probabilities > confidence_level
            # Calculate accuracy
            n_labels = labels.size(1)

            n_incorrect_predictions = (predicted_classes != labels_tensor.byte()).sum().item()
            correct_predictions = ((predicted_classes == labels_tensor.byte()).sum().item()) / n_labels

            # updating number of incorrect predictions to file names
            if n_incorrect_predictions in n_incorrect_predictions_to_filenames:
                n_incorrect_predictions_to_filenames[n_incorrect_predictions].append(filename)
            else:
                n_incorrect_predictions_to_filenames[n_incorrect_predictions] = [filename]


            predicted_classes_list = predicted_classes.tolist()[0]
            ground_truth_labels_list = labels.tolist()[0]

            # getting the list of predicted and ground truth labels for the current crop
            predicted_classes_for_crop = []
            for x in range(len(predicted_classes_list)):
                if predicted_classes_list[x]:
                    predicted_classes_for_crop.append(labels_ref_for_run[x])

            gt_labels_for_crop = []
            for x in range(len(ground_truth_labels_list)):
                if ground_truth_labels_list[x] == 1.0:
                    gt_labels_for_crop.append(labels_ref_for_run[x])


            # updating true positives
            for elem in predicted_classes_for_crop:
                if elem in gt_labels_for_crop:
                    if elem in category_to_true_positive_counts:
                        category_to_true_positive_counts[elem].append(filename)
                    else:
                        category_to_true_positive_counts[elem] = [filename]

                    if elem in category_to_prediction_stats:
                        category_to_prediction_stats[elem]['true-positive'] += 1
                        category_to_prediction_details[elem]['true-positive'].append({'filename': filename, 'predicted': predicted_classes_for_crop, 'ground-truth': gt_labels_for_crop})
                    else:
                        category_to_prediction_stats[elem] = {'true-positive': 1, 'false-positive': 0, 'false-negative': 0, 'true-negative': 0}
                        category_to_prediction_details[elem] = {'true-positive': [{'filename': filename, 'predicted': predicted_classes_for_crop, 'ground-truth': gt_labels_for_crop}], 'false-positive': [], 'false-negative': [], 'true-negative': []}

            # updating false positives
            for elem in predicted_classes_for_crop:
                if elem not in gt_labels_for_crop:
                    if elem in category_to_false_positive_counts:
                        category_to_false_positive_counts[elem].append(filename)
                    else:
                        category_to_false_positive_counts[elem] = [filename]

                    if elem in category_to_prediction_stats:
                        category_to_prediction_stats[elem]['false-positive'] += 1
                        category_to_prediction_details[elem]['false-positive'].append({'filename': filename, 'predicted': predicted_classes_for_crop, 'ground-truth': gt_labels_for_crop})
                    else:
                        category_to_prediction_stats[elem] = {'true-positive': 0, 'false-positive': 1, 'false-negative': 0, 'true-negative': 0}
                        category_to_prediction_details[elem] = {'true-positive': [], 'false-positive': [{'filename': filename, 'predicted': predicted_classes_for_crop, 'ground-truth': gt_labels_for_crop}], 'false-negative': [], 'true-negative': []}

            # updating false negatives
            for elem in gt_labels_for_crop:
                if elem not in predicted_classes_for_crop:
                    if elem in category_to_false_negative_counts:
                        category_to_false_negative_counts[elem].append(filename)
                    else:
                        category_to_false_negative_counts[elem] = [filename]

                    if elem in category_to_prediction_stats:
                        category_to_prediction_stats[elem]['false-negative'] += 1
                        category_to_prediction_details[elem]['false-negative'].append({'filename': filename, 'predicted': predicted_classes_for_crop, 'ground-truth': gt_labels_for_crop})
                    else:
                        category_to_prediction_stats[elem] = {'true-positive': 0, 'false-positive': 0, 'false-negative': 1, 'true-negative': 0}
                        category_to_prediction_details[elem] = {'true-positive': [], 'false-positive': [], 'false-negative': [{'filename': filename, 'predicted': predicted_classes_for_crop, 'ground-truth': gt_labels_for_crop}], 'true-negative': []}

            print("{} | Correct percent = {} | Predicted = {} vs. "
                  "Ground Truth = {}:".format(filename, correct_predictions, predicted_classes_for_crop, gt_labels_for_crop))

    # update the global variables
    all_n_incorrect_predictions_to_filenames[conf_level] = n_incorrect_predictions_to_filenames
    all_category_to_true_positive_counts[conf_level] = category_to_true_positive_counts
    all_category_to_false_positive_counts[conf_level] = category_to_false_positive_counts
    all_category_to_false_negative_counts[conf_level] = category_to_false_negative_counts
    # all_category_to_true_negative_counts[confidence_threshold]  |  add true negative here
    all_category_to_prediction_stats[conf_level] = category_to_prediction_stats
    all_category_to_prediction_details[conf_level] = category_to_prediction_details


nc = len(labels_ref_for_run)  # number of classes.

classifier = DinoVisionTransformerClassifier("base", nc)

classifier.load_state_dict(
    torch.load('{}/'.format(local_directory) + model_name, map_location=torch.device(device)))

classifier = classifier.to(device)
classifier.eval()

# runs the inference for all confidence levels
for conf_level in confidence_levels:
    inference_on_validation_data(inference_model=classifier, confidence_level=conf_level)

# output_file_name = dataset_dir_path + '/inference-stats' + '-' + inference_set_dir_name + '-masked.json'
output_file_name = dataset_dir_path + '/inference-stats' + '-' + inference_set_dir_name + '.json'

# save the results to a file
with open(output_file_name, 'w') as f:

    all_stats = {
        'n_incorrect_predictions_to_filenames': all_n_incorrect_predictions_to_filenames,
        'category_to_prediction_stats': all_category_to_prediction_stats,
        'category_to_prediction_details': all_category_to_prediction_details
    }

    f.write(json.dumps(all_stats, indent=4))

    draw_confusion_matrices(all_stats, c12n_category, label_type, dataset_dir_path, inference_set_dir_name)

    for conf_level in all_n_incorrect_predictions_to_filenames:
        for key in all_n_incorrect_predictions_to_filenames[conf_level]:
            print("Number of incorrect predictions: {} | Count: {} | Confidence level: {}".format(key, len(all_n_incorrect_predictions_to_filenames[conf_level][key]), conf_level))

    print("-------------------")
    # print("True Positives:")
    #
    # for category in category_to_true_positive_counts:
    #     print(category + ": " + str(len(category_to_true_positive_counts[category])))
    #
    # print("-------------------")
    # print("False Positives:")
    #
    # for category in category_to_false_positive_counts:
    #     print(category + ": " + str(len(category_to_false_positive_counts[category])))
    #
    # print("-------------------")
    # print("False Negatives:")
    # for category in category_to_false_negative_counts:
    #     print(category + ": " + str(len(category_to_false_negative_counts[category])))
