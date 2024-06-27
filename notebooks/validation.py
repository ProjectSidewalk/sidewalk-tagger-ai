import json
import shutil

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
from sklearn.metrics import precision_recall_curve, auc, PrecisionRecallDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, RocCurveDisplay
from sklearn.metrics import average_precision_score, accuracy_score, hamming_loss, f1_score
import timm

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


class CLIP_Classifier(nn.Module):
    def __init__(self, model_name='', n_target_classes=7):
        super().__init__()
        self.model = timm.create_model(model_name=model_name, pretrained=True, in_chans=3)
        #         Replace the final head layers in model with our own Linear layer
        num_features = self.model.num_features
        self.model.head = nn.Linear(num_features, 256)
        self.fully_connect = nn.Sequential(nn.Linear(256, 128),
                                           nn.ReLU(),
                                           nn.Linear(128, n_target_classes))

        # self.model.load_state_dict(torch.load('{}/'.format(local_directory) + model_name, map_location=torch.device(device)))

        # self.transformer = deepcopy(self.model)

    def forward(self, image):
        x = self.model(image)
        # Using dropout functions to randomly shutdown some of the nodes in hidden layers to prevent overfitting.
        #         x = self.dropout(x)
        #         # Concatenate the metadata into the results.
        x = self.fully_connect(x)
        return x


def get_labels_ref_for_run(inference_set_dir):
    csv_file_path = os.path.join(inference_set_dir, 'test.csv')
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

    # for the CLIP model we don't have the newly added tags e.g. mailbox, seating etc.
    # but for the DINO model, trained on the validated data, we do have them in the training data.
    # we need to adjust the labels_ref_for_run for the CLIP model
    if params['label_type'] == 'obstacle' and params['pretrained_model_prefix'] == MODEL_PREFIXES['CLIP']:
        if len(labels_ref_for_run) == 20:
            labels_ref_for_run = labels_ref_for_run[:-3]
        else:
            raise ValueError('Unexpected number of labels for obstacle')

    return labels_ref_for_run


# enum to track the classification categories
C12N_CATEGORIES = {
    'TAGS': 'tags',
    'SEVERITY': 'severity',
}

MODEL_PREFIXES = {
    'CLIP': 'clip',
    'DINO': 'dino',
}

# ------------------------------
# all the parameters to be customized for the run
# ------------------------------

params = {
    'label_type': 'obstacle',
    'pretrained_model_prefix': MODEL_PREFIXES['DINO'],
    'dataset_type': 'validated',  # 'unvalidated' or 'validated'

    # these don't really change for now
    'c12n_category': C12N_CATEGORIES['TAGS'],
    'inference_set_dir_name': 'test',
}

# ------------------------------

# suppress the tags that have less than the threshold count in the plot
suppress_thresholds = {
    'crosswalk': 10,
    'obstacle': 10,
    'surfaceproblem': 10,
    'curbramp': 10
}

image_dimension = 256

if params['pretrained_model_prefix'] == MODEL_PREFIXES['CLIP']:
    image_dimension = 224

# This is what DinoV2 sees
target_size = (image_dimension, image_dimension)

# for temporarily skipping some cities
skip_cities = []
# skip_cities = ['oradell', 'walla_walla', 'cdmx', 'spgg', 'chicago', 'amsterdam', 'columbus', 'newberg']

dataset_dirname = 'crops-' + params['label_type'] + '-' + params['c12n_category']  # example: crops-surfaceproblem-tags-archive
# dataset_dirname = 'crops-' + label_type + '-' + c12n_category + '-validated'  # example: crops-surfaceproblem-tags-archive
dataset_dir_path = '../datasets/' + dataset_dirname  # example: ../datasets/crops-surfaceproblem-tags-archive

inference_dataset_dir = Path(dataset_dir_path + "/" + params['inference_set_dir_name'])

# top tp, fp, fn, tn images are saved here
inference_results_dir = Path("../inference-results")

model_name = 'models/' + params['dataset_type'] + '-' + params['pretrained_model_prefix'] + '-cls-b-' + params['label_type'] + '-' + params['c12n_category'] + '-best.pth'
# ------------------------------

local_directory = os.getcwd()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if torch.cuda.is_available():
    print("GPU available")
else:
    print("GPU not available")


if params['pretrained_model_prefix'] == MODEL_PREFIXES['CLIP']:
    img_resize_multiple = 32
else:
    img_resize_multiple = 14

data_transforms = {
    "inference": transforms.Compose([ResizeAndPad(target_size, img_resize_multiple),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                     ])
}

c12n_category_offset = 8


# it's okay if the csv contains more filenames than the images in the directory
# we will only load the images that are present in the directory and query the csv for labels
def images_loader(dir_path, batch_size, imgsz, transform):
    file_path = os.path.join(dir_path, 'test.csv')
    label_data = pd.read_csv(file_path)

    filenames = []
    images = []
    labels = []

    fs = os.listdir(dir_path)
    # ignore the csv file and .DS_Store (if present). this is just the list of images.
    fs = [x for x in fs if x.endswith('.png') or x.endswith('.jpg')]
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

# todo this needs a better name!
labels_ref_for_run = get_labels_ref_for_run(inference_dataset_dir)


all_tag_to_prediction_stats = {}
all_tag_to_prediction_details = {}

images_and_labels = data_loader(inference_dataset_dir, 1, image_dimension, 'inference')


# -----------------------------------------------------------------

def copy_top_instances_to_results_dir(tag, tp_filenames_and_conf, fp_filenames_and_conf, fn_filenames_and_conf, tn_filenames_and_conf):
    def copy_files(filenames_and_conf, inference_dataset_dir, tag_dir_path, category):
        conf_truncate_length = 5
        for i, (fn, conf) in enumerate(filenames_and_conf):
            src_file_path = os.path.join(inference_dataset_dir, fn)
            truncated_conf = str(conf)[:conf_truncate_length]
            dst_file_name = f'{fn.replace(".png", "")}-{truncated_conf}.png'
            dst_file_path = os.path.join(tag_dir_path, category, dst_file_name)
            shutil.copy2(src_file_path, dst_file_path)

    # create a directory for the label type if it doesn't exist
    os.makedirs(os.path.join(inference_results_dir, params['label_type']), exist_ok=True)

    # create directory for model and dataset type if it doesn't exist
    model_and_dataset_dir = os.path.join(inference_results_dir, params['label_type'], params['pretrained_model_prefix'] + '-' + params['dataset_type'])
    os.makedirs(model_and_dataset_dir, exist_ok=True)

    # create a directory for the tag if it doesn't exist
    tag_dir_path = os.path.join(model_and_dataset_dir, tag)
    os.makedirs(os.path.join(model_and_dataset_dir, tag), exist_ok=True)

    # create directories for tp, fp, fn, tn if they don't exist
    os.makedirs(os.path.join(tag_dir_path, 'tp'), exist_ok=True)
    os.makedirs(os.path.join(tag_dir_path, 'fp'), exist_ok=True)
    os.makedirs(os.path.join(tag_dir_path, 'fn'), exist_ok=True)
    os.makedirs(os.path.join(tag_dir_path, 'tn'), exist_ok=True)

    # clear the directories
    for dir_name in ['tp', 'fp', 'fn', 'tn']:
        for filename in os.listdir(os.path.join(tag_dir_path, dir_name)):
            os.remove(os.path.join(tag_dir_path, dir_name, filename))

    # copy the top instances to the inference-results directory
    copy_files(tp_filenames_and_conf, inference_dataset_dir, tag_dir_path, 'tp')
    copy_files(fp_filenames_and_conf, inference_dataset_dir, tag_dir_path, 'fp')
    copy_files(fn_filenames_and_conf, inference_dataset_dir, tag_dir_path, 'fn')
    copy_files(tn_filenames_and_conf, inference_dataset_dir, tag_dir_path, 'tn')


def check_for_mutual_exclusivity_and_total(tp_set, fp_set, fn_set, tn_set, images_and_labels):
    if len(tp_set.intersection(fp_set)) > 0:
        raise ValueError('TP and FP sets are not mutually exclusive')
    if len(tp_set.intersection(fn_set)) > 0:
        raise ValueError('TP and FN sets are not mutually exclusive')
    if len(tp_set.intersection(tn_set)) > 0:
        raise ValueError('TP and TN sets are not mutually exclusive')
    if len(fp_set.intersection(fn_set)) > 0:
        raise ValueError('FP and FN sets are not mutually exclusive')
    if len(fp_set.intersection(tn_set)) > 0:
        raise ValueError('FP and TN sets are not mutually exclusive')
    if len(fn_set.intersection(tn_set)) > 0:
        raise ValueError('FN and TN sets are not mutually exclusive')

    if len(tp_set.union(fp_set).union(fn_set).union(tn_set)) != len(images_and_labels):
        raise ValueError('The total of sets doesn\'t match the number of instances in the dataset')


def inference_on_validation_data(inference_model):

    y_true = []
    y_pred = []

    for idx in range(len(images_and_labels)):

        img_label_filename = images_and_labels[idx]

        img_tensor, labels, filename = img_label_filename

        print('Processing image. Index: {}, Filename: {}'.format(idx, filename))

        input_tensor = img_tensor.to(device)
        labels_tensor = labels.to(device)

        y_true.append(labels.tolist()[0])

        # run model on input image data
        with torch.no_grad():
            if params['pretrained_model_prefix'] == MODEL_PREFIXES['CLIP']:
                output_tensor = inference_model(input_tensor)
            else:
                embeddings = inference_model.transformer(input_tensor)
                x = inference_model.transformer.norm(embeddings)
                output_tensor = inference_model.classifier(x)

            # Convert outputs to probabilities using sigmoid
            if params['c12n_category'] == C12N_CATEGORIES['SEVERITY']:
                probabilities = torch.softmax(output_tensor, dim=1)
            elif params['c12n_category'] == C12N_CATEGORIES['TAGS']:
                probabilities = torch.sigmoid(output_tensor)

            y_pred.append(probabilities.tolist()[0])


    # IMPORTANT variables
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    # Convert predicted probabilities to binary predictions
    y_pred_binary = (y_pred_np >= 0.5).astype(int)

    # Compute micro-averaged F1 score
    f1_micro = f1_score(y_true_np, y_pred_binary, average='micro')
    f1_macro = f1_score(y_true_np, y_pred_binary, average='macro')
    f1_weighted = f1_score(y_true_np, y_pred_binary, average='weighted')

    # Create a list of tuples (tag, precision, recall, n_instances)
    # sum the columns of y_true_np to get the number of instances in the ground truth labels
    tag_to_n_instances = [(labels_ref_for_run[i], int(np.sum(y_true_np[:, i])), i) for i in range(len(labels_ref_for_run))]

    # Sort the list based on n_instances
    tag_to_n_instances.sort(key=lambda x: x[1], reverse=True)

    fig, ax1 = plt.subplots(1, 1, figsize=(16, 10))

    tags_not_plotted = []

    all_average_precisions = []

    for i in range(len(tag_to_n_instances)):

        tag_name, n_instances, tag_idx = tag_to_n_instances[i]

        sum_gt = np.sum(y_true_np[:, tag_idx])
        if sum_gt != n_instances:
            # throw an error and stop
            raise ValueError('What is happening! For tag {} sum of instances: {} and n_instances: {} are not equal'.format(tag_name, sum_gt, n_instances))

        # compute precision, recall, thresholds
        precision, recall, thresholds_pr = precision_recall_curve(y_true_np[:, tag_idx], y_pred_np[:, tag_idx], pos_label=1)
        pr_auc = auc(recall, precision)

        average_precision_val = average_precision_score(y_true_np[:, tag_idx], y_pred_np[:, tag_idx], average='weighted')

        all_f1_pr = 2 * precision * recall / (precision + recall)
        ix_pr = np.argmax(all_f1_pr)
        best_thresh_pr = thresholds_pr[ix_pr]
        precision_pr_at_best_conf = precision[ix_pr]
        recall_pr_at_best_conf = recall[ix_pr]
        f1_pr_at_best_conf = all_f1_pr[ix_pr] if not np.isnan(all_f1_pr[ix_pr]) else 0

        y_pred_class_pr = np.where(y_pred_np[:, tag_idx] > best_thresh_pr, 1, 0)

        all_tag_to_prediction_stats[tag_name] = {'n_instances': n_instances, 'precision': precision.tolist(), 'recall': recall.tolist(),
                                                               'thresholds': thresholds_pr.tolist(), 'pr_auc': pr_auc, 'average_precision_val': average_precision_val}

        if len(np.unique(y_true_np[:, tag_idx])) < 2 or len(np.unique(y_pred_np[:, tag_idx])) < 2:
            print('For tag {} all instances of y_true: {}'.format(tag_name, np.unique(y_true_np[:, tag_idx])))


        # don't plot if there are no instances of the tag in the ground truth labels
        st = suppress_thresholds[params['label_type']]
        if params['label_type'] not in suppress_thresholds:
            st = 0

        if n_instances < st:
            tags_not_plotted.append((tag_name, n_instances))
            continue

        # note: this should be done after the suppression part
        all_average_precisions.append(average_precision_val)

        # Create a PrecisionRecallDisplay and plot it on the same axis
        # pr_display = PrecisionRecallDisplay(precision=precision, recall=recall).plot(ax=ax1, name=tag_name + ' (n={})'.format(n_instances))
        pr_display = PrecisionRecallDisplay(precision=precision, recall=recall).plot(ax=ax1, name=tag_name + '\n(n={}, AUC={}, AP={})\n(conf={}, f1={})\n(prec={}, rec={})'.format(n_instances, round(pr_auc, 2), round(average_precision_val, 2), round(best_thresh_pr, 2), round(f1_pr_at_best_conf, 2), round(precision_pr_at_best_conf, 2), round(recall_pr_at_best_conf, 2)))


        # Create a RocCurveDisplay and plot it on the same axis
        # roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax2, name=tag_name + '\n(n={}, AUC={})\n(conf={}, acc={}, f1={}\n(prec={}, rec={}))'.format(n_instances, round(roc_auc, 2), round(best_thresh_roc, 2), round(accuracy_score_roc, 2), round(f1_score_roc, 2), round(precision_roc_at_best_conf, 2), round(recall_roc_at_best_conf, 2)))

        tp_indices_for_tag = np.where((y_true_np[:, tag_idx] == 1) & (y_pred_class_pr == 1))[0]
        fp_indices_for_tag = np.where((y_true_np[:, tag_idx] == 0) & (y_pred_class_pr == 1))[0]
        fn_indices_for_tag = np.where((y_true_np[:, tag_idx] == 1) & (y_pred_class_pr == 0))[0]
        tn_indices_for_tag = np.where((y_true_np[:, tag_idx] == 0) & (y_pred_class_pr == 0))[0]

        tp_filenames_and_conf = [(images_and_labels[i][2], y_pred_np[:, tag_idx][i]) for i in tp_indices_for_tag]
        fp_filenames_and_conf = [(images_and_labels[i][2], y_pred_np[:, tag_idx][i]) for i in fp_indices_for_tag]
        fn_filenames_and_conf = [(images_and_labels[i][2], y_pred_np[:, tag_idx][i]) for i in fn_indices_for_tag]
        tn_filenames_and_conf = [(images_and_labels[i][2], y_pred_np[:, tag_idx][i]) for i in tn_indices_for_tag]

        # check if all these sets are mutually exclusive
        check_for_mutual_exclusivity_and_total(set(tp_indices_for_tag), set(fp_indices_for_tag), set(fn_indices_for_tag), set(tn_indices_for_tag), images_and_labels)

        # sort the lists by confidence level in descending order
        tp_filenames_and_conf.sort(key=lambda x: x[1], reverse=True)
        fp_filenames_and_conf.sort(key=lambda x: x[1], reverse=True)
        fn_filenames_and_conf.sort(key=lambda x: x[1], reverse=True)
        tn_filenames_and_conf.sort(key=lambda x: x[1], reverse=True)

        # get the top N instances for each set
        N_top_instances = 50
        tp_filenames_and_conf = tp_filenames_and_conf[:N_top_instances]
        fp_filenames_and_conf = fp_filenames_and_conf[:N_top_instances]
        fn_filenames_and_conf = fn_filenames_and_conf[:N_top_instances]
        tn_filenames_and_conf = tn_filenames_and_conf[:N_top_instances]

        # copy the crops to the results directory
        copy_top_instances_to_results_dir(tag_name, tp_filenames_and_conf, fp_filenames_and_conf, fn_filenames_and_conf, tn_filenames_and_conf)

        all_tag_to_prediction_details[tag_name] = {'tp': tp_filenames_and_conf, 'fp': fp_filenames_and_conf, 'fn': fn_filenames_and_conf}

    mean_average_precision = sum(all_average_precisions) / len(all_average_precisions)

    # Add a legend to the plot
    legend1 = ax1.legend(title='Classes', fontsize='14', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add a legend to the ROC plot
    # ax2.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    #
    # # Set titles for the plots
    ax1.set_title('Precision-Recall Curve')
    # ax2.set_title('ROC Curve')

    # Set the plot title
    plot_title_str = 'PR and ROC Curves for label type: ' + params['label_type']
    plot_title_str += '\nTest set size: ' + str(len(images_and_labels)) + ' images'
    plot_title_str += '\nModel: ' + ('ViT CLIP Base' if params['pretrained_model_prefix'] == MODEL_PREFIXES['CLIP'] else 'DINOv2 Base')
    plot_title_str += ' | Train dataset: ' + params['dataset_type']

    plot_title_str += ('\nmAP: ' + str(round(mean_average_precision, 2)) +
                       ' | ' + 'Micro F1: ' + str(round(f1_micro, 2)) +
                       ' | ' + 'Macro F1: ' + str(round(f1_macro, 2)) +
                       ' | ' + 'Weighted F1: ' + str(round(f1_weighted, 2)))

    # Set title for the figure and save
    plt.suptitle(plot_title_str, fontsize=16)

    model_display_name = 'ViT CLIP Base' if params['pretrained_model_prefix'] == MODEL_PREFIXES['CLIP'] else 'DINOv2 Base'

    plt.tight_layout()

    pt_model_prefix = params['pretrained_model_prefix']
    inf_set_dir_name = params['inference_set_dir_name']
    dataset_type = params['dataset_type']

    if suppress_thresholds[params['label_type']] > 0:
        plt.savefig(f'{dataset_dir_path}/{dataset_type}-{pt_model_prefix}-pr-curve-{inf_set_dir_name}.svg')
    else:
        plt.savefig(f'{dataset_dir_path}/{dataset_type}-{pt_model_prefix}-pr-curve-{inf_set_dir_name}-all.svg')

    plt.show()


nc = len(labels_ref_for_run)  # number of classes.

if params['pretrained_model_prefix'] == MODEL_PREFIXES['CLIP']:
    classifier = CLIP_Classifier("vit_base_patch16_clip_224", nc)
else:
    classifier = DinoVisionTransformerClassifier("base", nc)

classifier.load_state_dict(torch.load('{}/'.format(local_directory) + model_name, map_location=torch.device(device)))

classifier = classifier.to(device)
classifier.eval()

# runs the inference for all confidence levels

inference_on_validation_data(inference_model=classifier)

output_file_name = dataset_dir_path + '/' + params['dataset_type'] + '-' + params['pretrained_model_prefix'] + '-inference-stats' + '-' + params['inference_set_dir_name'] + '.json'

# save the results to a file
with open(output_file_name, 'w') as f:

    all_stats = {
        # 'n_incorrect_predictions_to_filenames': all_n_incorrect_predictions_to_filenames,
        'category_to_prediction_stats': all_tag_to_prediction_stats,
        # 'category_to_prediction_details': all_category_to_prediction_details
    }

    f.write(json.dumps(all_stats, indent=4))

    print("-------------------")
