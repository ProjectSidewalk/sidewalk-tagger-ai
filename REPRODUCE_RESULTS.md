# Reproduce Results
This is a guide for how to reproduce the results published in the paper.
## 1. Download the pretrained models
```bash
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
mkdir notebooks/models
cd notebooks/models
alias wget='wget --content-disposition'  # Makes sure that file names don't include "?download=true" at the end.
wget https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models/resolve/main/unvalidated-dino-cls-b-crosswalk-tags-best.pth?download=true
wget https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models/resolve/main/unvalidated-dino-cls-b-curbramp-tags-best.pth?download=true
wget https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models/resolve/main/unvalidated-dino-cls-b-obstacle-tags-best.pth?download=true
wget https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models/resolve/main/unvalidated-dino-cls-b-surfaceproblem-tags-best.pth?download=true
wget https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models/resolve/main/validated-clip-cls-b-crosswalk-tags-best.pth?download=true
wget https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models/resolve/main/validated-clip-cls-b-curbramp-tags-best.pth?download=true
wget https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models/resolve/main/validated-clip-cls-b-obstacle-tags-best.pth?download=true
wget https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models/resolve/main/validated-clip-cls-b-surfaceproblem-tags-best.pth?download=true
wget https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models/resolve/main/validated-dino-cls-b-crosswalk-tags-best.pth?download=true
wget https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models/resolve/main/validated-dino-cls-b-curbramp-tags-best.pth?download=true
wget https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models/resolve/main/validated-dino-cls-b-obstacle-tags-best.pth?download=true
wget https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models/resolve/main/validated-dino-cls-b-surfaceproblem-tags-best.pth?download=true
cd ../..
```
## 2. Setup Conda Environment
Make sure your system has NVIDIA drivers and CUDA installed and then run the commands below. If you're on a Mac, you'll likely need to remove the `xformers` and `cuml-cu11` lines from `requirements.txt`.
```bash
conda create -n sidewalk-tagger-ai python=3.10
conda activate sidewalk-tagger-ai
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install xformers --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib timm
```
## 3. Download & Preprocess Test Dataset
Make sure you are still in the conda environment that we just created.
```bash
chmod +x download_and_process_test_dataset.sh
./download_and_process_test_dataset.sh
```
This can take a long time depending on your system specs.
## 4. Test
Make sure you are still in the conda environment that we created.

Run the test script with the desired label type, model, and dataset type:
```bash
cd notebooks
python evaluate.py --label-type crosswalk --model DINO --dataset-type validated
```

All arguments have defaults (`--label-type crosswalk`, `--model DINO`, `--dataset-type validated`), so you can omit any you don't need to change. You can also pass `--min-instances` (default 10) to control how many ground-truth positives a tag needs in order to appear in plots and aggregate metrics.

Results are written to `results/<label-type>/`.

![image](https://github.com/user-attachments/assets/e3c16ce4-ffb2-44a6-ab6d-e44928011dac)

You can repeat this for each label type / model / dataset-type combination you want to evaluate.

## 5. Generate precision-vs-threshold graph
After running `evaluate.py`, generate the precision-vs-threshold plot and threshold CSV:
```bash
cd notebooks
python analyze_thresholds.py --label-type crosswalk --target-precision 0.92
```

This reads `results/<label-type>/validated-dino-inference-stats.json` by default and writes a PNG and CSV to the same directory. Pass `--stats-file`, `--output-plot`, and `--output-csv` to override specific paths.
