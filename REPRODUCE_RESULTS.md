# Reproduce Results
This is a guide for how to reproduce the results published in the paper.
## 1. Download the pretrained models
```bash
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
mkdir notebooks/models
cd notebooks/models
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
Make sure your system has NVIDIA drivers and CUDA installed and then run these commands:
```bash
conda create -n sidewalk-tagger-ai python=3.10
conda activate sidewalk-tagger-ai
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install xformers --index-url https://download.pytorch.org/whl/cu118
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

In the `notebooks/test.py` file, there is a section called `params`.

```python
params = {
    'label_type': 'crosswalk', # 'crosswalk' 'curbramp' 'surfaceproblem' 'obstacle'
    'pretrained_model_prefix': MODEL_PREFIXES['DINO'], # 'DINO' or 'CLIP'
    'dataset_type': 'validated', # 'unvalidated' or 'validated'
    # ...
}
```

You should change these options to whatever you want to test. 

Now, we just run the test script!
```bash
cd notebooks
python test.py
```

Results are visible here:

![image](https://github.com/user-attachments/assets/52d19021-00c0-454a-aced-1cf15f9feaee)
![image](https://github.com/user-attachments/assets/e3c16ce4-ffb2-44a6-ab6d-e44928011dac)

You can repeat this process of specifying the params and running the test script until you are satisfied.
