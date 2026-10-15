export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE

# Pinned to the commit where the paper's dataset was published, so later uploads to HuggingFace don't change the results.
DATASET_REV=6e3a116a3c228dd35bcd72f6e5fb921f6ebb6a50

wget https://huggingface.co/datasets/projectsidewalk/sidewalk-tagger-ai-validated/resolve/$DATASET_REV/Validated/Crosswalk.zip
unzip Crosswalk.zip
mkdir -p datasets/crops-crosswalk-tags/test/
mv Crosswalk/crops/* datasets/crops-crosswalk-tags/test/
mv Crosswalk/csv/test.csv datasets/crops-crosswalk-tags/test/test.csv
rm -rf Crosswalk
rm -rf Crosswalk.zip

wget https://huggingface.co/datasets/projectsidewalk/sidewalk-tagger-ai-validated/resolve/$DATASET_REV/Validated/SurfaceProblem.zip
unzip SurfaceProblem.zip
mkdir -p datasets/crops-surfaceproblem-tags/test/
mv SurfaceProblem/crops/* datasets/crops-surfaceproblem-tags/test/
mv SurfaceProblem/csv/test.csv datasets/crops-surfaceproblem-tags/test/test.csv
rm -rf SurfaceProblem
rm -rf SurfaceProblem.zip

wget https://huggingface.co/datasets/projectsidewalk/sidewalk-tagger-ai-validated/resolve/$DATASET_REV/Validated/CurbRamp.zip
unzip CurbRamp.zip
mkdir -p datasets/crops-curbramp-tags/test/
mv CurbRamp/crops/* datasets/crops-curbramp-tags/test/
mv CurbRamp/csv/test.csv datasets/crops-curbramp-tags/test/test.csv
rm -rf CurbRamp
rm -rf CurbRamp.zip

wget https://huggingface.co/datasets/projectsidewalk/sidewalk-tagger-ai-validated/resolve/$DATASET_REV/Validated/Obstacle.zip
unzip Obstacle.zip
mkdir -p datasets/crops-obstacle-tags/test/
mv Obstacle/crops/* datasets/crops-obstacle-tags/test/
mv Obstacle/csv/test.csv datasets/crops-obstacle-tags/test/test.csv
rm -rf Obstacle
rm -rf Obstacle.zip

python crop.py

