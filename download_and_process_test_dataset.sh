export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE

# Pinned to the commit where the paper's dataset was published, so later uploads to HuggingFace don't change the results.
DATASET_REV=6e3a116a3c228dd35bcd72f6e5fb921f6ebb6a50

# Downloads Validated/<name>.zip, resuming a partial download, and stops unless its sha256 matches.
fetch() {
  wget -c -O "$1.zip" "https://huggingface.co/datasets/projectsidewalk/sidewalk-tagger-ai-validated/resolve/$DATASET_REV/Validated/$1.zip"
  echo "$2  $1.zip" | shasum -a 256 -c || exit 1
}

fetch Crosswalk 8a80275d9a27736b020fccb904f232dedd6424d3b7ed82042977c9b34b561e74
unzip Crosswalk.zip
mkdir -p datasets/crops-crosswalk-tags/test/
mv Crosswalk/crops/* datasets/crops-crosswalk-tags/test/
mv Crosswalk/csv/test.csv datasets/crops-crosswalk-tags/test/test.csv
rm -rf Crosswalk
rm -rf Crosswalk.zip

fetch SurfaceProblem fe8195a110f3fe2607c7c90aa78e6804cce6fd705cc9fb1abb7d5f1d549f99b3
unzip SurfaceProblem.zip
mkdir -p datasets/crops-surfaceproblem-tags/test/
mv SurfaceProblem/crops/* datasets/crops-surfaceproblem-tags/test/
mv SurfaceProblem/csv/test.csv datasets/crops-surfaceproblem-tags/test/test.csv
rm -rf SurfaceProblem
rm -rf SurfaceProblem.zip

fetch CurbRamp 5a8568353d720084ce4ec170ad9e3cc2b57b8d549bdd5d93099c180b82ed9a2d
unzip CurbRamp.zip
mkdir -p datasets/crops-curbramp-tags/test/
mv CurbRamp/crops/* datasets/crops-curbramp-tags/test/
mv CurbRamp/csv/test.csv datasets/crops-curbramp-tags/test/test.csv
rm -rf CurbRamp
rm -rf CurbRamp.zip

fetch Obstacle 2953039dd7e98a198644236f3fd559c20814b338c52c9b5200b9de963f5fb732
unzip Obstacle.zip
mkdir -p datasets/crops-obstacle-tags/test/
mv Obstacle/crops/* datasets/crops-obstacle-tags/test/
mv Obstacle/csv/test.csv datasets/crops-obstacle-tags/test/test.csv
rm -rf Obstacle
rm -rf Obstacle.zip

python crop.py

