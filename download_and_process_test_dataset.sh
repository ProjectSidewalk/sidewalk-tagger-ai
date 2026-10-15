#!/bin/bash
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE

# Pinned to the commit where the paper's dataset was published, so later uploads to HuggingFace don't change the results.
DATASET_REV=6e3a116a3c228dd35bcd72f6e5fb921f6ebb6a50

# sha256sum is missing on macOS, and shasum on some minimal Linux images.
if command -v sha256sum >/dev/null; then SHA256SUM="sha256sum"; else SHA256SUM="shasum -a 256"; fi

# Downloads Validated/<Name>.zip (resuming a partial download), checks its sha256, and moves its test set into
# datasets/crops-<name>-tags/test/. The images are cropped by crop.py at the end.
fetch_and_extract() {
  local name=$1 sha=$2
  local dir="datasets/crops-$(echo "$name" | tr '[:upper:]' '[:lower:]')-tags/test"
  wget -c -O "$name.zip" "https://huggingface.co/datasets/projectsidewalk/sidewalk-tagger-ai-validated/resolve/$DATASET_REV/Validated/$name.zip"
  if ! echo "$sha  $name.zip" | $SHA256SUM -c; then
    rm -f "$name.zip"
    echo "Deleted $name.zip because its sha256 didn't match. Rerun this script; the test sets extracted so far haven't been cropped yet." >&2
    exit 1
  fi
  unzip -o "$name.zip"
  mkdir -p "$dir"
  mv "$name"/crops/* "$dir"/
  mv "$name/csv/test.csv" "$dir/test.csv"
  rm -rf "$name" "$name.zip"
}

fetch_and_extract Crosswalk 8a80275d9a27736b020fccb904f232dedd6424d3b7ed82042977c9b34b561e74
fetch_and_extract SurfaceProblem fe8195a110f3fe2607c7c90aa78e6804cce6fd705cc9fb1abb7d5f1d549f99b3
fetch_and_extract CurbRamp 5a8568353d720084ce4ec170ad9e3cc2b57b8d549bdd5d93099c180b82ed9a2d
fetch_and_extract Obstacle 2953039dd7e98a198644236f3fd559c20814b338c52c9b5200b9de963f5fb732

python crop.py
