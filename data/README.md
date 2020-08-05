# Data
## Image Files

Image files should be indexed by interger (image ID) and saved to a same directory, i.e. `<root_directory>/<index: it should be integer>.(png/jpg)`. (Image IDs are arbitrary as long as each image has unique ID.)

## Data Format

Dataset file is a JSON file with following format:

```
[data_1, ..., data_K]
```

where each `data_k` is a split of data with following structure:

```
data = [
  [character_id_1, [image_id_1_1, ..., image_id_1_N]],
  ...,
  [character_id_M, [image_id_M_1, ..., image_id_M_N']]
]
```

# Preprocessing

## Detecting Face in Images

```
python preprocess_face_detection.py --image-root <directory> \
    --dataset <filename> [--cuda]
```

Output preprocessed dataset will be saved in `<dataset filename>.faceBB`.

### Install Dependencies

```
# Please change CUDA version according to your environment.
pip3 install pycocotools numpy==1.16 opencv-python tqdm tensorboard tensorboardX pyyaml webcolors torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# Clone EfficientDet implementation.
# Enter the directory name where this repository is cloned
# in "EfficientDetDirectory" entry in ../config.ini
git clone https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch.git

# Download pretrained face detection weight.
# Enter the filename of downloaded weight in "EfficientDetWeight" entry
# in ../config.ini
wget https://github.com/kosuke1701/pretrained-efficientdet-character-head/releases/download/0.0/character-face-efficientdet-c2.pth
```

### Output Format

Each data split in the original dataset will be converted into:

```
new_data = {
  image_id: list_of_face_positions, ...
}
```

'list_of_face_positions' is a list of bounding box coordinates which are `[x_min, y_min, x_max, y_max]`. (X-axis is horizontal, and (0,0) is top-left corner.)