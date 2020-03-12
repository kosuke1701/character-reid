# Character Re-identification

This project contains tools and experimental codes to identify characters depicted in illustrations.

## Dependency

Codes in this repository are developed for Python 3 and uses following libraries:

* Pillow (PIL)
* numpy
* scipy
* sci-kit learn
* pytorch
  - torchvision
* pytorch-metric-learning
  - https://github.com/KevinMusgrave/pytorch-metric-learning

# Getting Started

Currently only CUI tool is available.

Trained models used by the tool can be downloaded from [this Google drive link](https://drive.google.com/open?id=1KQziuxDo35ziMz9LGUHm3bCd0LxyB1_-).

### CUI Tool

You can test trained models by running following command in `app` directory.

```
python cui_tool.py --plugin-dir <plugin-directory> --gpu <gpu-id> {command} [[additional command specific options]]
```

* `plugin-directory` is a directory which contains model definition and input configurations. See `Plugins` subsection for more details.
* If `gpu-id >= 0`, the program will use GPU with given index. If `gpu-id < 0`, it will use CPU.

Currently available command is:

* `similarity`
  - Output similarity score of given two images.
  - Additional option:
    - `--files <image filepath 1> <image filepath 2>`

### Plugins

Currently following plugins are available:

* `plugins/res18_single`
  - It uses ResNet18 and one linear projection layer to encode images.
  - Please download pre-trained model file `res18_0222.mdl` from the Google drive link and place it in `plugins/res18_single` directory.
