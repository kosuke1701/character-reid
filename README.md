# Character Re-identification

This project contains tools and experimental codes to identify characters depicted in illustrations.

<!-- * pytorch-metric-learning
  - https://github.com/KevinMusgrave/pytorch-metric-learning -->

## Getting Started

### Prepare Environment

1. Install [Anaconda](https://www.anaconda.com/products/individual).

2. Create a new environment and install libraries.

    ```
    conda create -n CRID python=3.7 numpy=1.18 scikit-learn=0.22 pillow=7.0 -y
    conda activate CRID
    conda install -c anaconda pyqt -y
    ```

3. Install [PyTorch](https://pytorch.org/).

    ```
    # In case you want to use only CPU.
    conda install pytorch=1.4 torchvision cpuonly -c pytorch -y
    ```

4. Prepare plugins.

    Please see [Plugins section](#plugins).

### GUI Tool

You can open GUI window by executing `app/gui_tool.py`.

```
conda activate CRID
python app/gui_tool.py
```

Currently, I'm preparing user manual for GUI tool.

### CUI Tool

You can test trained models by running following command in `app` directory.

```
python cui_tool.py --plugin-dir <plugin-directory> --gpu <gpu-id> {command} [[additional command specific options]]
```

* `plugin-directory` is a directory which contains model definition and input configurations. See `Plugins` subsection for more details.
* If `gpu-id >= 0`, the program will use GPU with given index. If `gpu-id < 0`, it will use CPU.

Currently available commands are:

* `similarity`
  - Output similarity score of given two images.
  - Additional option:
    - `--files <image filepath 1> <image filepath 2>`
* `classify`
  - Classify drawn characters in image files. Reference images are used to define characters.
  - Additional options:
    - `--enroll-config-fn <JSON file>`
      - JSON file which specifies directories of reference images for each character.
      - (ex.) `{"Yang_Wenli": "reference_files/yang_wenli", "Reinhard": "reference_files/reinhard"}`
    - `--target-dir <directory name>`
      - Directory which contains image files to be classified.
    - `--mode [Any/Avg]`
      - If this option is `Any`, each image will be classified as a character with a reference image with maximum similarity with the input image.
      - If this option is `Avg`, average similarity between input image and reference images of a specific character is computed as a score for the character, and input images are classified as a character with the highest score.
    - `--result-dir <directory name>`
      - Classified image will be copied into `<directory name>/<character name specified in the JSON file>`.
  - Note: Classification with too many reference images (e.g. 1000 for each character) will take long time due to I/O time to load images.

### Plugins

Currently following plugins are available:

* `plugins/res18_single`
  - It uses ResNet18 and one linear projection layer to encode images.
  - Please download pre-trained model file `res18_0222.mdl` from [Google drive link](https://drive.google.com/open?id=1KQziuxDo35ziMz9LGUHm3bCd0LxyB1_-) and place it in `plugins/res18_single` directory.
