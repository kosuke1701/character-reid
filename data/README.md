# Image Files

Image files should be indexed by interger (image ID) and saved to a same directory, i.e. `<root_directory>/<index: it should be integer>.(png/jpg)`. (Image IDs are arbitrary as long as each image has unique ID.)

# Data Format

Dataset file is a JSON file with following format:

```
[
  [character_id_1, [image_id_1_1, ..., image_id_1_N]],
  ...,
  [character_id_M, [image_id_M_1, ..., image_id_M_N']]
]
```
