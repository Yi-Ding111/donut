# Datasets (package) 

https://huggingface.co/docs/datasets/index

Datasets is a library for easily accessing and sharing datasets for 

* **Audio**
* **Computer Vision**, 
* **Natural Language Processing (NLP)**



## Images



### Use local data to create Dataset

https://huggingface.co/docs/datasets/create_dataset

```python
# read file paths
annotations_dir=Path("/kaggle/input/benetech-making-graphs-accessible/train/annotations")
images_dir=Path("/kaggle/input/benetech-making-graphs-accessible/train/images")
annotations_files=annotations_dir.glob("*.json")
```

 now we need to generate two dictionaries, for creating Dataset.

```python
# return an iterator
all_annotations = list(annotations_dir.iterdir())

# write all json file content into a list
json_list = []
for path in all_annotations:
    with open(path, 'r') as file:
        json_str = file.read()
    json_obj = json.dumps(json_str)
    json_list.append(json_obj) 
```

next, we need to get all json files' names **in order**, the same order with above.

```python
# grab all file names in order
# 这是为了对图像文件的路径进行排序，在生成dataset时候能保持一一对应
annotation_filenames = [f.name for f in annotations_files]
```

In order to make the picture path with the same ID consistent with the above json reading order.

为了让有相同ID的图片路径和上面的json读取顺序一致.

```python
image_paths = []
for filename in annotation_filenames:
    image_path = '/kaggle/input/benetech-making-graphs-accessible/train/images'+'/'+ filename.replace(".json", ".jpg")
    image_paths.append(image_path)
```

And then we could create the Dataset to combine the json file(ground truth) and image together.

```python
from datasets import Dataset
from datasets import Image as ds_img
from datasets import load_dataset
```

load a dataset from the image path: https://huggingface.co/docs/datasets/image_load#local-files

create a dataset from local files by specifying the path to the data files:https://huggingface.co/docs/datasets/image_load#local-files

```python
# generate Dataset
ds = Dataset.from_dict({"ground_truth": json_list, "image": image_paths}).cast_column("image", ds_image())
```







## Natural Language







## Audio







