# [Auto Class](https://huggingface.co/docs/transformers/v4.14.1/model_doc/auto)

AutoClasses could help us to automatically retrieve the relevant model (pretrained model) give the path.

the pretraied model path:

* give one example: if we search one DoNut base model from huggingFace. 

* go to https://huggingface.co/models
* type in: donut
* select one provided model: https://huggingface.co/naver-clova-ix/donut-base
* copy the model name

This is the nam path which the AutoClass needs.

```python
from transformers import AutoConfig, AutoModel,


```

