# [DoNut](https://huggingface.co/docs/transformers/main/en/model_doc/donut)

The official tutorial notebook could be found here: [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Donut)





## Vision Encoder Decoder Models

VisionEncoderDecoderModel Initializes an image-to-text model with 

1. any pretrained Transformer-based vision model as the encoder (*e.g.* [ViT](https://huggingface.co/docs/transformers/model_doc/vit), [BEiT](https://huggingface.co/docs/transformers/model_doc/beit), [DeiT](https://huggingface.co/docs/transformers/model_doc/deit), [Swin](https://huggingface.co/docs/transformers/model_doc/swin)) 
2. any pretrained language model as the decoder (*e.g.* [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta), [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2), [BERT](https://huggingface.co/docs/transformers/model_doc/bert), [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert)).



Donut is an instance of [VisionEncoderDecoderModel](https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder).

Donut 是一个可以用于图像编码和解码的机器学习模型实例。它是一个 VisionEncoderDecoderModel 类型的对象。

VisionEncoderDecoderModel 是一个类,代表一个图像编码-解码模型。

Donut 是这个类的一个具体实例,一个对象。换句话说,Donut 是一个具体的图像编码-解码模型,它是 VisionEncoderDecoderModel 这个类的一个实例对象。







## VisionEncoderDecoderConfig

https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderConfig

* **VisionEncoderDecoderConfig is the configuration class to *<u>store the configuration</u>* of a VisionEncoderDecoderModel**, defining the encoder and decoder configs.

* The configuration objects can be used to control the model outputs.



### Loading model and config from pretrained folder

```python
from transformers import VisionEncoderDecoderConfig

encoder_decoder_config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")
```

"naver-clova-ix/donut-base" is from: https://huggingface.co/naver-clova-ix/donut-base



## DonutProcessor

https://huggingface.co/docs/transformers/model_doc/donut#transformers.DonutProcessor

* Constructs a Donut processor which wraps a Donut image processor and an XLMRoBERTa tokenizer into a single processor.

* Instantiate a processor associated with a pretrained model.



```python
from transformers import DonutProcessor

# instantiate with our custom processor
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
```



## VisionEncoderDecoderModel

https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel

* Initialize an image-to-text-sequence model.

```python
from transformers import VisionEncoderDecoderModel

model = VisionEncoderDecoderModel.from_pretrained(model_name, config=config)
```





## Prediction

1. 在训练完成后,您可以使用模型进行预测。主要分为以下几步:1. 加载最佳模型权重。您可以从wandb中找到val_loss最低的模型,并下载其权重:

```python
model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()
```

.load_state_dict()和.eval()是PyTorch中常用的两个方法:.load_state_dict()用于加载预训练模型权重。例如,在本例中我们从wandb中找到val_loss最佳的模型权重BEST_MODEL_PATH,并加载到model中。

这样,model的权重将被重置为那个最佳检查点的值。然后,我们可以使用这个模型进行预测和测试。.eval()将模型设置为评估模式。在训练模式下,某些层 (如Dropout) 会随机抛弃一些活动单元的输出。

然而,在测试/推理模式下,我们不希望这些随机效果,所以在评估模式下这些层会默认禁用这种功能。简而言之: .load_state_dict()加载预训练模型权重.

.eval()将模型设置为评估模式,禁用随机dropout等效果.



可能遇到的问题：

```
RuntimeError                              Traceback (most recent call last)
Cell In[56], line 1
----> 1 model.load_state_dict(torch.load(‘/kaggle/working/Donut/hugdnozy/checkpoints/epoch=3-step=2880.ckpt’))


Unexpected key(s) in state_dict: “epoch”, “global_step”, “pytorch-lightning_version”, “state_dict”, “loops”, “callbacks”, “optimizer_states”, “lr_schedulers”, “MixedPrecisionPlugin”.
```

这个RuntimeError表示在加载检查点时出现错误。根据错误信息,有两类问题:1. Missing key(s):缺少某些键。这表示检查点中的某些权重值缺失,无法加载到模型中。2. Unexpected key(s):出现某些意外的键。这表示检查点包含模型不需要的参数,无法正确加载。要修复这个错误,我们有几种方案:1. 忽略缺失和意外的参数,只加载检查点中与模型匹配的参数:

```python
model.load_state_dict(torch.load(ckpt_path), strict=False)
```

strict=False表示在加载过程中忽略错误和缺失的参数。

根据错误信息,在检查点和模型之间手动映射参数:

```python 
ckpt = torch.load(ckpt_path)

# 处理缺失的参数
ckpt["encoder.embeddings.patch_embeddings.projection.weight"] = ...

# 移除意外的参数
del ckpt["epoch"] 
del ckpt["global_step"]
...

model.load_state_dict(ckpt)
```





2. 准备预测数据。这可能需要与训练数据相同的预处理步骤,如分词、填充等。
3. 通过model进行前向传播以得到预测结果。例如:

```python
preds = model(predict_data)
```

4. 可选:根据任务类型后处理预测结果。例如:- 对于分类,取最大概率的类作为预测:

```python
pred_cls = preds.argmax(-1)
```

对于序列标注,取最大概率的标注作为预测:

```python
pred_seq = preds.argmax(-1) 
```

对于生成任务,从word embedding中取出字符序列:

```python
pred_text = tokenizer.sequence_to_text(pred_seq)
```

5. 计算预测评估指标。这与训练时使用的指标相同,例如:

   \- 准确率Accuracy 

   \- F1 Score

   \- 编辑距离ED
