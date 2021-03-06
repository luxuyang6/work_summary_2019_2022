# 研究生工作总结

## 基于变分网络与关键词的视频描述生成（Video_PHVM）

### 工作介绍

Video-PHVM方法首先从视频中提取出包含视频中实体、动作等信息的关键词组合，之后将不同的关键词组合用于视频描述生成，具体的模型结构如图所示。

该方法使用LSTM对视频与文本进行编码，为了生成更加多样与可控的视频描述，使用变分网络对输入的关键词组合进行融合处理，得到关键词级别的变分向量，之后使用分层文本生成的方法，使用句子级别变分向量逐句生成内容丰富的视频描述。

该方法基于[Long and Diverse Text Generation with Planning-based Hierarchical Variational Model](https://arxiv.org/abs/1908.06605)中PHVM模型进行实现。
![image](https://github.com/luxuyang6/work_summary_2019_2022/blob/master/Video_PHVM.png)

### 代码实现
详见Video_PHVM目录，基于Python 3 and PyTorch 1.4实现。
```
cd Video_PHVM/src

# training
python main.py --model_name "Video_PHVM" --test_batch_size 16 --test_data "$test_data$" --load_path $load_path$ --mode 'test' 

# inference
python main.py --model_name "Video_PHVM" --batch_size 4 --max_epoch 20 --mode 'train'  --train_file $train_data$ --val_file $val_data$
```



## 基于场景图引导与交互的视频描述生成（SGI）

### 工作介绍
SGI是一种基于场景图引导与交互的视频描述方法。其中，场景图是一种包含实体，关系和属性节点的图结构，
可以同时从视频与文本两种模态中提取。

该方法的研究目的在于使用场景图充分挖掘并表示视频中的关键语义内容与结构信息，并通过准确的场景图来引导模型生成内容更
加真实的视频描述。
该方法主要由两个部分组成：准确场景图生成模块与多模态交
互文本生成模型。

准确场景图生成模块使用预处理的视频场景图与文本场景图数据训
练多个节点分类器，之后直接从视频中生成包含内容与结构信息的准确场景图，为后
续的文本生成器提供引导。

多模态交互文本生成模型将视频特征和上述准确场景图作
为输入，使用交互注意力机制对视频信息、场景图信息、文本信息进行充分交互，使
用流注意力机制对图场景图的构信息进行解码，生成包含真实视频内容的视频描述。
模型结构如图所示：
![image](https://github.com/luxuyang6/work_summary_2019_2022/blob/master/SGI.jpg)

### 代码实现
详见SGI目录，基于Python 3 and PyTorch 1.4实现。

准确场景图生成模块由多个分类器组成，实现代码在SGI目录中给出。
```
cd SGI/
export PYTHONPATH=$(pwd):${PYTHONPATH}
cd src/
export PYTHONPATH=$(pwd):${PYTHONPATH}

cd controlimcap/driver
mtype=rgcn.flow.memory 

# config
python configs/prepare_charades_config.py $mtype # the config file can be changed
resdir='' # copy the output string of the previous step

# training
python main.py $resdir/model.json $resdir/path.json $mtype --eval_loss --is_train --num_workers 8

# inference
python main.py $resdir/model.json $resdir/path.json $mtype --eval_set tst --num_workers 8
```

 


## 相关数据
### 视频描述数据集
#### [ActivityNet Captions](http://activity-net.org/download.html)
* 原始视频百度云链接：https://pan.baidu.com/s/11QEhcx7PCjW_f5MYtPguyg 
提取码:ouj4
* Resnet特征数据见209服务器/home/stockpile/shares/VideoCaptioningDatasets/Activitynet-Captions/features/目录
* 帧图片数据见209服务器/home/stockpile/shares/VideoCaptioningDatasets/Activitynet-Captions/images/目录

#### [Charades Captions](https://prior.allenai.org/projects/charades)
* Resnet特征数据见209服务器/home/stockpile/shares/VideoCaptioningDatasets/Charades/Charades_feature/目录
* 帧图片数据见209服务器/home/stockpile/shares/VideoCaptioningDatasets/Charades/Charades_v1_rgb/目录

### 场景图数据

#### 数据格式
场景图数据格式示例
```json
"Video_ID": {
        "relationships": [
            {"text": ["person","open","bag"],
                "relationship_id": 0,
                "name": "open",
                "subject_id": 0,
                "object_id": 1},
            {"text": ["person","put","shoe"],
                "relationship_id": 1,
                "name": "put",
                "subject_id": 0,
                "object_id": 2}
        ],
        "phrase": "a person opens a plastic bag then begins to put several shoes in it .",
        "objects": [{"object_id": 0,"name": "person","attributes": []},
            {"object_id": 1,"name": "bag","attributes": ["plastic"]},
            {"object_id": 2,"name": "shoe","attributes": ["several"]}]
    }
```
#### 视频场景图
使用场景图预训练模型Motifs-TDE提取，参考https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch

#### 文本场景图
使用基于java的斯坦福场景图提取器提取，参考链接https://nlp.stanford.edu/software/scenegraph-parser.shtml ，文本场景图处理代码详见SceneGraphParser目录，需要配合stanford-corenlp包使用。

## 相关论文与专利
> [1] Xuyang Lu, Yang Gao. Guide and Interact: Scene-graph based Generation and Control
of Video Captions[C]. China Multimedia (ChinaMM2022), 2022: In Press.

> [2] 高扬，陆旭阳. 一种生成视频描述的方法 [P] : 中国, CN202110854988.8, 2022-03-
29



