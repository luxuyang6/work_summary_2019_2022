# 研究生工作总结

## 基于变分网络与关键词的视频描述生成（Video_PHVM）

### 工作介绍
![image](https://github.com/luxuyang6/work_summary_2019_2022/blob/master/Video_PHVM.png)

### Training & Inference
Python 3 and PyTorch 1.4.
```
cd Video_PHVM
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



## 基于场景图引导与交互的视频描述生成（SGI）

### 工作介绍
![image](https://github.com/luxuyang6/work_summary_2019_2022/blob/master/SGI.jpg)

### Training & Inference
Python 3 and PyTorch 1.4.
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



