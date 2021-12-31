# Object-location-centernet-keras
A simple object localisation algorithm that can also be applied to other datasets.

## TODO
Data augmentation


## Dataset
Toloka WaterMeters dataset, download [here](https://toloka.ai/datasets) <br> put it in the dataset folder and use IO.py to generate image.txt file


## Folder structure
```
${ROOT}
└── src/
    ├── ckpt/
    │   
    ├── dataset/
    │   ├── image.txt
    │   ├── data.tsv
    │   ├── images/
    │   ├── collage/
    │   ├── masks/    
    |
    ├── generator/
    │   ├── data_generator.py
    │   ├── IO.py
    │   ├── utils.py
    │
    │
    ├── model/
    │   ├── DNN.py
    │   ├── losses.py
    |
    ├── results
    ├── demo.py
    ├── param.py
    ├── test.py
    └── train.py
├── README.md 
└── requirements.txt
```


## References

[1] CenterNet: [Objects as Points](https://github.com/xingyizhou/CenterNet) <br>
[2] CenterNet: [Keras Implementation](https://github.com/xuannianz/keras-CenterNet) <br>
