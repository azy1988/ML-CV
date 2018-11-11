---
layout: page
mathjax: true
permalink: /assignments2018/assignment2/
---
这是cs231n的第二次作业，下面是步骤

1. 获取数据

可以按照assignmetn1中的方法，在该目录下

```bash
cd cs231n/datasets/;
bash get_datasets.sh
```

可以得到下面的效果

<img src='https://ws3.sinaimg.cn/large/006tNbRwly1fvbq7y6bm4j30zw0a4gmv.jpg' width='700'>

也可以把 assignmetn1 中 `assignment1/cs231n/datasets/cifar-10-batches-py` 复制到 `assignment2/cs231n/datasets` 中

2. 安装依赖并且打开jupyter notebook

```bash
pip3 install future --user;
cd cs231n;
python setup.py build_ext --inplace
jupyter notebook
```

