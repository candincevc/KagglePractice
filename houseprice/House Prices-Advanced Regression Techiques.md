### 1 问题描述（Competition Description）
- target : predict the final price of each home
- given features: 79 explanatory variables 

### 2 期望练习的数据挖掘技能（Practice Skills）
- creative feature engineering
- advanced regression techniques: rf and gbt

### 3 参考的Notebook

[Stacked Regressions: Top 4% on LeardBoard](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard)

#### 3.1 data preprocessing

##### 3.1.1 异常值处理

##### 3.1.2 缺失值处理

- object : 空值填充 \ 众数填充

- int/float: 0 填充 \ 中位数填充

##### 3.1.3 数值变换

- 数值型转类别型

- 类别型转数值型
    
    类别型各类大小有意义

- 连续型变量是否服从正态分布

    - BOX 变换
    
    - log 变换

- 类别型变量onehot编码

##### 3.1.4 y变量分析
- 是否服从正态分布——不服从需做log变换

- Q-Q图

##### 3.1.5 相关代码
```python
# data preprocessing 部分运行
# 1 处理训练数据 train.csv
python datapreprocessing.py

# 2 处理测试数据 test.csv
python datapreprocessing.py --orgdata test.csv --outdata test.npz

```

#### 3.2 model training
- 模型训练
    
    - 基础模型训练
        
        lasso、ENet、KRR、GBoost
    
    - 模型融合
    
        AveragingModels
        
        StackingAveragedModels

- 模型代码

```python

# 模型代码

modelTrain.py

# 输入数据
train.csv
test.csv

# 运行
python modelTrain.py

# 运行结果
输出submission.csv 文件

 scores of model 0:0.1314(0.0146)


 scores of model 1:0.1265(0.0100)


 scores of model 2:0.1264(0.0102)


 scores of model 3:0.1259(0.0115)


 scores of model 4:0.1257(0.0111)


 scores of model 5:0.1210(0.0110)


```

### 4 kaggle 相关
- [kaggle API 安装](https://github.com/Kaggle/kaggle-api)
