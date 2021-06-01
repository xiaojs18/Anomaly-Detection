运行文件：
preprocess.py 预处理，训练数据转h5
train.py 训练文件
test.py 测试文件，输出视频打分，针对图片
test_video.py 针对视频
test_ablation.py 注意力层消融实验
test_visual.py 特征图可视化

数据集：
trainset 训练集及gt
testset_33 测试集及gt

必需文件：
config.yml 配置文件
model.py 主干网络搭建
sp_layer.py 注意力机制、包级池化
sp_loss.py 损失函数
