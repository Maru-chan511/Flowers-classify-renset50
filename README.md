# Flowers-classify-renset50
 An example that help people to use resnet50 network.
数据来源：这个小项目是一个小白入门深度学习的练手项目，训练数据来源于tf提供的下载链接下载下来的图像，为花卉分类数据，一共有5种花。
数据预处理：每种花我分成了两个数据集，训练数据集和测试数据集，也可以分3部分：训练数据集，验证数据集和测试数据集。训练数据集统一每类是480张，剩余的作为测试数据；
标签处理：按照代码的输入自己写了个lable生成的代码，并生成了json文件；
预训练模型：需要去tensoflow 官网下载，本次代码使用的是resnet_v1_50的模型；

------------------------------------------------------------------------------------
1.先运行DataPrepare.py文件里的run()函数，下载花卉数据集；
2.将下载下来的数据集按照自己的要求分成训练集和测试集；
3.运行DataPrepare.py的LabelWrite()函数，制作label，生成标签json文件；
4.构建网络，model.py文件；
5.运行generate_tfrecord.py文件生成tf.record数据；
6.训练，train.py文件；
7.完成训练后运行export_inference_graph.py文件将模型由ckpt格式导出pb格式；
8.运行predict.py文件进行推理测试。

