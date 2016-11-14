data : source data
result: model save path
	里面含有3个训练模型结果存储文件

	默认加载模型时用到checkpoint，checkpoint 对应的 cnn_pickleModel1.ckpt-9899

	该类存储方式还有待多加了解熟悉


src: python script

### deepModel.py

	模型基本框架， 以及数据和模型的各种参数，需要调参时调整该脚本里的参数
	数据参数：
		image_size, 
		num_class, 
		num_examples_train, 
		num_examples_eval
	模型参数：
		learning_rate : 可以设置成衰减型
		momentum： 动量
		batch_size：训练 评估 测试 都必须是这个batch_size
	权重参数:
		初始值： 初始值不能过小
		wd: 权重衰减参数, 其实是L2规范化参数


### train.py

	max_epochs: 最大训练epochs
	提前推出机制参数 


