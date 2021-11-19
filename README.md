### 模型训练
    --checkpoint设置模型的保存地址，--name设置名称，--save_epoch_freq设置保存频率
    --continue_train可以继续训练，默认从同名模型的上一个latest_net_G.pth和latest_net_D.pth
    上继续继续训练,--epoch_count设置继续训练的当前epoch编号
    --phase选择度读取的数据集名称，默认训练为train,测试为test
    --display_freq设置生成可视化训练图片的频率

#### pix2pix_add
    通过使用add_dataset.py,input_nc==2,为每张图添加一个通道的附加信息图来训练

### 模型测试
    --name选择要使用的模型名称，--epoch选择训练的epoch数目，默认为latest最新的一个
    --phase选择度读取的数据集名称，测试一般为test
    --results_dir选择测试完的图片的存放位置
#### 亮度降低
    暂时在dataset中将A*0.6来使原始图的亮度降低，测试模型对全局信息的敏感性

### 模型的设置
    --model可以选择使用cyclegan或者pix2pix的模型
    --input_nc选择输入的通道数目，默认为1，训练pix2pix_add为2
    --output_nc选择输出的通道数目
    --netD选择判别器的类型，--n_layers_D选择判别器的层数
    --netG选择生成器的类型有resnet和unet

### dataset
    mammo300中的数据为但通道的16bit的图
    训练caclegan默认使用unaligned_dataset.py,产生不成对的图片，
    训练pix2pix默认使用aligned_dataset.py，产生成对的图片，也可以用--dataset_mode特别指定训练所使用的dataset
    --dataroot指向的是数据的csv文件的地址，数据集划分为trainA.csv、trainB.csv、testA.csv、testB.csv，csv为
    单列，列名为image_path,内容为所指向图片的相对地址。对于mammo300数据集训练集和测试集为1000：240

### 数据处理
    --preprocess中可以设置resize和crop等操作
    --loadsize设置resize的大小，--crop设置裁片的大小

### util.visualizer
    可视化工具暂时为了方便删去了大部分功能，只保留了生成realA，realB，fakeB等的图片
