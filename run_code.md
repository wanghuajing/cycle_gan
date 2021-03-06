### 训练pix2pix
    --dataroot
    /home/zhao/mydata/datasets/mammo300/
    --name
    pix2pix_res9
    --model
    pix2pix
    --input_nc
    1
    --output_nc
    1
    --netD
    n_layers
    --n_layers_D
    6
    --netG
    resnet_9blocks
    --preprocess
    crop
    --no_flip
    --crop_size
    512
    --batch_size
    4
    
### 训练生成256*256分辨率的图片参数
	--dataroot
	/home/zhao/mydata/datasets/mammo300/
	--name
	pix2pix_256
	--model
	pix2pix_256
	--input_nc
	1
	--output_nc
	1
	--netD
	n_layers
	--n_layers_D
	6
	--netG
	resnet_9blocks
	--preprocess
	resize
	--no_flip
	--crop_size
	256
	--batch_size
	1

### 利用256的图片作为补充信息训练
	--dataroot
	/home/zhao/mydata/datasets/mammo300/
	--name
	pix2pix_add
	--model
	pix2pix
	--input_nc
	2
	--output_nc
	1
	--netD
	n_layers
	--n_layers_D
	6
	--netG
	resnet_9blocks
	--preprocess
	crop
	--no_flip
	--crop_size
	512
	--batch_size
	1
	--dataset_mode
	add

### 利用裁片进行pix2pix_add模型的训练
    --dataroot
    /media/zhao/HD1/data/mammo300/all/crop/patch_512x512_o256/
    --name
    pix2pix_add_unet256
    --model
    pix2pix
    --input_nc
    2
    --output_nc
    1
    --netD
    n_layers
    --n_layers_D
    3
    --netG
    unet_256
    --preprocess
    none
    --no_flip
    --num_threads
    16
    --batch_size
    16
    --gpu_ids
    1,2
    --dataset_mode
    add1000
    --shuffle
    --norm
    batch
#### test
    --dataroot
    /media/zhao/HD1/data/mammo300/all/
    --name
    pix2pix_add_unet256
    --results_dir
    ./datasets/
    --model
    pix2pix
    --input_nc
    2
    --output_nc
    1
    --netG
    unet_256
    --preprocess
    none
    --no_flip
    --num_threads
    16
    --batch_size
    16
    --gpu_ids
    1,2
    --dataset_mode
    add
    --norm
    batch
    --epoch
    200
    --direction
    AtoB
    --eval

### 如上方法在全图上训练,不提前裁片,会有大量的黑色背景
    --dataroot
    /media/zhao/HD1/data/ai-postprocess/mammo300_png/full_cut20/
    --name
    pix2pix_add_unet256_nocrop
    --model
    pix2pix
    --input_nc
    2
    --output_nc
    1
    --netD
    n_layers
    --n_layers_D
    3
    --netG
    unet_256
    --preprocess
    crop
    --crop_size
    512
    --no_flip
    --num_threads
    16
    --batch_size
    16
    --gpu_ids
    1,2
    --dataset_mode
    add
    --norm
    batch
    --n_epochs
    200
    --n_epochs_decay
    600


