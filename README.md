# Self-supervised Traversability Estimation in Off-road Environment

## 环境配置

### 第一步：系统要求
* CUDA 11.3  
* cuDNN 8  
* Ubuntu 20.04  

### 第二步：创建 Conda 环境
```
conda create -n SFNet python=3.8
conda activate SFNet
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### 第三步：修改 torch-encoding 脚本
根据 [这个链接](https://github.com/zhanghang1989/PyTorch-Encoding/issues/328#issuecomment-749549857) 修改 torch-encoding 脚本：

> `cd anaconda3/envs/SFNet/lib/python3.8`（此路径因环境不同而异）  
>
> 1. 修改 `site-packages/encoding/nn/syncbn.py`（大约第200行）  
> 将  
> `return syncbatchnorm(........).view(input_shape)`  
> 改为  
> `x, _, _=syncbatchnorm(........)`  
> `x=x.view(input_shape)`  
> `return x`  
>
> 2. 修改 `site-packages/encoding/functions/syncbn.py`（大约第102行）  
> 将  
> `ctx.save_for_backward(x,_ex,_exs,gamma,beta)`  
> `return y`  
> 改为  
> `ctx.save_for_backward(x,_ex,_exs,gamma,beta)`  
> `ctx.mark_non_differentiable(running_mean,running_var)`  
> `return y,running_mean,running_var`  
>
> 3. 修改 `site-packages/encoding/functions/syncbn.py`（大约第109行）  
> 将  
> `def backward(ctx,dz):`  
> 改为  
> `def backward(ctx,dz,_druning_mean,_druning_var):`  

### 第四步：安装 GFL 模块
```
cd exts
python setup.py install
```

## 数据准备

### RELLIS-3D 数据集

1. 下载 [RELLIS-3D 数据集](https://unmannedlab.github.io/research/RELLIS-3D)，其文件结构如下：
```
RELLIS-3D
├── Rellis-3D
|   ├── 00000
|   |   ├── os1_cloud_node_kitti_bin
|   |   ├── pylon_camera_node
|   |   ├── calib.txt
|   |   ├── camera_info.txt
|   |   └── poses.txt    
|   ├── 00001
|   └── ...
└── Rellis_3D
    ├── 00000
    |   └── transforms.yaml
    ├── 00001
    └── ...
```

2. 预处理训练数据，运行以下脚本：
```
sh ./data_prep/rellis_preproc.sh
```

3. 最终数据文件结构如下：
```
RELLIS-3D
├── Rellis-3D
├── Rellis_3D
└── Rellis-3D-custom
    ├── 00000
    |   ├── foot_print
    |   ├── super_pixel   # 可选，但推荐用于更清晰的输出
    |   └── surface_normal
    ├── 00001
    └── ...
```

### ORFD 数据集

1. 下载 [ORFD 数据集](https://github.com/chaytonmin/Off-Road-Freespace-Detection)，其文件结构如下：
```
ORFD
└── Final_Dataset
    ├── training
    |   ├── calib
    |   ├── dense_depth
    |   ├── gt_image
    |   ├── image_data
    |   ├── lidar_data
    |   └── sparse_depth    
    ├── validation
    └── testing
```

2. 此数据集不含位姿信息，因此需要使用点云估计位姿。我们使用的是 [PyICP-SLAM](https://github.com/gisbi-kim/PyICP-SLAM)。将估计好的位姿数据放入如下目录：
```
ORFD
├── Final_Dataset
└── ORFD-custom
    ├── training
    |   └── pose
    |       └── pose_16197787.csv
    ├── validation
    └── testing
```

3. 预处理数据，运行以下脚本：
```
sh ./data_prep/orfd_preproc.sh
```

4. 最终数据结构如下：
```
ORFD
├── Final_Dataset
└── ORFD-custom
    ├── training
    |   ├── foot_print
    |   ├── pose
    |   ├── super_pixel
    |   └── surface_normal
    ├── validation
    └── testing
```

## 可通行性估计

### 模型训练  
在 `train.yaml` 文件中设置 `data_config/data_root` 路径，然后运行：
```
python train.py configs/train.yaml
```

### 模型测试  
在 `test.yaml` 文件中设置 `data_config/data_root` 和 `resume_path` 路径，然后运行：
```
python test.py configs/test.yaml
```

## 路径规划

### 绘制全局代价地图
```
python ./plot_map/plot_map_rellis.py \
        --start_num 400 --end_num 900 \
        --save_rgb_img --save_valid_map \
        --cost_path ../outputs/prediction/your-ckpt-name
```

### 生成路径
```
python ./path_plan/path_plan.py \
        --start_num 400 --end_num 900 \
        --local_planner_type TRRTSTAR \
        --max_path_iter 1000 --max_extend_length 10 --bias_sampling \
        --cost_map_name /path/to/your-ckpt-name.png
```

## 备注
* 本项目代码基于[Ftfood](https://github.com/yurimjeon1892/FtFoot)。


