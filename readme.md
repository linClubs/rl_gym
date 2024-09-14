
# RL_gym-legged_gym

# 1 环境依赖安装

+ isaac_gym：  仿真环境
+ rsl_rl:      PPO算法
+ legged_gym： 腿式gym代码

~~~python
# 1 创建虚拟环境
conda create -n rl python=3.8 

# 2 激活虚拟环境并安装torch
## 2.1 激活虚拟环境
conda activate rl

## 2.2 安装torch和torchvision
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

## 2.3 安装setuptools
## 运行pip install -e . 需要setuptools>=64.0.1
pip install setuptools==64.0.1

# 3 拉取源码
## 3.1 本人使用的下面仓库
git clone https://github.com/linClubs/rl_gym.git
## 3.2 legged_gym源码
git clone https://mirror.ghproxy.com/https://github.com/leggedrobotics/legged_gym.git
## 3.3 unitree_rl_gym源码
https://github.com/unitreerobotics/unitree_rl_gym.git

# 3.4 后面的步骤都在rl_gym目录下, 虚拟环境rl下进行
cd rl_gym

# 4 urdf解压
tar -xf resources.tar.gz

# 5 设置Isaacgym、rsl_rl中torch和torchvision
Isaacgym、rsl_rl中setup.py要求的torch和torchvision都屏蔽掉, 直接注释即可


# 6 安装rsl_rl==v1.0.2
git clone https://mirror.ghproxy.com/https://github.com/leggedrobotics/rsl_rl.git -b v1.0.2

## 本人仓库自带rsl_rl，不用执行上面的下载代码
cd rsl_rl && pip install -v -e .
# ./miniconda3/envs/tv/lib/python3.8/site-packages/rsl-rl.egg-link

# 7 安装isaacgym, 进入isaacgym/python运行setup.py脚本
cd isaacgym/python
pip install -v -e .

# 8 安装legged_gym
pip install -v -e .
## 下载的legged_gym仓库源码就执行下面
cd legged_gym && pip install -v -e .

# 9 更新库的版本，防止代码运行报错
pip install numpy==1.23.4 tensorboard==2.14.0 pillow==9.0.0 setuptools==59.0.1
~~~

1. 验证isaacgym是否安装成功
~~~python
cd isaacgym/python/examples

# 运行joint_monkey仿真环境
python joint_monkey.py
~~~


2. 运行legged_gym

~~~python
# 1 进入legged_gym目录
cd legged_gym

# 2 运行train代码样例
# 支持AzureLoong、go2、h1、g1、h1_2机器人
python legged_gym/scripts/train.py --task=AzureLoong --num_envs=8
python legged_gym/scripts/train.py --task=anymal_c_flat --num_envs=8 --headless

# 即可看到终端打印如下，， 表明已经成功安装完环境：
...
################################################################################
                       Learning iteration 11/300                        

                       Computation: 4353 steps/s (collection: 0.307s, learning 0.046s)
               Value function loss: 0.0115
...
 Mean episode rew_tracking_lin_vel: 0.0967
--------------------------------------------------------------------------------
                   Total timesteps: 18432
                    Iteration time: 0.35s
                        Total time: 4.51s
                               ETA: 108.5s
...
~~~

# 2 错误汇总
~~~python
# 1 错误1：AttributeError: module 'numpy' has no attribute 'float'.
# 修改：降低numpy版本即可
pip install numpy==1.20.0

# 2 错误2：ModuleNotFoundError: No module named 'tensorboard'
# 修改：安装pillow==tensorboard
pip install tensorboard

# 3 错误3：AttributeError: module 'numpy.typing' has no attribute 'NDArray'
# 修改：安装pillow==9.0.0
pip install pillow==9.0.0

# 4 错误4：ValueError: too many values to unpack (expected 2)
You should use RSL-RL v1.0.2 as 
# 修改：安装rsl_rl的v1.0.2版本即可

# 5 错误5： Could not find a version that satisfies the requirement rsl-rl或者isaacgym
# 修改：安装rsl-r和isaacgym即可

# 6 错误6：显存不够, 使用headless模式
PxgCudaDeviceMemoryAllocator fail to allocate memory 339738624 bytes!! Result = 2
[Error] [carb.gym.plugin] Gym cuda error: an illegal memory access was encountered: ../../../source/plugins/carb/gym/impl/Gym/GymPhysX.cpp: 4210
...
RuntimeError: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
# 修改：测试环境用--num_envs=1，如果还是报错，运行时不可视化 加上--headless参数即可
python legged_gym/scripts/train.py --task=anymal_c_flat --sim_device=cuda --rl_device=cuda --pipeline=gpu --num_envs=2048 --headless
# 我运行的下面这条, 进入训练后，即可看到终端有打印信息：
python legged_gym/scripts/train.py --task=anymal_c_flat --num_envs=1 --headless
# 使用cpu可以出现渲染结果
python legged_gym/scripts/train.py --task=anymal_c_flat --num_envs=1 --sim_device=cpu

# 调整 2**8->2**24
## 2**20 = 1024 * 1024
##  GPU触点缓冲区的大小。Factory 中的默认值为 1024^2。由于 GPU 内存限制，您可能无法超过此值的 50 倍。
max_gpu_contact_pairs

# AttributeError: module 'distutils' has no attribute 'version'
pip install setuptools==59.0.1

# ModuleNotFoundError: No module named 'setuptools.command.build'
pip install setuptools==64.0.1
~~~

# 3 自定义机器人进行配置

1.每个环境由一个env文件`legged_robot.py`和一个配置文件`legged_robot_config.py`。
2.`env`和`config`类都使用继承base_config和base_task。
3.在`cfg`中指定的每个非零奖励等级将向元素列表添加一个具有相应名称的函数，该函数将被相加以获得总奖励。
4.必须使用`task_registry.register(name、EnvClass、EnvConfig、TrainConfig)`注册任务。这是在`envs/init.py`中完成的，但也可以在此存储库之外完成。
