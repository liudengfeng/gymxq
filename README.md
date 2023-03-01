# gymxq 中国象棋
Chinese chess game gymnasium environment 

## pygame WSL2中文字体显示问题

```bash
复制中文字体，如阿里[alibabapuhuiti20] -> /usr/share/fonts/truetype/liberation
fc-cache -fv
```

```python
import pygame
pygame.font.get_fonts()
```

## `V0`
以单个屏幕输出作为观察对象

| 项目              | 描述                          |
| :---------------- | :---------------------------- |
| Action Space      | Discrete(2086)                |
| Observation Space | (600,800,3)                   |
| Observation High  | 255                           |
| Observation Low   | 0                             |
| Import            | gymnasium.make("gym_xq/xqv1") |

### Description
以单个屏幕输出作为观察对象

### Actions
space [0,2085]
| 编码 | 移动  | 描述                                  |
| :--- | :---: | :------------------------------------ |
| 0    | 0001  | 棋盘坐标x0=0,y0=0移动 ☞ 坐标x0=0,y0=1 |


### Observations
The environment returns the RGB image that is displayed to human players as an observation.

默认显示$H=600,W=800$,其中宽度有效部分为$W=800-260=540$, 使用`ResizeObservation`缩小尺寸后$H=200,W=180,C=3$

### Rewards

+ 超出最大步数判和，即0
+ 由于不涉及到连续循环三次，游戏结果可能与实际存在偏差

### Arguments

+ `render_mode` 不支持`ansi`
+ `init_fen`

## feature

state * num_history 作为特征

### 

单个屏幕输出作为state,$$shape=(H, W, C)$$,其中$H=600,W=800-260=540,C=3$。$state -> (H W C)$，堆积18个历史状态，特征$shape=(18*C,540$

### `V1`


## reward

## `render`

- 使用`pygame`绘图
- W*H = 800 * 600 
- 右边宽度260为信息部分
- 作为演示，禁止键盘或鼠标输入

## 性能

+ 使用`v1`版本，`render_mode`使用`ansi`速度较`rgb_array`或`human`快30倍
+ 除非有必要使用中文棋谱，否则应将`gen_qp`设置为`False`。生成棋谱用时将增加60%
+ 鉴于性能原因，只有在使用`tensorboard`显示时，为提供丰富显示信息才使用`rgb_array`模式及添加棋谱、AI提示

## `wrappers`

源自`gymnasium`自定义环境模板，暂时没有使用。

## 依赖包

- 安装`xqcpp`

## install 
```bash
git clone 
cd gymxq
pip install .
```