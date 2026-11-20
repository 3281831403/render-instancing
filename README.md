# render-instancing

## 题面

草皮和碎石改成实例化绘制后，批次忽多忽少、边缘还闪：分组用 set，组顺序每次不同；打包按字节而不是实例数，超出硬件上限；剔除只测中心点，边缘闪烁；排序用不稳定 sort，同材质顺序乱；索引映射用 `instance_id*vertex_count`，溢出；去重直接比较列表，浮点误差导致重复；奇异矩阵不校验，实例塌成一条线；不可见实例仍提交，浪费批次；批量共用分组表，帧之间串；`max_instances=0` 不校验，除零。instancing/batcher.py 是核心，tests/test_batcher.py 15个红。要修的是：按契约补齐 `instancing/batcher.py` 的 8 个实现，修掉上述 10 类缺陷，让 `tests/test_batcher.py` 的 15 个测试全部转绿。

## 运行

在仓库根目录执行 `python -m pytest tests -q`。

初始桩状态必须恰好得到 `15 failed`，且没有 collection / import / syntax error；15 个测试全部调用未实现的桩，并以 `NotImplementedError` 失败。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/test_batcher.py` 不可修改；判分以该文件中的 15 个测试为准。只实现 `instancing/batcher.py`，使用标准库（`math`、`collections`）和 pytest，离线运行，不使用随机数。
