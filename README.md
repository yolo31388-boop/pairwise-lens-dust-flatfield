# pairwise-lens-dust-flatfield

## 领域

游戏、图形与多媒体

## 主题

镜头污渍平场校正

## 说明

多尺度污渍检测、通道联合增益、光圈焦距换算。

初始状态下 `imaging/dust_flatfield.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
