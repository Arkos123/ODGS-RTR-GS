# Viewer + noVNC 交互式查看器搭建与优化

## 背景

RTR-GS 训练完成后，使用 `viewer_pygame.py` 可以交互式地查看 3D 场景。但这个 viewer 基于 Pygame，需要显示器和输入设备。在无桌面的远程 Linux 服务器上，无法直接使用。

解决方案：通过 **Xvfb**（虚拟显示器）+ **x11vnc**（VNC 服务端）+ **noVNC**（Web VNC 客户端）将画面和交互流式传输到本地浏览器。

## 安装（免 sudo）

所有工具均可安装在用户目录下，不需要 root 权限。

```bash
# Step 1: x11vnc + libvncserver1
cd /tmp
apt download x11vnc libvncserver1
mkdir -p ~/tools/x11vnc
dpkg -x x11vnc_*.deb ~/tools/x11vnc/
dpkg -x libvncserver1_*.deb ~/tools/x11vnc/

# Step 2: noVNC
git clone https://github.com/novnc/noVNC.git ~/tools/noVNC

# Step 3: pygame
conda activate odgs-rtr
pip install pygame
```

### 常见问题

1. **`x11vnc: error while loading shared libraries: libvncserver.so.1`**
   - 原因：x11vnc 需要 `libvncserver1` 动态库，但 deb 包是分开的
   - 修复：同时下载 `x11vnc` 和 `libvncserver1`，解压到同一目录；脚本中设置 `LD_LIBRARY_PATH` 指向库路径
   - 参见 `scripts/start_viewer_novnc.sh` 中 `X11VNC_LIBDIR` 和 `LD_LIBRARY_PATH` 的配置

2. **`ModuleNotFoundError: No module named 'pygame'`**
   - 修复：在 conda 环境中 `pip install pygame`

## 启动脚本

通过 `scripts/start_viewer_novnc.sh` 一键启动。脚本会依次：

1. 检查 Xvfb / x11vnc / noVNC 是否就位
2. 启动 Xvfb（虚拟显示器 :99）
3. 启动 x11vnc（监听 5900 端口）
4. 启动 noVNC proxy（监听 6080 端口，Web 入口）
5. 激活 conda 环境
6. 运行 `viewer_pygame.py`

### 使用方式

```bash
# 先编辑脚本顶部参数（CHECKPOINT, OCCLUSION_PATH 等）
bash scripts/start_viewer_novnc.sh
```

然后本地浏览器访问 `http://<服务器IP>:6080/vnc.html`，或通过 SSH 隧道更安全地访问：

```bash
ssh -L 6080:localhost:6080 user@server
# 浏览器打开 http://localhost:6080/vnc.html
```

## viewer_pygame.py 的功能增强

本次对 viewer 做了以下增强：

### 1. 初始相机位置自适应

- **有 `-s` 参数**（提供了场景数据路径）：初始位置自动设为第一个相机的位姿
  - 位置计算公式：`position = -R @ T`（从 Camera 的 R,T 推导相机中心）
  - 目标点（target）仍为场景中心，保持 Orbit 模式
- **无 `-s` 参数**：保持原有行为，从场景上方默认位置开始
- **Colmap 数据集**：特殊处理，因为 test_cams 被替换为循环环绕路径

### 2. 环境光加载策略

`load_scene_data()` 函数修改：

- 如果传入了 `--envmap_path` 且文件存在 → 用指定的 HDR 环境光
- 否则 → 自动查找 `cubemap_chkpntXXXXX.pth`（训练分解出的环境光）
- 两者都没有 → 报错提示

这样 viewer 默认就可以看到训练时分解出的环境光照效果。

### 3. 默认值清理

- `--envmap_path` 默认值从硬编码的 Windows 路径改为 `None`
- 去掉了一些冗余注释

## 架构说明

数据流：

```
Your browser (noVNC client)
    ↓ WebSocket
noVNC proxy (port 6080)
    ↓ VNC protocol
x11vnc (port 5900)
    ↓ captures
Xvfb (virtual display :99)
    ↓ Pygame renders to
viewer_pygame.py
```

所有键盘和鼠标事件通过 noVNC → x11vnc → Xvfb → Pygame 完整透传，交互体验与本地运行一致。

## 相关文件

- `viewer_pygame.py` — 主 viewer 程序
- `scripts/start_viewer_novnc.sh` — 一键启动脚本
- `README.md` — 安装指南（Interactive Viewer 章节）
