# 构建与环境排雷记录

这篇文档记录 `mini-vllm-cuda` 在 Ubuntu/Debian 上搭建 PyTorch C++/CUDA extension 时遇到的真实问题。

它不是普通安装教程，而是工程日志：记录每个错误是什么意思、为什么会发生、最后怎么解决。以后再做 CUDA 算子项目时，可以直接回来查。

## 当前可用环境

这次跑通时的环境大致是：

```bash
python --version          # Python 3.12，位于虚拟环境中
torch.__version__         # 2.11.0+cu130
CUDA toolkit              # /usr/local/cuda，检测为 CUDA 13.1
GPU target arch           # sm_89，面向 RTX 4060 Laptop
```

常用检查命令：

```bash
which python
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import mini_vllm_cuda as mvc; print(mvc.is_extension_available()); print(mvc.extension_error())"
nvcc --version
```

## 问题 1：PEP 668 / externally-managed-environment

### 现象

```text
error: externally-managed-environment
/usr/bin/python3: No module named pytest
```

### 原因

新版 Debian/Ubuntu 会保护系统 Python。直接对 `/usr/bin/python3` 执行 `pip install ...` 会被 PEP 668 拦住，避免把系统 Python 环境搞乱。

### 解决

使用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

不推荐但可以强行绕过的方式：

```bash
python3 -m pip install --break-system-packages ...
```

### 记忆点

CUDA/PyTorch extension 项目优先用 venv。不要随便污染系统 Python。

## 问题 2：venv 路径中不能包含 `:`

### 现象

```text
Error: Refusing to create a venv ... because it contains the PATH separator :.
```

### 原因

最开始项目路径里有冒号：

```text
Mini-vLLM: CUDA LLM Decode Engine
```

Linux 里 `:` 是 `PATH` 这类环境变量的分隔符，所以 Python 拒绝在这个路径下创建 venv。

### 解决

把父目录改名，去掉冒号，最好也避免空格：

```text
Mini-vLLM--CUDA-LLM-Decode-Engine
```

### 记忆点

CUDA 项目的路径越朴素越好。

推荐：

```text
~/projects/mini-vllm-cuda
~/projects/Mini-vLLM--CUDA-LLM-Decode-Engine
```

容易出问题：

```text
~/projects/Mini-vLLM: CUDA LLM Decode Engine
```

## 问题 3：ensurepip 不存在，venv 创建失败

### 现象

```text
The virtual environment was not created successfully because ensurepip is not available.
On Debian/Ubuntu systems, you need to install the python3.12-venv package.
```

### 原因

系统 Python 没有安装 venv 支持包。

### 解决

安装系统包：

```bash
sudo apt update
sudo apt install python3.12-venv python3.12-dev
```

如果包名不匹配，可以用通用名字：

```bash
sudo apt install python3-venv python3-dev
```

其中 `python3-dev` 很重要，因为 PyTorch C++ extension 编译时需要 `Python.h`。

## 问题 4：venv 创建被 Ctrl+C 中断

### 现象

```text
bash: .venv/bin/activate: No such file or directory
```

### 原因

执行 `python3 -m venv .venv` 时被 `Ctrl+C` 中断了，留下了一个半成品 `.venv`。这个目录存在，但里面没有完整的 `bin/activate`。

### 解决

删除后重新创建：

```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
```

### 记忆点

`source .venv/bin/activate` 前，必须等 venv 创建完整结束。

## 问题 5：PyTorch wheel 太大，下载很慢

### 现象

```text
Downloading torch-...whl (820.3 MB)
eta 4:18:12
```

### 原因

CUDA 版 PyTorch wheel 很大。普通 venv 默认看不到系统或用户环境里已经安装的 PyTorch，所以 pip 会重新下载一份。

### 本项目采用的解决方式

创建 venv 时复用系统/user site packages：

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -c "import torch; print(torch.__version__)"
```

这次成功复用了已有 PyTorch：

```text
2.11.0+cu130
```

然后只安装缺的小包：

```bash
python -m pip install pytest
```

### 记忆点

如果机器上已经有合适的 PyTorch，而且网络很慢，可以用 `--system-site-packages` 复用已有安装。

## 问题 6：`pip install -e .` 卡在 Installing build dependencies

### 现象

```text
Installing build dependencies ...
```

看起来像卡住很久。

### 原因

最初的 `pyproject.toml` 把 `torch` 写进了 build dependency：

```toml
[build-system]
requires = ["setuptools>=64", "wheel", "torch"]
```

这会让 pip 创建隔离构建环境，并在那个临时环境里重新安装 PyTorch。PyTorch 很大，所以看起来像“卡死”。

### 解决

从 build-system requirements 中移除 `torch`：

```toml
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"
```

然后使用当前 venv 中已有的 PyTorch 来构建：

```bash
python -m pip install -e . --no-build-isolation
```

### 记忆点

PyTorch C++/CUDA extension 项目里，`torch` 通常应该提前安装好，然后用 `--no-build-isolation` 构建，避免 pip 在临时环境里重新下载 PyTorch。

## 问题 7：缺少 `Python.h`

### 现象

```text
fatal error: Python.h: No such file or directory
```

### 原因

PyTorch C++ extension 本质上会编译一个 CPython 扩展模块，所以编译器需要 Python 开发头文件。

### 解决

```bash
sudo apt install python3.12-dev
```

或者：

```bash
sudo apt install python3-dev
```

### 记忆点

只装 `python3` 不够，编译 C/C++ extension 还需要 `python3-dev`。

## 问题 8：CUDA minor version mismatch 警告

### 现象

```text
The detected CUDA version (13.1) has a minor version mismatch with the version that was used to compile PyTorch (13.0).
Most likely this should not be a problem.
```

### 原因

本地 CUDA toolkit 是 13.1，而 PyTorch wheel 是用 CUDA 13.0 编译的。

### 处理

这次不需要处理。它是 warning，不是 error。minor version mismatch 通常可以继续编译。

如果之后遇到真实编译或运行错误，再考虑让本地 CUDA toolkit 和 PyTorch CUDA 版本更严格一致。

## 问题 9：nvcc 编译占位 `.cu` 文件时段错误

### 现象

```text
/usr/local/cuda/bin/nvcc ... csrc/decode_attention.cu
Segmentation fault (core dumped)
error: command '/usr/local/cuda/bin/nvcc' failed with exit code 139
```

### 原因

最初脚手架把一些还没有真正实现 CUDA kernel 的占位实现放在了 `.cu` 文件里：

```text
rope.cu
decode_attention.cu
int8_gemv.cu
```

这些文件当时只是用 ATen/PyTorch C++ API 做 placeholder，比如：

```cpp
torch::einsum
torch::softmax
torch::matmul
```

这类高层 ATen C++ 代码交给 `nvcc` 编译很重，也比较脆弱。在当前 CUDA 13.1 + PyTorch 头文件组合下，`nvcc` 编译 `decode_attention.cu` 时直接 segfault。

### 解决

只让 `nvcc` 编译真正的 CUDA kernel。

当前结构改成：

```text
csrc/bindings.cpp       # pybind 入口，g++ 编译
csrc/placeholders.cpp   # ATen 占位实现，g++ 编译
csrc/rmsnorm.cu         # 真正的 RMSNorm CUDA kernel，nvcc 编译
```

`setup.py` 现在只包含：

```python
sources = [
    "csrc/bindings.cpp",
    "csrc/placeholders.cpp",
    "csrc/rmsnorm.cu",
]
```

原来的 `.cu` 文件继续保留，作为后续 kernel 设计说明和实现目标。等它们真的写成 CUDA kernel 后，再加入编译列表。

### 记忆点

不要把大量高层 PyTorch C++ placeholder 放进 `.cu` 让 nvcc 编译。ATen 占位实现先放 `.cpp`，真正 CUDA kernel 再放 `.cu`。

## 问题 10：import extension 时找不到 `libc10.so`

### 现象

```text
libc10.so: cannot open shared object file: No such file or directory
```

### 原因

自定义扩展 `_C.so` 链接了 PyTorch 的动态库，比如 `libc10.so`。如果在导入 `_C.so` 前没有先导入 `torch`，动态链接器可能还不知道 PyTorch 动态库在哪里。

### 解决

在加载 `_C` 之前先 import torch：

```python
import importlib
import torch  # 先加载 PyTorch shared libraries

_C = importlib.import_module("mini_vllm_cuda._C")
```

这个修复已经写在：

```text
mini_vllm_cuda/ops.py
```

### 记忆点

PyTorch extension 的 Python 包入口里，先 `import torch` 再加载自定义 `.so`，可以避免很多动态库路径问题。

## 问题 11：为什么 Nsight profiling 结果不提交到 Git

### 现象

RMSNorm v0 profiling 会生成这些文件：

```text
results/nsys_rmsnorm_v0.nsys-rep
results/nsys_rmsnorm_v0.sqlite
results/ncu_rmsnorm_v0.ncu-rep
results/*.html
```

这些文件对本机分析很有用，但不适合直接提交到 GitHub。

### 原因

1. Nsight 报告通常很大。
   例如 `.ncu-rep` 很容易达到几十 MB，后续多 profile 几次会快速膨胀仓库体积。

2. Nsight 报告是机器相关的。
   里面包含 GPU 型号、驱动、CUDA 版本、进程信息、采样细节等，本质上是一次本地实验记录。

3. Git 适合保存可复现的源代码、脚本和文字结论。
   profiling 原始产物更适合放在本地 `results/`，或者必要时放到 release/artifact，而不是跟源码一起版本化。

4. 后续 benchmark/profiling 会反复生成新文件。
   如果把 `results/` 提交进 Git，很容易造成无意义 diff。

### 当前做法

`.gitignore` 已经忽略：

```text
results/
*.nsys-rep
*.qdrep
*.sqlite
```

推荐保存到 Git 的内容是：

```text
README.md                 # 项目状态和复现命令
docs/build_notes.md        # 环境和构建排雷
LEARNING_GUIDE.md          # 本地学习笔记，不对外发布
TASKS.md                   # 本地任务进度，不对外发布
```

如果 profiling 结论值得公开，应该把它整理成文字，例如：

```text
Nsight Systems shows that PyTorch reference launches multiple ATen kernels,
while custom RMSNorm v0 appears as one dedicated kernel.

Nsight Compute on a num_tokens=1 launch shows tiny-grid underutilization:
grid=1, block=256, Waves Per SM about 0.01.
```

也就是说：

```text
提交结论，不提交大体积原始报告。
```

## 最终可用安装流程

从项目根目录执行：

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install pytest
python -m pip install -e . --no-build-isolation
```

验证 extension 是否可用：

```bash
python -c "import mini_vllm_cuda as mvc; print(mvc.is_extension_available()); print(mvc.extension_error())"
```

期望输出：

```text
True
None
```

运行 RMSNorm 测试：

```bash
pytest tests/test_rmsnorm.py -s
```

观察到的结果：

```text
18 passed
```

运行 RMSNorm benchmark：

```bash
python benchmarks/bench_rmsnorm.py
```

观察到的结果：custom RMSNorm CUDA kernel 在当前测试形状下大约比 PyTorch reference 快 5x 到 10x。

## 总结经验

1. CUDA 项目路径尽量简单，避免 `:`、空格和奇怪字符。
2. 优先使用 venv；如果已有可用 PyTorch 且网络慢，可以用 `--system-site-packages`。
3. PyTorch C++/CUDA extension 构建时，先确认 `import torch` 可用，再用 `pip install -e . --no-build-isolation`。
4. 高层 ATen placeholder 放 `.cpp`，真正 CUDA kernel 放 `.cu`。
5. 自定义 extension 加载前先 `import torch`，避免 `libc10.so` 等动态库找不到。
6. 构建错误不是杂事，它们是 CUDA 工程能力的一部分。
