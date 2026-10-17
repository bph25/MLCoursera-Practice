# Machine Learning 无答案练习包

包含仓库中全部 **11 份正式编程作业**及 **6 份新增基础实验**。原 Optional Labs 主要是讲解与演示，未把它们简单删成空白文件；新增实验提供明确目标、函数接口与自测。

## 从这里开始

先做 [NumPy 实验](labs/00_numpy.ipynb)，再做 [梯度下降实验](labs/01_gradient_descent.ipynb)，然后进入 [线性回归正式作业](assignments/01_linear_regression/assignment.ipynb)。

已掌握 Python/NumPy 可以直接从正式作业 01 开始。按照下表顺序推进；每题先看题干与公式，完成 TODO，再运行紧随其后的测试。默认运行到未完成的题目时会出现 `NotImplementedError`，这是预期行为。不要一开始直接 Run All；完成后再重启内核并 Run All 验收。

## 本地使用

在终端进入解压后的 `ML-Practice-No-Answers` 文件夹，创建环境并启动 Jupyter：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-base.txt
jupyter lab
```

在 Jupyter 中按目录打开 notebook。每份作业与它的数据、图片、辅助模块放在一起；不要只移动单个 ipynb，也不要改变 notebook 的工作目录。Windows 激活命令为 `.venv\Scripts\activate`。

基础实验和正式作业 01、02、06、07、08 不需要 TensorFlow。后续神经网络等作业需要在 TensorFlow 支持的 Python 版本/环境中安装：

```bash
python -m pip install -r requirements-tensorflow.txt
```

当前机器的 Python 是 3.14，未安装 TensorFlow；未验证它与课程旧版代码的兼容性。请为 TensorFlow 作业使用独立环境或课程实验环境，不要为了第一天学习就安装强化学习全部依赖。

## 正式作业（原题、数据与测试）

| 顺序 | 作业 | 你需要完成 | 依赖 |
|---|---|---|---|
| 01 | [C1 W2 · 线性回归](assignments/01_linear_regression/assignment.ipynb) | 代价函数、梯度 | NumPy |
| 02 | [C1 W3 · 逻辑回归](assignments/02_logistic_regression/assignment.ipynb) | Sigmoid、损失、梯度、分类、正则化 | NumPy |
| 03 | [C2 W1 · 神经网络二分类](assignments/03_neural_network_binary/assignment.ipynb) | 网络建模、单样本/批量前向传播 | TensorFlow |
| 04 | [C2 W2 · 神经网络多分类](assignments/04_neural_network_multiclass/assignment.ipynb) | Softmax、多分类网络 | TensorFlow |
| 05 | [C2 W3 · 模型评估与改进](assignments/05_model_evaluation/assignment.ipynb) | 误差、复杂/简单/正则化模型 | TensorFlow |
| 06 | [C2 W4 · 决策树](assignments/06_decision_tree/assignment.ipynb) | 熵、数据划分、信息增益、最佳特征 | NumPy |
| 07 | [C3 W1 · K-means](assignments/07_kmeans/assignment.ipynb) | 质心分配与更新、图像压缩 | NumPy |
| 08 | [C3 W1 · 异常检测](assignments/08_anomaly_detection/assignment.ipynb) | 高斯参数估计、F1 与阈值选择 | NumPy |
| 09 | [C3 W2 · 协同过滤](assignments/09_collaborative_filtering/assignment.ipynb) | 循环/向量化代价函数（均需自己实现） | TensorFlow |
| 10 | [C3 W2 · 内容推荐](assignments/10_content_recommender/assignment.ipynb) | 双塔网络、向量距离 | TensorFlow |
| 11 | [C3 W3 · Deep Q-learning](assignments/11_deep_q_learning/assignment.ipynb) | Q 网络、目标网络、Bellman 损失 | TensorFlow + Gym |

## 基础实验（新增，无参考答案）

| 实验 | 建议时间 |
|---|---|
| [NumPy 与形状](labs/00_numpy.ipynb) | C1 W1 前 |
| [代价函数与梯度下降](labs/01_gradient_descent.ipynb) | C1 W1–W2 |
| [特征缩放与数据泄漏](labs/02_feature_scaling.ipynb) | C1 W2 |
| [逻辑回归与正则化](labs/03_logistic_regularization.ipynb) | C1 W3 |
| [神经网络前向传播](labs/04_forward_softmax.ipynb) | C2 W1–W2 |
| [数据划分与模型选择](labs/05_model_selection.ipynb) | C2 W3 |

## 已清理内容及使用边界

- 所有正式作业的评分函数只保留函数接口和 docstring，整个函数体改为 TODO；建模单元改为待填写占位。已清除全部运行输出、执行编号和 widgets 保存状态。
- 所有嵌套折叠提示均移除，避免提示内藏完整答案；保留题干公式、伪代码、图解与 Expected Output 数值，供理解与自测。
- 协同过滤的额外向量化实现也已清空，作为进阶练习；后面的 TensorFlow 训练依赖你完成它。
- 清除了绘图工具中与本题重复的 Sigmoid/分类误差实现，工具会调用你写的函数。其余数据加载、绘图、训练调度和原课程提供的非待填代码保留；这是带脚手架的课程练习，不是要求从零重建全部工具。
- 不含原答案 notebook、测验答案、archive、checkpoints、缓存、已有 `.h5` 训练模型及视频输出。保留课程演示所需的 `.npy` 参数文件。保留原仓库 MIT LICENSE；课程内容来源为 Andrew Ng / DeepLearning.AI / Stanford 的 Machine Learning Specialization，原仓库为 greyhatguy007/Machine-Learning-Specialization-Coursera。
- 保留了原有 public_tests；它们包含测试输入与期望值，属于验收数据而非参考实现。请不要修改测试来获得通过。
- 新增实验的自测覆盖基本形状、已知值与部分边界，并非穷尽性证明；开放实验还需要文字解释。

## 环境细节

- 原作业中的 `%matplotlib widget` 已改为 `%matplotlib inline`，减少前端插件配置。原来的交互式示意图可能只显示静态图；不影响函数练习和单元测试。
- 修正了原逻辑回归训练单元把权重初始化为 `(n,1)` 的问题，统一为题目规定的 `(n,)`；绘图工具也显式提取单元素值，避免新 NumPy 的标量转换错误。
- 内容推荐的辅助模块改用 Python 标准库 `pickle`，不需要另装 `pickle5`。现代 Keras 的输入形状要写成单元素元组；不要直接把 `tf.linalg` 操作应用于 Keras 符号输入，可放入 Lambda 层。
- 原 C2 W3 的 MSE 函数按课程定义返回平方误差和除以 **2m**，注意与通常除以 m 的 MSE 区分。
- Deep Q-learning 保留了原课程 Gym 旧接口（`LunarLander-v2`，reset 返回状态，step 返回四项），未自动迁移到 Gymnasium。`requirements-rl-legacy.txt` 是可选旧接口依赖清单，**未在当前机器验证完整安装/训练**。本地无需 Xvfb，已移除强制虚拟显示；Box2D 仍可能需要系统构建依赖。建议最后单独配置或使用课程环境。不要直接把 gym 换为 gymnasium 而不修改交互、训练循环和视频工具。

## 学习记录与重新生成

各 notebook 末尾留有学习记录。写完一份后，重启内核再按顺序运行，记录测试结果并解释至少一个边界情况。

`manifest.json` 记录题目来源、待完成目标、支持文件及源 notebook 的 SHA-256。`verify_practice.py` 可检查结构、语法、清理状态与本地资源；它不替你完成题目，也不要求未填写的 TODO 通过测试：

```bash
python verify_practice.py .
```

若需重新生成，运行 `build_practice.py --source 原仓库路径 --output 一个全新的输出目录`。生成器拒绝覆盖现有目录，保护已完成的练习；不修改源仓库。
