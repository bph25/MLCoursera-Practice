"""Build an answer-free, self-contained practice pack; never modify the source repo."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import textwrap
import zipfile

C1 = "C1 - Supervised Machine Learning - Regression and Classification"
C2 = "C2 - Advanced Learning Algorithms"
C3 = "C3 - Unsupervised Learning, Recommenders, Reinforcement Learning"
ASSIGNMENTS = [
    ("01_linear_regression", "C1 W2 · 线性回归", f"{C1}/week2/C1W2A1/C1_W2_Linear_Regression.ipynb", "NumPy", "代价函数、梯度"),
    ("02_logistic_regression", "C1 W3 · 逻辑回归", f"{C1}/week3/C1W3A1/C1_W3_Logistic_Regression.ipynb", "NumPy", "Sigmoid、损失、梯度、分类、正则化"),
    ("03_neural_network_binary", "C2 W1 · 神经网络二分类", f"{C2}/week1/C2W1A1/C2_W1_Assignment.ipynb", "TensorFlow", "网络建模、单样本/批量前向传播"),
    ("04_neural_network_multiclass", "C2 W2 · 神经网络多分类", f"{C2}/week2/C2W2A1/C2_W2_Assignment.ipynb", "TensorFlow", "Softmax、多分类网络"),
    ("05_model_evaluation", "C2 W3 · 模型评估与改进", f"{C2}/week3/C2W3A1/C2_W3_Assignment.ipynb", "TensorFlow", "误差、复杂/简单/正则化模型"),
    ("06_decision_tree", "C2 W4 · 决策树", f"{C2}/week4/C2W4A1/C2_W4_Decision_Tree_with_Markdown.ipynb", "NumPy", "熵、数据划分、信息增益、最佳特征"),
    ("07_kmeans", "C3 W1 · K-means", f"{C3}/week1/C3W1A/C3W1A1/C3_W1_KMeans_Assignment.ipynb", "NumPy", "质心分配与更新、图像压缩"),
    ("08_anomaly_detection", "C3 W1 · 异常检测", f"{C3}/week1/C3W1A/C3W1A2/C3_W1_Anomaly_Detection.ipynb", "NumPy", "高斯参数估计、F1 与阈值选择"),
    ("09_collaborative_filtering", "C3 W2 · 协同过滤", f"{C3}/week2/C3W2/C3W2A1/C3_W2_Collaborative_RecSys_Assignment.ipynb", "TensorFlow", "循环/向量化代价函数（均需自己实现）"),
    ("10_content_recommender", "C3 W2 · 内容推荐", f"{C3}/week2/C3W2/C3W2A2/C3_W2_RecSysNN_Assignment.ipynb", "TensorFlow", "双塔网络、向量距离"),
    ("11_deep_q_learning", "C3 W3 · Deep Q-learning", f"{C3}/week3/C3W3A1/C3_W3_A1_Assignment.ipynb", "TensorFlow + Gym", "Q 网络、目标网络、Bellman 损失"),
]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(s).strip().splitlines(keepends=True)}


def code(s, task=False):
    return {"cell_type": "code", "metadata": {"tags": ["exercise"]} if task else {},
            "source": textwrap.dedent(s).strip().splitlines(keepends=True), "outputs": [], "execution_count": None}


def notebook(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}


def stub_function(node):
    """Keep the API and docstring, but remove ALL implementation, not just marked lines."""
    node = copy.deepcopy(node)
    doc = ast.get_docstring(node, clean=False)
    node.body = ([ast.Expr(ast.Constant(doc))] if doc else []) + [
        ast.Raise(ast.Call(ast.Name("NotImplementedError", ast.Load()),
                           [ast.Constant(f"TODO: 实现 {node.name}，然后运行下方测试。")], []), None)]
    return ast.unparse(ast.fix_missing_locations(node))


def clean_markdown(s):
    # Nested <details> sections contain complete solutions; a single non-greedy regex is unsafe.
    # Several source notebooks omit closing tags; these cells contain only hints/output.
    if re.match(r"\s*<details\b", s, re.I):
        return ""
    depth, cursor, pieces = 0, 0, []
    for match in re.finditer(r"</?details\b[^>]*>", s, re.I):
        closing = match.group().startswith("</")
        if not closing:
            if depth == 0:
                pieces.append(s[cursor:match.start()])
            depth += 1
        elif depth:
            depth -= 1
            if depth == 0:
                cursor = match.end()
    if not depth:
        pieces.append(s[cursor:])
    result = re.sub(r"</details\s*>", "", "".join(pieces), flags=re.I)
    return re.sub(r"\]\((www\.[^)]+)\)", r"](https://\1)", result)


def copy_support(src, dst):
    ignored = {"archive", ".ipynb_checkpoints", "__pycache__", "videos", ".git"}
    copied = []
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if any(x in ignored or x.startswith(".") for x in rel.parts):
            continue
        if not p.is_file() or p.suffix.lower() in {".ipynb", ".pyc", ".h5", ".mp4", ".m4v", ".trace", ".md"}:
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        copied.append(str(rel))
    return copied


def patch_helper_callback(folder, module, name):
    """Remove duplicate answers from plotting helpers and bind the student's implementation later."""
    path = folder / f"{module}.py"
    s = path.read_text()
    tree = ast.parse(s)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = s.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = [stub_function(node) + "\n"]
    path.write_text("".join(lines))


MODEL_VARIABLES = {
    "03_neural_network_binary": [("model",)],
    "04_neural_network_multiclass": [("model",)],
    "05_model_evaluation": [("model",), ("model_s",), ("model_r",)],
    "10_content_recommender": [("user_NN", "item_NN", "model")],
    "11_deep_q_learning": [("q_network", "target_q_network", "optimizer")],
}
MODEL_NOTES = {
    "03_neural_network_binary": "根据上一段题目创建 model；保留输入维度与层结构要求。",
    "04_neural_network_multiclass": "创建并编译 model；按照题目要求选择输出层与 from_logits 参数。",
    "05_model_evaluation": "根据本题说明创建并编译模型。优化器的学习率参数使用 learning_rate。",
    "10_content_recommender": "创建 user_NN 与 item_NN，再按题目搭建输入、归一化、点积与完整 model。现代 Keras 中输入形状请写为 (num_user_features,) 与 (num_item_features,)；归一化可放入 Lambda 层。",
    "11_deep_q_learning": "创建 q_network、target_q_network 和 optimizer，满足上方题目要求。",
}


def clean_assignment(src, dst, slug):
    original = json.loads(src.read_text())
    nb = notebook([])
    nb["cells"].append(md(f"""
        # 无答案练习版

        先阅读题目与公式，再完成标有 `TODO` 的函数或建模单元。函数调用时出现
        `NotImplementedError` 表示该练习尚未完成，不是环境错误。每完成一题，先运行紧随其后的测试。
        最后重启内核并从头运行，检查是否依赖了旧变量。

        已移除练习实现、折叠代码提示、运行输出和保存的交互状态；保留原题、数据、测试与教学用辅助代码。
        题目中的 Expected Output 是测试对照值，数学公式与算法伪代码是题干的一部分。
        本地练习不用于 Coursera 自动评分提交。
    """))
    exercise_names, removed_hint_cells, task_cells, model_index = [], 0, 0, 0
    for cell in original["cells"]:
        c = copy.deepcopy(cell)
        s = "".join(c.get("source", []))
        c["metadata"] = {}
        if c["cell_type"] == "markdown":
            cleaned = clean_markdown(s)
            removed_hint_cells += int(cleaned != s)
            if not cleaned.strip():
                continue
            c["source"] = cleaned.splitlines(keepends=True)
        elif c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
            # Use an ordinary notebook backend; original widget backend needs extra frontend setup.
            s = s.replace("%matplotlib widget", "%matplotlib inline")
            if slug == "02_logistic_regression":
                # The original training cell contradicts its (n,) API by initializing (n,1).
                s = s.replace("np.random.rand(2).reshape(-1,1)", "np.random.rand(2)")
            s = s.replace("from pyvirtualdisplay import Display\n", "")
            s = re.sub(r"^Display\(visible=0, size=\(840, 480\)\)\.start\(\);\s*$", "# 本地 notebook 无需启动 Xvfb 虚拟显示。", s, flags=re.M)
            if "UNQ_C" in s:
                tree = ast.parse(s)
                funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
                marker = re.search(r"# UNQ_C\d+", s).group()
                if funcs:
                    assert len(funcs) == 1
                    exercise_names.append(funcs[0].name)
                    s = marker + "\n" + stub_function(funcs[0])
                else:
                    variables = MODEL_VARIABLES[slug][model_index]
                    model_index += 1
                    exercise_names.extend(variables)
                    s = marker + "\n# TODO: " + MODEL_NOTES[slug] + "\n"
                    if slug == "10_content_recommender":
                        s += "num_outputs = 32\ntf.random.set_seed(1)\n"
                    s += "\n".join(f"{name} = None" for name in variables)
                    s += '\nraise NotImplementedError("TODO: 完成本单元中的模型定义。")'
                c["metadata"] = {"tags": ["exercise"]}
                task_cells += 1
            elif slug == "09_collaborative_filtering" and "def cofi_cost_func_v(" in s:
                node = next(n for n in ast.parse(s).body if isinstance(n, ast.FunctionDef))
                s = "# BONUS: 实现 TensorFlow 向量化代价函数，以用于后续自动微分训练。\n" + stub_function(node)
                c["metadata"] = {"tags": ["exercise"]}
                exercise_names.append(node.name)
                task_cells += 1
            c["source"] = s.splitlines(keepends=True)
        nb["cells"].append(c)
        # Plotting helper modules must use the student's code rather than carry duplicate answers.
        if c["cell_type"] == "code" and c["metadata"].get("tags") == ["exercise"]:
            if slug == "02_logistic_regression" and "def sigmoid(" in s:
                nb["cells"].append(code("import utils as _plot_helpers\n_plot_helpers.sig = sigmoid"))
            if slug == "05_model_evaluation" and "def eval_cat_err(" in s:
                nb["cells"].append(code("import assigment_utils as _plot_helpers\n_plot_helpers.eval_cat_err = eval_cat_err"))
            if slug == "09_collaborative_filtering" and "def cofi_cost_func_v(" in s:
                nb["cells"].append(code('''
                    # BONUS 自测：与循环版本一致，并能被 TensorFlow 自动微分。
                    for regularization in (0., 1.5):
                        np.testing.assert_allclose(
                            cofi_cost_func_v(X_r, W_r, b_r, Y_r, R_r, regularization),
                            cofi_cost_func(X_r, W_r, b_r, Y_r, R_r, regularization), rtol=1e-5)
                    trainable_x = tf.Variable(X_r, dtype=tf.float64)
                    with tf.GradientTape() as tape:
                        loss_check = cofi_cost_func_v(trainable_x, tf.cast(W_r, tf.float64),
                                                     tf.cast(b_r, tf.float64), tf.cast(Y_r, tf.float64),
                                                     tf.cast(R_r, tf.float64), 1.5)
                    gradient_check = tape.gradient(loss_check, trainable_x)
                    assert gradient_check is not None and np.isfinite(gradient_check.numpy()).all()
                    print("通过：向量化代价与自动微分")
                '''))
    nb["cells"].append(md("""
        ## 完成记录

        - [ ] 所有 TODO 已完成，原有测试通过。
        - [ ] 重启内核后按顺序运行成功。
        - [ ] 能解释每个输入、输出的形状，以及所用损失函数。
        - [ ] 修改一个参数或边界输入，记录预测并用实验验证。

        请在下面记录：遇到的问题、原因、解决过程，以及下次能独立写出的部分。
    """))
    nb["cells"].append(md("学习记录：待填写。"))
    dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    return {"exercise_cells": task_cells, "targets": exercise_names, "removed_hint_cells": removed_hint_cells}


def lab(title, intro, tasks):
    cells = [md(f"# {title}\n\n{intro}\n\n本实验为新增练习，仅依赖 NumPy；题干、测试均不包含参考实现。\n从上到下执行，完成每个 TODO 后运行对应自测。"),
             code("import numpy as np\nrng = np.random.default_rng(42)")]
    for heading, prompt, body, checks in tasks:
        cells.extend([md(f"## {heading}\n\n{prompt}"), code(body, task=True), code(checks)])
    cells.extend([md("## 实验记录\n\n写出输入/输出形状，解释一个边界情况，再记录一次修改参数前后的观察。"), md("待填写。")])
    return notebook(cells)


def make_labs():
    return [
        ("00_numpy", "NumPy 与形状", "C1 W1 前", lab("实验 00 · NumPy 与形状", "目标：掌握批量数据的形状、矩阵乘法与广播。", [
            ("1 · 批量线性预测", "实现 predict_batch(X, w, b)。X 形状 (m,n)，w 形状 (n,)，b 为标量，返回 (m,)；不能修改输入。可先用循环，再改为向量化。", '''
                def predict_batch(X, w, b):
                    """Return one linear prediction per row of X, shape (m,)."""
                    raise NotImplementedError("TODO: 批量预测")
                ''', '''
                X = np.array([[1., 2.], [3., 4.], [-1., 0.]])
                w = np.array([2., -1.])
                X_before, w_before = X.copy(), w.copy()
                out = np.asarray(predict_batch(X, w, 0.5))
                assert out.shape == (3,)
                np.testing.assert_allclose(out, [0.5, 2.5, -1.5])
                np.testing.assert_array_equal(X, X_before)
                np.testing.assert_array_equal(w, w_before)
                assert np.asarray(predict_batch(np.empty((0, 2)), w, 0.)).shape == (0,)
                print("通过：批量预测、空批次、输入未修改")
                '''),
            ("2 · 同步打乱数据", "实现 shuffle_pairs(X,y,seed)，返回两个新数组。保持每行 X 和对应 y 的配对；同一 seed 结果相同；不同 seed 通常结果不同。不要修改原数组。", '''
                def shuffle_pairs(X, y, seed):
                    raise NotImplementedError("TODO: 同步打乱")
                ''', '''
                X = np.arange(40).reshape(20, 2)
                y = np.arange(20)
                a, b = shuffle_pairs(X, y, 7)
                c, d = shuffle_pairs(X, y, 7)
                np.testing.assert_array_equal(a, c)
                np.testing.assert_array_equal(b, d)
                np.testing.assert_array_equal(a[:, 0], 2 * b)
                np.testing.assert_array_equal(np.sort(b), y)
                assert not np.array_equal(b, y), "必须实际打乱顺序"
                assert not np.shares_memory(a, X) and not np.shares_memory(b, y)
                assert not np.array_equal(shuffle_pairs(X, y, 8)[1], b)
                np.testing.assert_array_equal(X, np.arange(40).reshape(20, 2))
                print("通过：配对、可复现、新数组、随机种子")
                '''),
        ])),
        ("01_gradient_descent", "代价函数与梯度下降", "C1 W1–W2", lab("实验 01 · 代价函数与梯度下降", "目标：把损失、梯度与参数更新连起来，观察学习率的影响。", [
            ("1 · 单变量回归的代价与梯度", "实现 cost_and_gradient(x,y,w,b)，返回 (J,dw,db)，J 使用课程约定的平方误差和除以 2m。支持一维数组。", '''
                def cost_and_gradient(x, y, w, b):
                    raise NotImplementedError("TODO: 代价与梯度")
                ''', '''
                x = np.array([1., 2., 3.]); y = np.array([3., 5., 7.])
                np.testing.assert_allclose(cost_and_gradient(x, y, 2., 1.), [0., 0., 0.])
                J, dw, db = cost_and_gradient(x, y, 0., 0.)
                np.testing.assert_allclose([J, dw, db], [83 / 6, -34 / 3, -5.])
                eps = 1e-6
                numeric_dw = (cost_and_gradient(x, y, 0.3 + eps, -0.2)[0] - cost_and_gradient(x, y, 0.3 - eps, -0.2)[0]) / (2 * eps)
                numeric_db = (cost_and_gradient(x, y, 0.3, -0.2 + eps)[0] - cost_and_gradient(x, y, 0.3, -0.2 - eps)[0]) / (2 * eps)
                np.testing.assert_allclose(cost_and_gradient(x, y, 0.3, -0.2)[1:], [numeric_dw, numeric_db], rtol=1e-6)
                print("通过：已知值、有限差分梯度检查")
                '''),
            ("2 · 参数更新与收敛", "实现 train_linear(x,y,w,b,alpha,steps)。调用上一题函数，使用同一步的梯度同时更新 w、b。返回最终 w、b 和长度 steps+1 的损失记录，首项为更新前损失。", '''
                def train_linear(x, y, w, b, alpha, steps):
                    raise NotImplementedError("TODO: 梯度下降与损失记录")
                ''', '''
                x = np.array([1., 2., 3.]); y = np.array([3., 5., 7.])
                w, b, history = train_linear(x, y, 0., 0., 0.05, 2000)
                assert len(history) == 2001
                np.testing.assert_allclose(history[0], 83 / 6)
                np.testing.assert_allclose([w, b], [2., 1.], atol=1e-4)
                np.testing.assert_allclose(history[-1], cost_and_gradient(x, y, w, b)[0])
                assert np.all(np.diff(history) <= 1e-10)
                w0, b0, h0 = train_linear(x, y, 0.2, 0.3, 0.05, 0)
                np.testing.assert_allclose([w0, b0], [0.2, 0.3]); assert len(h0) == 1
                print("通过：损失记录、收敛、零步边界")
                '''),
            ("3 · 学习率实验", "用 alpha=0.001、0.05、0.5 分别运行 100 步，画出损失曲线。运行前先预测哪种情况最慢、哪种可能发散；不要求所有曲线都下降。", '''
                # TODO: 创建至少三组实验并画图；可使用 matplotlib。
                raise NotImplementedError("TODO: 学习率实验")
                ''', '''
                # 此题为开放实验：请在最后的实验记录中解释观察，不能仅粘贴图。
                '''),
        ])),
        ("02_feature_scaling", "特征缩放与数据泄漏", "C1 W2", lab("实验 02 · 特征缩放与数据泄漏", "目标：只在训练集拟合预处理参数，并正确处理常量特征。", [
            ("1 · 拟合标准化参数", "实现 fit_standardizer(X_train)，返回 (mu,scale)，形状都为 (n,)。使用总体标准差（ddof=0）；常量列的 scale 设为 1。禁止原地修改数据。", '''
                def fit_standardizer(X_train):
                    raise NotImplementedError("TODO: 拟合均值与尺度")
                ''', '''
                X = np.array([[1., 10., 4.], [3., 30., 4.]])
                before = X.copy()
                mu, scale = fit_standardizer(X)
                np.testing.assert_allclose(mu, [2., 20., 4.])
                np.testing.assert_allclose(scale, [1., 10., 1.])
                np.testing.assert_array_equal(X, before)
                assert np.asarray(mu).shape == (3,) and np.asarray(scale).shape == (3,)
                print("通过：总体标准差、常量列、输入未修改")
                '''),
            ("2 · 转换与逆转换", "实现 transform(X,mu,scale) 与 inverse_transform(Z,mu,scale)。验证集必须复用训练集参数，不能重新拟合。两者均返回新数组。", '''
                def transform(X, mu, scale):
                    raise NotImplementedError("TODO: 标准化转换")

                def inverse_transform(Z, mu, scale):
                    raise NotImplementedError("TODO: 逆转换")
                ''', '''
                train = np.array([[1., 10., 4.], [3., 30., 4.]])
                val = np.array([[100., 1000., 4.]])
                mu, scale = fit_standardizer(train)
                Z = transform(train, mu, scale)
                np.testing.assert_allclose(Z, [[-1., -1., 0.], [1., 1., 0.]])
                np.testing.assert_allclose(transform(val, mu, scale), [[98., 98., 0.]])
                np.testing.assert_allclose(inverse_transform(Z, mu, scale), train)
                assert not np.shares_memory(Z, train)
                print("通过：训练/验证参数复用、逆转换")
                '''),
        ])),
        ("03_logistic_regularization", "逻辑回归与正则化", "C1 W3", lab("实验 03 · 逻辑回归与正则化", "目标：处理数值稳定性，并确认正则化不作用于偏置。", [
            ("1 · 稳定的 Sigmoid", "实现 sigmoid_stable(z)，支持标量与任意形状数组，保留数组形状。对于 ±1000 也不能发生溢出。", '''
                def sigmoid_stable(z):
                    raise NotImplementedError("TODO: 稳定的 Sigmoid")
                ''', '''
                with np.errstate(over='raise', invalid='raise'):
                    np.testing.assert_allclose(sigmoid_stable(np.array([-1000., 0., 1000.])), [0., 0.5, 1.])
                np.testing.assert_allclose(sigmoid_stable(0.), 0.5)
                assert np.asarray(sigmoid_stable(np.zeros((2, 3)))).shape == (2, 3)
                np.testing.assert_allclose(sigmoid_stable(np.array([-2., 2.])).sum(), 1.)
                print("通过：极端输入、标量、多维形状")
                '''),
            ("2 · 正则化梯度", "实现 logistic_gradient(X,y,w,b,lambda_)，返回 (dw,db)，形状分别 (n,) 与标量。使用平均二元交叉熵，正则项为 lambda_/(2m) × 权重平方和；不正则化 b。", '''
                def logistic_gradient(X, y, w, b, lambda_):
                    raise NotImplementedError("TODO: 带 L2 正则的梯度")
                ''', '''
                X = np.array([[1., 0.], [0., 2.]])
                y = np.array([0., 1.]); w = np.array([0.4, -0.2]); b = 0.3
                dw0, db0 = logistic_gradient(X, y, w, b, 0.)
                dw1, db1 = logistic_gradient(X, y, w, b, 3.)
                np.testing.assert_allclose(dw1 - dw0, [0.6, -0.3])
                np.testing.assert_allclose(db1, db0)
                d0, b0 = logistic_gradient(X, y, np.zeros(2), 0., 0.)
                np.testing.assert_allclose(d0, [0.25, -0.5]); np.testing.assert_allclose(b0, 0.)
                assert np.asarray(dw0).shape == (2,) and np.asarray(db0).ndim == 0
                X1 = np.array([[1.], [2.]])
                np.testing.assert_allclose(logistic_gradient(X1, np.ones(2), np.zeros(1), 0., 0.)[1], -0.5)
                print("通过：已知值、正则项、偏置不正则化")
                '''),
        ])),
        ("04_forward_softmax", "神经网络前向传播", "C2 W1–W2", lab("实验 04 · 神经网络前向传播", "目标：先用 NumPy 理解批量前向传播，再学习框架。", [
            ("1 · 批量 Dense 层", "实现 dense_batch(A,W,b,activation)，A 为 (m,n)，W 为 (n,k)，b 为 (k,)，activation 是可调用函数，返回 (m,k)。", '''
                def dense_batch(A, W, b, activation):
                    raise NotImplementedError("TODO: 批量 Dense 层")
                ''', '''
                A = np.array([[1., 2.], [-1., 0.]])
                W = np.array([[1., 0., -1.], [0., 2., 1.]])
                b = np.array([0., -1., 0.])
                out = dense_batch(A, W, b, lambda z: np.maximum(z, 0.))
                assert out.shape == (2, 3)
                np.testing.assert_allclose(out, [[1., 3., 1.], [0., 0., 1.]])
                np.testing.assert_allclose(dense_batch(A, W, b, lambda z: z), [[1., 3., 1.], [-1., -1., 1.]])
                print("通过：维度、激活函数、批量运算")
                '''),
            ("2 · 按行稳定 Softmax", "实现 softmax_rows(Z)，Z 为 (m,k)，返回每行的概率分布。对绝对值很大的 logits 也应稳定。不要修改输入。", '''
                def softmax_rows(Z):
                    raise NotImplementedError("TODO: 稳定的批量 Softmax")
                ''', '''
                Z = np.array([[1000., 1001., 1002.], [-1000., -1000., -1000.]])
                before = Z.copy()
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    P = softmax_rows(Z)
                assert P.shape == Z.shape and np.isfinite(P).all()
                assert (P >= 0).all() and (P <= 1).all()
                np.testing.assert_allclose(P.sum(axis=1), [1., 1.])
                np.testing.assert_allclose(P[0], [0.09003057, 0.24472847, 0.66524096], rtol=1e-6)
                np.testing.assert_allclose(P[1], [1/3, 1/3, 1/3])
                np.testing.assert_allclose(softmax_rows(Z + 12345.), P)
                np.testing.assert_array_equal(Z, before)
                print("通过：行归一化、已知概率、稳定性、平移不变性")
                '''),
        ])),
        ("05_model_selection", "数据划分与模型选择", "C2 W3", lab("实验 05 · 数据划分与模型选择", "目标：建立训练、验证、测试的分工，防止测试集参与选择。", [
            ("1 · 划分数据索引", "实现 split_indices(n,seed)，返回 train_idx、val_idx、test_idx。先随机打乱索引，再按 floor(0.6n)、floor(0.2n)、剩余数量切分；互不重叠，覆盖所有样本，同一 seed 可复现。", '''
                def split_indices(n, seed):
                    raise NotImplementedError("TODO: 无重复的数据划分")
                ''', '''
                a, b, c = split_indices(23, 42)
                assert [len(a), len(b), len(c)] == [13, 4, 6]
                np.testing.assert_array_equal(np.sort(np.concatenate([a, b, c])), np.arange(23))
                for x, y in ((a,b), (a,c), (b,c)):
                    assert not np.intersect1d(x, y).size
                for x, y in zip((a,b,c), split_indices(23, 42)):
                    np.testing.assert_array_equal(x,y)
                assert not np.array_equal(a, np.arange(13))
                assert not np.array_equal(a, split_indices(23, 7)[0])
                print("通过：比例、完整覆盖、无重叠、可复现")
                '''),
            ("2 · 只用验证误差选择模型", "实现 choose_model(validation_errors)，返回最小误差的索引；并列时选择最早的索引。本函数不能读取测试集。空列表抛出 ValueError。", '''
                def choose_model(validation_errors):
                    raise NotImplementedError("TODO: 验证集模型选择")
                ''', '''
                assert choose_model([0.30, 0.15, 0.20]) == 1
                assert choose_model([0.10, 0.10, 0.20]) == 0
                assert choose_model([0.25]) == 0
                try:
                    choose_model([])
                except ValueError:
                    pass
                else:
                    raise AssertionError("空列表必须抛出 ValueError")
                print("通过：最小验证误差、并列、空列表")
                '''),
            ("3 · 手动诊断偏差与方差", "下面三组分别给出训练/验证误差：A=(0.35,0.37)，B=(0.01,0.30)，C=(0.02,0.03)。假设任务可达到约 0.02 的误差。给每组提出一个优先实验并解释理由；说明为什么不能反复看测试误差选择超参数。", '''
                # TODO: 以字符串记录你的判断和下一步实验。
                diagnosis = None
                raise NotImplementedError("TODO: 误差诊断")
                ''', '''
                assert isinstance(diagnosis, str) and len(diagnosis.strip()) > 30
                print("格式通过；请自行核对解释是否有证据支持。")
                '''),
        ])),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not (source / ASSIGNMENTS[0][2]).is_file():
        raise SystemExit("Source is not the expected course repository")
    if output.exists():
        raise SystemExit("Output already exists; choose a new directory to protect completed work")
    if source == output or source in output.parents:
        raise SystemExit("Choose an output outside the original repository")
    output.mkdir(parents=True)
    manifest = {"source": str(source), "assignments": [], "labs": [], "original_notebook_hashes": {}}
    for slug, title, relative, dependency, topics in ASSIGNMENTS:
        src = source / relative
        manifest["original_notebook_hashes"][relative] = hashlib.sha256(src.read_bytes()).hexdigest()
        folder = output / "assignments" / slug
        folder.mkdir(parents=True)
        assets = copy_support(src.parent, folder)
        info = clean_assignment(src, folder / "assignment.ipynb", slug)
        if slug == "02_logistic_regression":
            patch_helper_callback(folder, "utils", "sig")
            helper = folder / "utils.py"
            helper.write_text(helper.read_text().replace(
                "z[i,j] = sig(np.dot(map_feature(u[i], v[j]), w) + b)",
                "z[i,j] = np.asarray(sig(np.dot(map_feature(u[i], v[j]), w) + b)).item()"))
        if slug == "05_model_evaluation":
            patch_helper_callback(folder, "assigment_utils", "eval_cat_err")
        if slug == "10_content_recommender":
            util = folder / "recsysNN_utils.py"
            util.write_text(util.read_text().replace("import pickle5 as pickle", "import pickle"))
        manifest["assignments"].append(dict(id=slug, title=title, notebook=f"assignments/{slug}/assignment.ipynb",
                                               source=relative, dependencies=dependency, topics=topics, assets=assets, **info))
    for slug, title, when, nb in make_labs():
        path = output / "labs" / f"{slug}.ipynb"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
        manifest["labs"].append(dict(id=slug, title=title, when=when, notebook=f"labs/{slug}.ipynb",
                                     exercise_cells=sum("exercise" in c.get("metadata", {}).get("tags", []) for c in nb["cells"])))
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    shutil.copy2(source / "LICENSE", output / "LICENSE")
    shutil.copy2(Path(__file__), output / "build_practice.py")
    verifier = Path(__file__).with_name("verify_practice.py")
    if verifier.exists():
        shutil.copy2(verifier, output / "verify_practice.py")
    (output / "requirements-base.txt").write_text("jupyterlab\nnumpy\nmatplotlib\nscipy\npandas\nscikit-learn\ntabulate\nipywidgets\nnetworkx\n")
    (output / "requirements-tensorflow.txt").write_text("-r requirements-base.txt\ntensorflow\n")
    (output / "requirements-rl-legacy.txt").write_text("-r requirements-tensorflow.txt\ngym[box2d]==0.24.1\nimageio\nimageio-ffmpeg\npillow\nstatsmodels\n")
    write_readme(output, manifest)
    print(json.dumps({"output": str(output), "assignments": len(manifest['assignments']), "labs": len(manifest['labs']),
                      "exercise_cells": sum(x['exercise_cells'] for x in manifest['assignments'] + manifest['labs'])}, ensure_ascii=False))


def write_readme(output, manifest):
    s = """# Machine Learning 无答案练习包

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

在 Jupyter 中按目录打开 notebook。每份作业与它的数据、图片、辅助模块放在一起；不要只移动单个 ipynb，也不要改变 notebook 的工作目录。Windows 激活命令为 `.venv\\Scripts\\activate`。

基础实验和正式作业 01、02、06、07、08 不需要 TensorFlow。后续神经网络等作业需要在 TensorFlow 支持的 Python 版本/环境中安装：

```bash
python -m pip install -r requirements-tensorflow.txt
```

当前机器的 Python 是 3.14，未安装 TensorFlow；未验证它与课程旧版代码的兼容性。请为 TensorFlow 作业使用独立环境或课程实验环境，不要为了第一天学习就安装强化学习全部依赖。

## 正式作业（原题、数据与测试）

| 顺序 | 作业 | 你需要完成 | 依赖 |
|---|---|---|---|
"""
    for i, x in enumerate(manifest["assignments"], 1):
        s += f"| {i:02} | [{x['title']}]({x['notebook']}) | {x['topics']} | {x['dependencies']} |\n"
    s += "\n## 基础实验（新增，无参考答案）\n\n| 实验 | 建议时间 |\n|---|---|\n"
    for x in manifest["labs"]:
        s += f"| [{x['title']}]({x['notebook']}) | {x['when']} |\n"
    s += """
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
"""
    (output / "README.md").write_text(s)


if __name__ == "__main__":
    main()
