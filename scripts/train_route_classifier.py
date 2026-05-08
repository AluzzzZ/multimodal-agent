"""
训练轻量 route classifier 并导出 ONNX。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from src.modules.route_classifier import ROUTE_LABELS, RouteClassifierFeatureizer


LABEL_TO_ID = {label: idx for idx, label in enumerate(ROUTE_LABELS)}


class LinearRouteClassifier(nn.Module):
    """
    线性路由分类器（PyTorch 版本）。

    结构说明: 单层线性映射 (feature_dim -> num_labels) + softmax 激活，
    直接使用手工特征向量作为输入，不依赖深度神经网络。
    训练时支持带类别权重的交叉熵损失，以缓解类别不平衡问题。

    导出 ONNX 后可脱离 PyTorch 在 CPU 上运行，适合生产部署。
    """

    def __init__(self, feature_dim: int, num_labels: int, class_weights: Optional[torch.Tensor] = None):
        """
        初始化线性分类器。

        Args:
            feature_dim: 输入特征向量维度（对应 RouteClassifierFeatureizer 的 dim）
            num_labels: 分类类别数（固定为 3，对应 service/manual/mixed）
            class_weights: 类别权重张量，用于平衡训练时的类别不均衡
        """
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_labels)
        self.class_weights = class_weights

    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播。

        Args:
            features: 批量输入特征向量，形状 (batch, feature_dim)
            labels: 可选，训练时传入用于计算损失

        Returns:
            若传入 labels: 返回交叉熵损失标量
            若未传入 labels: 返回 softmax 概率分布 (batch, num_labels)
        """
        logits = self.classifier(features)
        if labels is not None and self.class_weights is not None:
            loss = nn.functional.cross_entropy(logits, labels, weight=self.class_weights)
            return loss
        return torch.softmax(logits, dim=-1)


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    """
    加载 JSONL 格式的数据集文件。

    JSONL 格式: 每行一个 JSON 对象，包含 text（问题文本）和 label（路由标签）字段。

    Args:
        path: JSONL 文件路径

    Returns:
        包含字典的列表，每个字典对应一行数据
    """
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def vectorize(rows: Sequence[Dict[str, str]], featureizer: RouteClassifierFeatureizer) -> Tuple[np.ndarray, np.ndarray]:
    """
    将数据集中的文本批量编码为特征向量和标签数组。

    Args:
        rows: 数据集行列表，每行包含 "text" 和 "label" 字段
        featureizer: 已初始化的特征化器实例

    Returns:
        (特征矩阵, 标签数组):
        - 特征矩阵形状: (N, feature_dim)，float32
        - 标签数组形状: (N,)，int64
    """
    features = []
    labels = []
    for row in rows:
        feature = featureizer.encode(row["text"])
        features.append(feature)
        labels.append(LABEL_TO_ID[row["label"]])
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    计算分类评估指标。

    指标说明:
    - accuracy: 整体分类准确率
    - macro_f1: 宏平均F1分数（三类的F1简单平均，平等对待每类）
    - mixed_f1: 混合路由(mixed)的F1分数（重点关注，因为mixed样本最少最难分）
    - service_f1: 客服路由(service)的F1分数

    macro_f1计算: 对每个类别分别计算tp/fp/fn，推导precision/recall/F1，
    再对所有类别求平均。适合类别不平衡场景。

    Args:
        probs: 模型预测概率数组 (N, num_labels)
        labels: 真实标签数组 (N,)

    Returns:
        包含accuracy/macro_f1/mixed_f1/service_f1的字典
    """
    preds = np.argmax(probs, axis=1)
    accuracy = float(np.mean(preds == labels)) if len(labels) else 0.0

    # 宏平均F1: 每个类别的F1求平均
    f1_by_label = {}
    for label_name, label_id in LABEL_TO_ID.items():
        tp = int(np.sum((preds == label_id) & (labels == label_id)))
        fp = int(np.sum((preds == label_id) & (labels != label_id)))
        fn = int(np.sum((preds != label_id) & (labels == label_id)))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        f1_by_label[label_name] = f1

    macro_f1_parts = [f1_by_label.get(label, 0.0) for label in ROUTE_LABELS]

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(float(np.mean(macro_f1_parts)), 4),
        "mixed_f1": round(f1_by_label.get("mixed", 0.0), 4),
        "service_f1": round(f1_by_label.get("service", 0.0), 4),
        "manual_f1": round(f1_by_label.get("manual", 0.0), 4),
    }


def compute_class_weights(labels: np.ndarray, num_labels: int) -> torch.Tensor:
    """计算类别权重，平衡数据集"""
    counts = Counter(labels)
    total = len(labels)
    # 使用 sqrt(n/max_count) 作为权重，减少过拟合
    max_count = max(counts.values())
    weights = []
    for i in range(num_labels):
        count = counts.get(i, 1)
        # 使用 sqrt 缩放，避免极端权重
        weight = (total / (num_labels * count)) ** 0.5
        weights.append(weight)
    weights = torch.tensor(weights, dtype=torch.float32)
    # 归一化
    weights = weights / weights.sum() * len(weights)
    return weights


def summarize_label_counts(rows: Sequence[Dict[str, str]]) -> Dict[str, int]:
    """统计样本标签分布，便于训练前后审计。"""
    counts = Counter(row["label"] for row in rows)
    return {label: int(counts.get(label, 0)) for label in ROUTE_LABELS}


def oversample_minority_rows(
    rows: Sequence[Dict[str, str]],
    seed: int = 42,
    target_ratio: float = 0.4,
) -> List[Dict[str, str]]:
    """
    对训练集中的少数类执行随机过采样。

    策略：
    - 仅处理训练集，不改动验证/测试集
    - 将 `service` / `mixed` 补齐到多数类的目标比例，而不是直接拉平
    - 例如 target_ratio=0.4 时，若 manual=1658，则少数类会被补到约 663 条
    - 采样采用固定随机种子，保证结果可复现
    """
    if not rows:
        return []

    grouped: Dict[str, List[Dict[str, str]]] = {label: [] for label in ROUTE_LABELS}
    passthrough: List[Dict[str, str]] = []
    for row in rows:
        label = row.get("label")
        if label in grouped:
            grouped[label].append(dict(row))
        else:
            passthrough.append(dict(row))

    label_sizes = {label: len(items) for label, items in grouped.items() if items}
    if not label_sizes:
        return [dict(row) for row in rows]

    majority_size = max(label_sizes.values())
    target_size = max(1, int(np.ceil(majority_size * target_ratio)))
    rng = random.Random(seed)
    balanced_rows: List[Dict[str, str]] = []

    for label in ROUTE_LABELS:
        items = grouped[label]
        if not items:
            continue
        balanced_rows.extend(items)
        desired_size = len(items)
        if label != "manual":
            desired_size = max(len(items), target_size)
        if len(items) < desired_size:
            extra = [dict(rng.choice(items)) for _ in range(desired_size - len(items))]
            balanced_rows.extend(extra)

    balanced_rows.extend(passthrough)
    rng.shuffle(balanced_rows)
    return balanced_rows


def run_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    labels: torch.Tensor,
    use_weighted_loss: bool = True,
) -> float:
    """
    执行一次训练 epoch。

    训练流程:
    1. 前向传播计算 logits
    2. 计算交叉熵损失（带类别权重）
    3. 反向传播更新参数

    Args:
        model: 待训练的模型
        optimizer: 优化器（支持 Adam 等）
        features: 训练特征张量，形状 (N, feature_dim)
        labels: 训练标签张量，形状 (N,)
        use_weighted_loss: 是否使用类别权重（默认 True）

    Returns:
        本 epoch 的平均损失值（标量）
    """
    model.train()
    if use_weighted_loss and hasattr(model, 'class_weights') and model.class_weights is not None:
        logits = model.classifier(features)
        loss = nn.functional.cross_entropy(logits, labels, weight=model.class_weights)
    else:
        logits = model.classifier(features)
        loss = nn.functional.cross_entropy(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


def predict_probs(model: nn.Module, features: torch.Tensor) -> np.ndarray:
    """
    使用模型对特征矩阵进行推理，返回预测概率。

    推理时使用 torch.no_grad() 上下文，禁用梯度计算以节省显存。

    Args:
        model: 训练好的 PyTorch 模型
        features: 输入特征张量，形状 (N, feature_dim)

    Returns:
        预测概率矩阵，形状 (N, num_labels)，numpy 数组
    """
    model.eval()
    with torch.no_grad():
        probs = model(features).cpu().numpy()
    return probs


def export_onnx(model: nn.Module, output_path: Path, feature_dim: int) -> None:
    """
    将PyTorch模型导出为ONNX格式。

    导出配置说明:
    - input_names/output_names: 定义输入输出张量名称，供ONNXRuntime引用
    - dynamic_axes: 定义批次维度(batch)为动态轴，使模型支持任意批次推理
    - opset_version=18: 使用ONNX算子集第18版，兼容ONNXRuntime 1.17+

    导出后的ONNX模型可直接用onnxruntime.InferenceSession加载，
    无需PyTorch依赖，适合生产部署。

    Args:
        model: 训练好的PyTorch模型
        output_path: ONNX文件输出路径
        feature_dim: 输入特征维度
    """
    dummy_input = torch.randn(1, feature_dim, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["features"],
        output_names=["probabilities"],
        dynamic_axes={"features": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version=18,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 route classifier 并导出 ONNX")
    parser.add_argument("--dataset-dir", default=str(settings.route_classifier_data_path / "dataset"))
    parser.add_argument("--model-dir", default=str(settings.route_classifier_model_dir))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--no-class-weight", action="store_true", help="禁用类别权重平衡")
    parser.add_argument("--no-oversample", action="store_true", help="禁用训练集少数类过采样")
    parser.add_argument("--oversample-seed", type=int, default=42, help="过采样随机种子")
    parser.add_argument(
        "--oversample-target-ratio",
        type=float,
        default=0.3,
        help="少数类过采样目标比例，相对于多数类样本数，推荐 0.3-0.5",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(dataset_dir / "train.jsonl")
    val_rows = load_jsonl(dataset_dir / "val.jsonl")
    test_rows = load_jsonl(dataset_dir / "test.jsonl")
    train_distribution_before = summarize_label_counts(train_rows)
    oversample_enabled = not args.no_oversample
    if oversample_enabled:
        train_rows = oversample_minority_rows(
            train_rows,
            seed=args.oversample_seed,
            target_ratio=args.oversample_target_ratio,
        )
    train_distribution_after = summarize_label_counts(train_rows)
    print(f"训练集标签分布(原始): {train_distribution_before}")
    print(f"训练集标签分布(过采样后): {train_distribution_after}")

    featureizer = RouteClassifierFeatureizer(settings.route_classifier_feature_dim)
    train_x, train_y = vectorize(train_rows, featureizer)
    val_x, val_y = vectorize(val_rows, featureizer)
    test_x, test_y = vectorize(test_rows, featureizer)

    # 计算类别权重
    use_class_weight = not args.no_class_weight
    class_weights = None
    if use_class_weight:
        class_weights = compute_class_weights(train_y, len(ROUTE_LABELS))
        print(f"类别权重: {dict(zip(ROUTE_LABELS, class_weights.tolist()))}")

    model = LinearRouteClassifier(
        feature_dim=train_x.shape[1],
        num_labels=len(ROUTE_LABELS),
        class_weights=class_weights,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_features = torch.from_numpy(train_x)
    train_labels = torch.from_numpy(train_y)
    val_features = torch.from_numpy(val_x)

    best_state = None
    best_macro_f1 = -1.0
    best_service_f1 = 0.0
    best_mixed_f1 = 0.0
    patience = 15
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        loss = run_epoch(model, optimizer, train_features, train_labels, use_weighted_loss=True)
        val_probs = predict_probs(model, val_features)
        val_metrics = compute_metrics(val_probs, val_y)
        history.append({"epoch": epoch, "loss": round(loss, 6), **val_metrics})

        # 早停：监控 macro_f1
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_service_f1 = val_metrics.get("service_f1", 0.0)
            best_mixed_f1 = val_metrics.get("mixed_f1", 0.0)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter == 0:
            print(f"Epoch {epoch}: loss={loss:.4f}, macro_f1={val_metrics['macro_f1']:.4f}, "
                  f"accuracy={val_metrics['accuracy']:.4f}, mixed_f1={val_metrics['mixed_f1']:.4f}")

        if patience_counter >= patience:
            print(f"早停于 epoch {epoch}，最佳 macro_f1={best_macro_f1:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"加载最佳模型，macro_f1={best_macro_f1:.4f}, service_f1={best_service_f1:.4f}, mixed_f1={best_mixed_f1:.4f}")

    test_probs = predict_probs(model, torch.from_numpy(test_x))
    test_metrics = compute_metrics(test_probs, test_y)

    torch.save(model.state_dict(), model_dir / "route_classifier.pt")
    export_onnx(model, model_dir / "route_classifier.onnx", feature_dim=train_x.shape[1])

    manifest = {
        "backend": "onnx",
        "feature_dim": int(train_x.shape[1]),
        "labels": ROUTE_LABELS,
        "class_weight_enabled": use_class_weight,
        "oversample_enabled": oversample_enabled,
        "oversample_seed": args.oversample_seed if oversample_enabled else None,
        "oversample_target_ratio": args.oversample_target_ratio if oversample_enabled else None,
        "train_distribution_before": train_distribution_before,
        "train_distribution_after": train_distribution_after,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "high_threshold": settings.route_classifier_high_threshold,
        "low_threshold": settings.route_classifier_low_threshold,
        "metrics": {
            "val_best_macro_f1": best_macro_f1,
            "val_service_f1": best_service_f1,
            "val_mixed_f1": best_mixed_f1,
            "test": test_metrics,
        },
        "history_tail": history[-10:],
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"模型已导出到: {model_dir}")


if __name__ == "__main__":
    main()
