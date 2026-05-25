# """
# 一键模拟运行所有 8 个任务类型的 dry_run。
# 无需生产环境（无需真实数据、DVC、训练服务器、容器）。
#
# 运行方式:
#     python simulate_all.py
#
# 退出码:
#     0 = 全部通过
#     1 = 存在失败
# """
# import sys
# import time
# import traceback
#
# from het_ai.studio import TrainConfig
#
# from tests.cases.case1_torch_tabular import PytorchTabularTrainer
# from tests.cases.case2_torch_multitarget import MultiTargetTrainer
# from tests.cases.case3_tf_image import TFImageTrainer
# from tests.cases.case4_sklearn_gbt import SklearnGBTrainer
# from tests.cases.case5_clustering import ClusteringTrainer
# from tests.cases.case6_pymc_bayesian import BayesianMixtureTrainer
# from tests.cases.case7_numpy_logistic import NumpyLogisticTrainer
# from tests.cases.case8_blackbox import BlackBoxTrainer
#
#
# CASES = [
#     ("案例一  PyTorch 表格单目标",     PytorchTabularTrainer),
#     ("案例二  PyTorch 多目标分类",     MultiTargetTrainer),
#     ("案例三  TensorFlow 图像分类",    TFImageTrainer),
#     ("案例四  sklearn 梯度提升",       SklearnGBTrainer),
#     ("案例五  无监督聚类（双目标）",   ClusteringTrainer),
#     ("案例六  PyMC 贝叶斯推断",        BayesianMixtureTrainer),
#     ("案例七  纯 NumPy 手写模型",      NumpyLogisticTrainer),
#     ("案例八  黑盒外部进程",           BlackBoxTrainer),
# ]
#
# SEP  = "=" * 64
# SEP2 = "-" * 64
#
#
# def run_all() -> bool:
#     print(f"\n{SEP}")
#     print("  MLOps Trainer Framework  —  Dry Run 模拟测试")
#     print(SEP)
#
#     results = []
#
#     for name, TrainerClass in CASES:
#         print(f"\n{SEP2}")
#         print(f"  {name}")
#         print(SEP2)
#         t0 = time.time()
#         try:
#             # n_trials 设为 2，仅影响正式 run()，dry_run 不受此参数影响
#             config  = TrainConfig(n_trials=2)
#             trainer = TrainerClass(config)
#             report  = trainer.dry_run()
#
#             score       = report['score']
#             elapsed     = report['elapsed']
#             export_path = report['export_path'] or '（无导出产物）'
#
#             status = "✅ PASS"
#             detail = (
#                 f"score={score}  "
#                 f"elapsed={elapsed:.2f}s  "
#                 f"export={export_path}"
#             )
#         except Exception as e:
#             status = "❌ FAIL"
#             detail = f"{type(e).__name__}: {e}"
#             traceback.print_exc()
#
#         wall = time.time() - t0
#         print(f"  {status}  [{wall:.1f}s]  {detail}")
#         results.append((name, status, detail))
#
#     # ── 汇总 ─────────────────────────────────────────────────────
#     passed = sum(1 for _, s, _ in results if 'PASS' in s)
#     failed = len(results) - passed
#
#     print(f"\n{SEP}")
#     print("  汇总")
#     print(SEP)
#     for name, status, _ in results:
#         print(f"  {status}  {name}")
#     print(SEP)
#     print(f"  通过: {passed}/{len(CASES)}    失败: {failed}/{len(CASES)}")
#     print(SEP)
#
#     return failed == 0
#
#
# if __name__ == "__main__":
#     sys.exit(0 if run_all() else 1)


import time

from het_ai.studio import TrainConfig

from tests.cases.case1_torch_tabular import PytorchTabularTrainer
from tests.cases.case2_torch_multitarget import MultiTargetTrainer
from tests.cases.case3_tf_image import TFImageTrainer
from tests.cases.case4_sklearn_gbt import SklearnGBTrainer
from tests.cases.case5_clustering import ClusteringTrainer
from tests.cases.case6_pymc_bayesian import BayesianMixtureTrainer
from tests.cases.case7_numpy_logistic import NumpyLogisticTrainer
from tests.cases.case8_blackbox import BlackBoxTrainer


CASES = [
    ("案例一 PyTorch 表格单目标", PytorchTabularTrainer),
    ("案例二 PyTorch 多目标分类", MultiTargetTrainer),
    ("案例三 TensorFlow 图像分类", TFImageTrainer),
    ("案例四 sklearn 梯度提升", SklearnGBTrainer),
    ("案例五 无监督聚类（双目标）", ClusteringTrainer),
    ("案例六 PyMC 贝叶斯推断", BayesianMixtureTrainer),
    ("案例七 纯 NumPy 手写模型", NumpyLogisticTrainer),
    ("案例八 黑盒外部进程", BlackBoxTrainer),
]


def _run_case(trainer_cls):
    config = TrainConfig(n_trials=2)
    trainer = trainer_cls(config)
    report = trainer.dry_run()
    assert "score" in report
    assert "elapsed" in report
    return report


def test_all_cases_smoke():
    """
    一个总冒烟测试。
    任一 case 失败，整个测试失败。
    """
    for _, trainer_cls in CASES:
        report = _run_case(trainer_cls)
        assert report is not None


# 更推荐：参数化，pytest 会显示 8 个独立用例
import pytest

@pytest.mark.parametrize("case_name,trainer_cls", CASES)
def test_each_case_dry_run(case_name, trainer_cls):
    report = _run_case(trainer_cls)
    assert report["elapsed"] >= 0
