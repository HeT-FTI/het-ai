# """
# 案例六：贝叶斯推断（PyMC，WAIC 优化）
# 框架: PyMC + ArviZ  |  任务: 贝叶斯参数估计  |  导出: NetCDF
# """
# import numpy as np
#
# from het_ai.studio import BaseTrainer, DataBundle, TrainConfig
#
#
# class BayesianMixtureTrainer(BaseTrainer):
#
#     # ── 数据 ─────────────────────────────────────────────────────
#
#     def load_data(self, dvc_data_root: str) -> DataBundle:
#         import pandas as pd
#
#         df  = pd.read_csv(
#             f"{dvc_data_root}/repository_0/files/label_data/obs.csv"
#         )
#         obs = df['value'].values.astype(np.float64)
#         return DataBundle(
#             splits={'all': {'obs': obs}},
#             meta={'n_obs': len(obs)},
#         )
#
#     def mock_data(self) -> DataBundle:
#         rng = np.random.default_rng(0)
#         obs = rng.normal(loc=2.5, scale=1.2, size=80)
#         return DataBundle(
#             splits={'all': {'obs': obs}},
#             meta={'n_obs': 80},
#         )
#
#     # ── 训练 ─────────────────────────────────────────────────────
#
#     @BaseTrainer.search(
#         mu_sigma    = BaseTrainer.TunableFloat(0.5, 10.0),
#         sigma_prior = BaseTrainer.TunableFloat(0.1, 5.0),
#         draws       = BaseTrainer.TunableInt(500, 2000, step=500),
#         tune        = BaseTrainer.TunableInt(200, 1000, step=200),
#     )
#     def train(self, data: DataBundle, mu_sigma, sigma_prior, draws, tune):
#         import pymc as pm
#         import arviz as az
#
#         obs = data.splits['all']['obs']
#
#         with pm.Model():
#             mu    = pm.Normal('mu',    mu=0,   sigma=mu_sigma)
#             sigma = pm.HalfNormal('sigma',     sigma=sigma_prior)
#             _     = pm.Normal('obs',   mu=mu,  sigma=sigma, observed=obs)
#             idata = pm.sample(
#                 draws=int(draws),
#                 tune=int(tune),
#                 progressbar=False,
#                 return_inferencedata=True,
#                 random_seed=self.config.random_state,
#             )
#
#         waic_score = float(az.waic(idata).elpd_waic)
#         return waic_score, idata
#
#     # ── 导出 ─────────────────────────────────────────────────────
#
#     def export_model(self, artifact, export_dir: str) -> str:
#         if artifact is None:
#             return export_dir
#         path = f"{export_dir}/inference_data.nc"
#         artifact.to_netcdf(path)
#         return path
#
#     def on_study_end(self, study, best_trial) -> dict:
#         return {'inference_framework': 'PyMC', 'metric': 'WAIC'}
#
#
# def main(dvc_data_root: str = "dvc_data"):
#     config  = TrainConfig(direction='maximize')
#     trainer = BayesianMixtureTrainer(config)
#     trainer.dry_run(dvc_data_root)
#     return trainer.run(dvc_data_root).to_tuple()

"""
案例六：贝叶斯推断（PyMC，轻量级单目标优化）
框架: PyMC  |  任务: 贝叶斯参数估计  |  导出: NetCDF

说明：
1. dry_run 仍然真实走 pm.sample()，但会自动缩小采样规模以加快测试。
2. 为了提升兼容性与稳定性，不再依赖 arviz.waic。
3. 优化指标改为 posterior sample_stats 中 lp 的均值（越大越好）。
"""
import numpy as np
from het_ai.mlflow import MLflowConfig
from het_ai.studio import BaseTrainer, DataBundle, TrainConfig, TrainResult

class BayesianMixtureTrainer(BaseTrainer):

    # ── 数据 ─────────────────────────────────────────────────────

    def load_data(self, dvc_data_root: str) -> DataBundle:
        if dvc_data_root == '__mock__':
            return self.mock_data()

        import pandas as pd

        df = pd.read_csv(
            f"{dvc_data_root}/repository_0/files/label_data/obs.csv"
        )
        obs = df["value"].values.astype(np.float64)
        return DataBundle(
            splits={"all": {"obs": obs}},
            meta={"n_obs": len(obs)},
        )

    def mock_data(self) -> DataBundle:
        rng = np.random.default_rng(0)
        obs = rng.normal(loc=2.5, scale=1.2, size=80)
        return DataBundle(
            splits={"all": {"obs": obs}},
            meta={"n_obs": 80},
        )

    # ── 内部评分函数 ─────────────────────────────────────────────

    @staticmethod
    def _score_inferencedata(idata) -> float:
        """
        从 InferenceData 中提取一个稳定、可比较的单目标分数。
        这里使用 sample_stats["lp"] 的均值，越大越好。

        相比 az.waic：
        - 更轻量
        - 兼容性更高
        - 不依赖额外 log_likelihood 计算
        """
        try:
            lp = idata.sample_stats["lp"].values
            return float(lp.mean())
        except Exception as e:
            raise RuntimeError(
                f"无法从 InferenceData 中提取 sample_stats['lp'] 作为评分指标: {e}"
            )

    # ── 训练 ─────────────────────────────────────────────────────

    @BaseTrainer.search(
        mu_sigma    = BaseTrainer.TunableFloat(0.5, 10.0),
        sigma_prior = BaseTrainer.TunableFloat(0.1, 5.0),
        draws       = BaseTrainer.TunableInt(500, 2000, step=500),
        tune        = BaseTrainer.TunableInt(200, 1000, step=200),
    )
    def train(self, data: DataBundle, mu_sigma, sigma_prior, draws, tune):
        import pymc as pm
        from sklearn.model_selection import train_test_split
        np.random.seed(self.config.random_state)

        obs = data.splits["all"]["obs"]
        obs_train, obs_val = train_test_split(
            obs,
            test_size=0.2,
            random_state=self.config.random_state,
            shuffle=True,
        )

        # ── dry_run 模式下自动缩小采样规模，提升测试速度 ───────────
        if getattr(self, "_in_dry_run", False):
            draws = min(int(draws), 50)
            tune  = min(int(tune), 50)
            chains = 1
            cores = 1
        else:
            draws = int(draws)
            tune  = int(tune)
            chains = 2
            cores = 1

        with pm.Model():
            mu = pm.Normal("mu", mu=0.0, sigma=mu_sigma)
            sigma = pm.HalfNormal("sigma", sigma=sigma_prior)
            _ = pm.Normal("obs", mu=mu, sigma=sigma, observed=obs_train)

            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                progressbar=False,
                return_inferencedata=True,
                random_seed=self.config.random_state,
                compute_convergence_checks=False,  # dry_run/测试场景减少额外开销
            )

        score = self._score_inferencedata(idata)
        mu_samples = idata.posterior["mu"].values.reshape(-1)
        sigma_samples = np.clip(
            idata.posterior["sigma"].values.reshape(-1),
            1e-6,
            None,
        )

        def _nll(x, mu, sigma):
            return 0.5 * np.log(2.0 * np.pi * sigma * sigma) + 0.5 * ((x - mu) / sigma) ** 2

        train_loss_history = []
        val_loss_history = []
        for epoch, (mu_s, sigma_s) in enumerate(zip(mu_samples, sigma_samples)):
            train_loss = float(np.mean(_nll(obs_train, mu_s, sigma_s)))
            val_loss = float(np.mean(_nll(obs_val, mu_s, sigma_s)))
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            if self.report(step=epoch, value=-val_loss):
                break

        return score, idata, {
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
        }

    # ── 导出 ─────────────────────────────────────────────────────

    def export_model(self, artifact, export_dir: str) -> str:
        if artifact is None:
            return export_dir
        path = f"{export_dir}/inference_data.nc"
        try:
            artifact.to_netcdf(path, engine="scipy")
            return path
        except Exception:
            # Fall back to a lightweight NumPy archive if NetCDF backends are unavailable.
            fallback_path = f"{export_dir}/inference_data_fallback.npz"
            np.savez_compressed(
                fallback_path,
                mu=artifact.posterior["mu"].values,
                sigma=artifact.posterior["sigma"].values,
                lp=artifact.sample_stats["lp"].values,
            )
            return fallback_path

    def on_study_end(self, study, best_trial) -> dict:
        return {
            "inference_framework": "PyMC",
            "metric": "mean_log_posterior",
        }

    def before_mlflow_log(self, result: TrainResult):
        # 在 MLflow 记录之前附加额外的文件或信息
        result.artifact_file_paths.append("./tests/cases/case6_pymc_bayesian.py")
        return result

def main(dvc_data_root: str = "__mock__", mlflow_config: MLflowConfig | None = None):
    config  = TrainConfig(
        n_trials=10,
        mlflow=MLflowConfig(
            tracking_uri="http://10.12.8.110:5000",
            experiment_name="case6_pymc_bayesian",
        ),
    )
    trainer = BayesianMixtureTrainer(config)
    result = trainer.run(dvc_data_root)
    return result.to_tuple()

if __name__ == "__main__":
    print(main())
