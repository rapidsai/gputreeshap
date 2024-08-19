# Copyright (c) 2020-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Memory
from sklearn import datasets

memory = Memory('./cachedir', verbose=0)


# Contains a dataset in numpy format as well as the relevant objective and metric
class TestDataset:
    def __init__(self, name, Xy, objective):
        self.name = name
        self.objective = objective
        self.X, self.y = Xy

    def set_params(self, params_in):
        params_in["objective"] = self.objective
        if self.objective == "multi:softmax":
            params_in["num_class"] = int(np.max(self.y) + 1)
        return params_in

    def get_dmat(self):
        return xgb.QuantileDMatrix(self.X, self.y, enable_categorical=True)

    def get_test_dmat(self, num_rows):
        rs = np.random.RandomState(432)
        if hasattr(self.X, "iloc"):
            x = self.X.iloc[rs.randint(0, self.X.shape[0], size=num_rows), :]
        else:
            x = self.X[rs.randint(0, self.X.shape[0], size=num_rows), :]
        return xgb.DMatrix(x, enable_categorical=True)


@memory.cache
def train_model(dataset: TestDataset, max_depth: int, num_rounds: int) -> xgb.Booster:
    dmat = dataset.get_dmat()
    params = {'tree_method': 'hist', "device": "gpu", 'max_depth': max_depth, 'eta': 0.01}
    params = dataset.set_params(params)
    model = xgb.train(params, dmat, num_rounds, evals=[(dmat, 'train')])
    return model


@memory.cache
def fetch_adult():
    X, y = datasets.fetch_openml("adult", return_X_y=True)
    y_binary = np.array([y_i != '<=50K' for y_i in y])
    return X, y_binary


@memory.cache
def fetch_fashion_mnist():
    X, y = datasets.fetch_openml("Fashion-MNIST", return_X_y=True)
    return X, y.astype(np.int64)


@memory.cache
def get_model_stats(model):
    depths = []
    for t in model.get_dump():
        for line in t.splitlines():
            if "leaf" in line:
                depths.append(line.count('\t'))
    return len(model.get_dump()), len(depths), np.mean(depths)


class Model:
    def __init__(
        self, name: str, dataset: TestDataset, num_rounds: int, max_depth: int
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.num_rounds = num_rounds
        self.max_depth = max_depth
        print("Training " + name)
        self.xgb_model = train_model(dataset, max_depth, num_rounds)
        self.num_trees, self.num_leaves, self.average_depth = get_model_stats(
            self.xgb_model
        )


def check_accuracy(shap, margin):
    shap = np.sum(shap, axis=len(shap.shape) - 1)

    if not np.allclose(shap, margin, 1e-1, 1e-1):
        print("Warning: Failed 1e-1 accuracy")


def get_models(model: str) -> list[Model]:
    test_datasets = [
        TestDataset("adult", fetch_adult(), "binary:logistic"),
        TestDataset("covtype", datasets.fetch_covtype(return_X_y=True), "multi:softmax"),
        TestDataset("cal_housing", datasets.fetch_california_housing(return_X_y=True),
                    "reg:squarederror"),
        TestDataset("fashion_mnist", fetch_fashion_mnist(), "multi:softmax"),
    ]

    models = []
    for d in test_datasets:
        small_name = d.name + "-small"
        if small_name in model or model == "all" or model == "small":
            models.append(Model(small_name, d, 10, 3))
        med_name = d.name + "-med"
        if med_name in model or model == "all" or model == "med":
            models.append(Model(med_name, d, 100, 8))
        large_name = d.name + "-large"
        if large_name in model or model == "all" or model == "large":
            models.append(Model(large_name, d, 1000, 16))
    return models


def print_model_stats(models, args):
    # get model statistics
    models_df = pd.DataFrame(
        columns=[
            "model",
            "num_rounds",
            "num_trees",
            "num_leaves",
            "max_depth",
            "average_depth",
        ]
    )
    for i, m in enumerate(models):
        df = pd.DataFrame.from_dict(
            {
                "model": [m.name],
                "num_rounds": [m.num_rounds],
                "num_trees": [m.num_trees],
                "num_leaves": [m.num_leaves],
                "max_depth": [m.max_depth],
                "average_depth": [m.average_depth],
            }
        )
        models_df = pd.concat([models_df, df])
    print(models_df)
    print("Writing model statistics to: " + args.out_models)
    models_df.to_csv(args.out_models, index=False)


def run_benchmark(args: argparse.Namespace) -> None:
    models = get_models(args.model)
    print_model_stats(models, args)

    devices = ["cpu", "gpu"]
    test_rows = args.nrows
    df = pd.DataFrame(
        columns=["model", "test_rows", "cpu_time(s)", "cpu_std", "gpu_time(s)", "gpu_std",
                 "speedup"])
    for m in models:
        dtest = m.dataset.get_test_dmat(test_rows)
        result_row = {"model": m.name, "test_rows": test_rows, "cpu_time(s)": 0.0}
        for p in devices:
            m.xgb_model.set_param({"device": p})
            samples = []
            for i in range(args.niter):
                start = time.perf_counter()
                if args.interactions:
                    xgb_shap = m.xgb_model.predict(dtest, pred_interactions=True)
                else:
                    xgb_shap = m.xgb_model.predict(dtest, pred_contribs=True)
                samples.append(time.perf_counter() - start)
            if p == "gpu":
                result_row["gpu_time(s)"] = np.mean(samples)
                result_row["gpu_std"] = np.std(samples)
            else:
                result_row["cpu_time(s)"] = np.mean(samples)
                result_row["cpu_std"] = np.std(samples)
            # Check result
            margin = m.xgb_model.predict(dtest, output_margin=True)
            check_accuracy(xgb_shap, margin)

        result_row["speedup"] = result_row["cpu_time(s)"] / result_row["gpu_time(s)"]
        df = pd.concat([df, pd.DataFrame.from_records([result_row])])
        print(df)
    print("Writing results to: " + args.out)
    df.to_csv(args.out, index=False)


def main():
    parser = argparse.ArgumentParser(description='GPUTreeShap benchmark')
    parser.add_argument("-model", default="all", type=str,
                        help="The model to be used for benchmarking. 'all' for all datasets.")

    parser.add_argument("-nrows", default=10000, type=int,
                        help=(
                            "Number of test rows."))
    parser.add_argument("-niter", default=5, type=int,
                        help=(
                            "Number of times to repeat the experiment."))
    parser.add_argument("-format", default="text", type=str,
                        help="Format of output tables. E.g. text,latex,csv")

    parser.add_argument("-out", default="results.csv", type=str)
    parser.add_argument("-interactions", default=False, type=bool)
    parser.add_argument("-out_models", default="models.csv", type=str)

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == '__main__':
    main()
