import xgboost as xgb
import numpy as np
import time
from sklearn import datasets
from joblib import Memory
import pandas as pd
import argparse

memory = Memory('./cachedir', verbose=0)


# Contains a dataset in numpy format as well as the relevant objective and metric
class TestDataset:
    def __init__(self, name, Xy, objective
                 ):
        self.name = name
        self.objective = objective
        self.X, self.y = Xy

    def set_params(self, params_in):
        params_in['objective'] = self.objective
        if self.objective == "multi:softmax":
            params_in["num_class"] = int(np.max(self.y) + 1)
        return params_in

    def get_dmat(self):
        return xgb.DMatrix(self.X, self.y)

    def get_test_dmat(self, num_rows):
        rs = np.random.RandomState(432)
        return xgb.DMatrix(self.X[rs.randint(0, self.X.shape[0], size=num_rows), :])


@memory.cache
def train_model(dataset, max_depth, num_rounds):
    dmat = dataset.get_dmat()
    params = {'tree_method': 'gpu_hist', 'max_depth': max_depth, 'eta': 0.01}
    params = dataset.set_params(params)
    model = xgb.train(params, dmat, num_rounds, [(dmat, 'train')])
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
    def __init__(self, name, dataset, num_rounds, max_depth):
        self.name = name
        self.dataset = dataset
        self.num_rounds = num_rounds
        self.max_depth = max_depth
        print("Training " + name)
        self.xgb_model = train_model(dataset, max_depth, num_rounds)
        self.num_trees, self.num_leaves, self.average_depth = get_model_stats(self.xgb_model)


def run_benchmark(args):
    test_datasets = [
        TestDataset("covtype", datasets.fetch_covtype(return_X_y=True), "multi:softmax"),
        TestDataset("cal_housing", datasets.fetch_california_housing(return_X_y=True),
                    "reg:squarederror"),
        TestDataset("fashion_mnist", fetch_fashion_mnist(), "multi:softmax"),
        TestDataset("adult", fetch_adult(), "binary:logistic"),
        ]
    models = []
    for d in test_datasets:
        small_name = d.name + "-small"
        if small_name in args.model or args.model == "all":
            models.append(Model(small_name, d, 10, 3))
        med_name = d.name + "-med"
        if med_name in args.model or args.model == "all":
            models.append(Model(med_name, d, 100, 8))
        large_name = d.name + "-large"
        if large_name in args.model or args.model == "all":
            models.append(Model(large_name, d, 1000, 16))

    # get model statistics
    models_df = pd.DataFrame(
        columns=["model", "num_rounds", "num_trees", "num_leaves", "max_depth", "average_depth"])
    for m in models:
        models_df = models_df.append(
            {"model": m.name, "num_rounds": m.num_rounds, "num_trees": m.num_trees,
             "num_leaves": m.num_leaves, "max_depth": m.max_depth,
             "average_depth": m.average_depth},
            ignore_index=True)
    print("Model size:")
    print(models_df)

    predictors = ["cpu_predictor", "gpu_predictor"]
    # predictors = ["gpu_predictor"]
    test_rows = 10000
    df = pd.DataFrame(columns=["model", "test_rows", "cpu_time", "gpu_time", "speedup"])
    for m in models:
        dtest = m.dataset.get_test_dmat(test_rows)
        result_row = {"model": m.name, "test_rows": test_rows}
        for p in predictors:
            m.xgb_model.set_param({"predictor": p})
            start = time.perf_counter()
            xgb_shap = m.xgb_model.predict(dtest, pred_contribs=True)
            runtime = time.perf_counter() - start
            if p is "gpu_predictor":
                result_row["gpu_time"] = runtime
            else:
                result_row["cpu_time"] = runtime
            # Check result
            margin = m.xgb_model.predict(dtest, output_margin=True)

            if not np.allclose(np.sum(xgb_shap, axis=len(xgb_shap.shape) - 1), margin, 1e-1, 1e-1):
                print("Warning: Failed 1e-1 accuracy")
            if not np.allclose(np.sum(xgb_shap, axis=len(xgb_shap.shape) - 1), margin, 1e-3, 1e-3):
                print("Warning: Failed 1e-3 accuracy")

        result_row["speedup"] = result_row["cpu_time"] / result_row["gpu_time"]
        df = df.append(result_row,
                       ignore_index=True)
    print("Results:")
    print(df)


parser = argparse.ArgumentParser(description='GPUTreeShap benchmark')
parser.add_argument("-model", default="all", type=str,
                    help="The model to be used for benchmarking. 'all' for all datasets.")

parser.add_argument("-nrows", default=10000, type=int,
                    help=(
                        "Number of test rows."))

args = parser.parse_args()
run_benchmark(args)
