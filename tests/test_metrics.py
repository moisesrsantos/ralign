import numpy as np
import pandas as pd
from ralign.metrics import RashomonAlignment


class FakePredict:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        # return constant prediction with same length
        n = len(X)
        return np.array([self.value] * n)


class FakeProba:
    def __init__(self, classes):
        self.classes = classes

    def predict(self, X):
        # simulate failure to force predict_proba
        raise RuntimeError("no predict")

    def predict_proba(self, X):
        n = len(X)
        # return uniform probabilities across classes
        k = len(self.classes)
        proba = np.ones((n, k)) / k
        return proba


def test_generate_uniform_samples_basic():
    df = pd.DataFrame({
        "num": [0.0, 1.0, 2.0],
        "bin": [0, 1, 0],
        "cat": ["a", "b", "a"],
    })
    gen = RashomonAlignment.generate_uniform_samples(df, n_samples=50, random_state=0)
    assert gen.shape == (50, 3)
    assert gen["num"].min() >= 0.0 - 1e-8
    assert gen["num"].max() <= 2.0 + 1e-8
    assert set(gen["bin"].unique()).issubset({0, 1})
    assert set(gen["cat"].unique()).issubset({"a", "b"})


def test_apply_models_to_generated_predict_and_proba():
    df = pd.DataFrame({"x": [0, 1, 2]})
    models = [FakePredict(1), FakeProba([0, 1, 2])]
    res = RashomonAlignment.apply_models_to_generated(models, df)
    assert 0 in res and 1 in res
    assert np.all(res[0] == 1)
    arr = np.asarray(res[1])
    # apply_models_to_generated returns hard predictions (argmax of proba)
    assert arr.ndim == 1 and arr.shape[0] == 3


def test_compare_models_alignment():
    # create a tiny df
    df = pd.DataFrame({"a": [0, 1]})
    # model 0 always predicts 0, model 1 always predicts 1
    m0 = FakePredict(0)
    m1 = FakePredict(1)
    res = RashomonAlignment.geometric(models=[m0, m1], df=df, feature_cols=["a"], n_samples=30, random_state=1)
    # since predictions differ, jaccard alignment should be 0.0
    assert res["alignment"] == 0.0


def test_rashomon_alignment_methods():
    a = np.array([0, 1, 1, 0])
    b = np.array([0, 1, 0, 0])
    j = RashomonAlignment.distributional(a, b, method="jaccard")
    k = RashomonAlignment.distributional(a, b, method="cohen_kappa")
    assert 0.0 <= j <= 1.0
    assert -1.0 <= k <= 1.0
