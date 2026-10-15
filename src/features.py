import numpy as np
import pandas as pd


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    if "koi_depth" in X and "koi_depth_err1" in X and "koi_depth_err2" in X:
        depth_safe = X["koi_depth"].replace(0, np.nan)
        X["depth_rel_err"] = (
            np.abs(X["koi_depth_err1"]) + np.abs(X["koi_depth_err2"])
        ) / (2 * depth_safe)

    if "koi_prad" in X and "koi_prad_err1" in X:
        prad_safe = X["koi_prad"].replace(0, np.nan)
        X["prad_rel_err"] = np.abs(X["koi_prad_err1"]) / prad_safe

    if "koi_srad" in X and "koi_steff" in X:
        X["star_luminosity_proxy"] = (X["koi_srad"] ** 2) * (
            (X["koi_steff"] / 5778) ** 4
        )

    return X
