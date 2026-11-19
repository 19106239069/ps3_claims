# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from dask_ml.preprocessing import Categorizer
from glum import GeneralizedLinearRegressor, TweedieDistribution
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from ps3.data import create_sample_split, load_transform


# %%
# load data
df = load_transform()
print("Dataset shape:", df.shape)
print(df.head())
df.info()

# %%
# Train benchmark tweedie model. This is entirely based on the glum tutorial.
weight = df["Exposure"].values
df["PurePremium"] = df["ClaimAmountCut"] / df["Exposure"]
y = df["PurePremium"]
# TODO: Why do you think, we divide by exposure here to arrive at our outcome variable?
# The longer the insurance coverage period is, 
# the more risk exposure information the policy usually contains.

# TODO: use your create_sample_split function here
# df = create_sample_split(...)
df = create_sample_split(
    df,
    id_column="IDpol",
    training_frac=0.8
)

train = np.flatnonzero(df["sample"].values == "train")
test = np.flatnonzero(df["sample"].values == "test")

y_train_t = y.iloc[train]
y_test_t = y.iloc[test]

w_train_t = weight[train]
w_test_t = weight[test]

df_train = df.iloc[train].copy()
df_test = df.iloc[test].copy()

print("Training set shape:", df_train.shape)
print("Testing set shape:", df_test.shape)

print("Sample distribution:")
print(df["sample"].value_counts())

categoricals = [
    "VehBrand",
    "VehGas",
    "Region",
    "Area",
    "DrivAge",
    "VehAge",
]

predictors = categoricals + [
    "BonusMalus",
    "VehPower",
    "Density",
]

glm_categorizer = Categorizer(columns=categoricals)

X_train_t = glm_categorizer.fit_transform(
    df_train[predictors]
)

X_test_t = glm_categorizer.transform(
    df_test[predictors]
)

TweedieDist = TweedieDistribution(1.5)

def weighted_deviance(y_true, y_pred, weights):
    return (
        TweedieDist.deviance(
            y_true,
            y_pred,
            sample_weight=weights
        )
        / np.sum(weights)
    )

t_glm1 = GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True)
t_glm1.fit(X_train_t, y_train_t, sample_weight=w_train_t)


pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm1.intercept_], t_glm1.coef_))},
    index=["intercept"] + t_glm1.feature_names_,
).T

df_test["pp_t_glm1"] = t_glm1.predict(X_test_t)
df_train["pp_t_glm1"] = t_glm1.predict(X_train_t)

train_deviance_glm1 = weighted_deviance(
    y_train_t, df_train["pp_t_glm1"], w_train_t
)
test_deviance_glm1 = weighted_deviance(
    y_test_t, df_test["pp_t_glm1"], w_test_t
)

print(
    "training loss t_glm1:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm1"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm1:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm1"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * t_glm1.predict(X_test_t)),
    )
)
# %%
# TODO: Let's add splines for BonusMalus and Density and use a Pipeline.
# Steps: 
# 1. Define a Pipeline which chains a StandardScaler and SplineTransformer. 
#    Choose knots="quantile" for the SplineTransformer and make sure, we 
#    are only including one intercept in the final GLM. 
# 2. Put the transforms together into a ColumnTransformer. Here we use OneHotEncoder for the categoricals.
# 3. Chain the transforms together with the GLM in a Pipeline.

# Let's put together a pipeline
numeric_spline_cols = [
    "BonusMalus",
    "VehPower",
]

linear_cols = ["Density"]

spline_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "spline",
            SplineTransformer(
                knots="quantile",
                n_knots=5,
                degree=3,
                include_bias=False
            )
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("spline", spline_pipeline, numeric_spline_cols),
        ("linear", "passthrough", linear_cols),
        (
            "cat",
            OneHotEncoder(
                sparse_output=False,
                drop="first",
                handle_unknown="ignore"
            ),
            categoricals
        ),
    ],
    remainder="drop"
)

preprocessor.set_output(transform="pandas")

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "estimate",
            GeneralizedLinearRegressor(
                family=TweedieDist,
                l1_ratio=1,
                fit_intercept=True
            )
        ),
    ]
)

# let's have a look at the pipeline
model_pipeline

# let's check that the transforms worked
model_pipeline[:-1].fit_transform(df_train)

model_pipeline.fit(df_train, y_train_t, estimate__sample_weight=w_train_t)

pd.DataFrame(
    {
        "coefficient": np.concatenate(
            ([model_pipeline[-1].intercept_], model_pipeline[-1].coef_)
        )
    },
    index=["intercept"] + model_pipeline[-1].feature_names_,
).T

df_test["pp_t_glm2"] = model_pipeline.predict(df_test)
df_train["pp_t_glm2"] = model_pipeline.predict(df_train)

train_deviance_glm2 = weighted_deviance(
    y_train_t, df_train["pp_t_glm2"], w_train_t
)
test_deviance_glm2 = weighted_deviance(
    y_test_t, df_test["pp_t_glm2"], w_test_t
)

print(
    "training loss t_glm2:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm2"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm2:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm2"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_glm2"]),
    )
)

# %%
# TODO: Let's use a GBM instead as an estimator.
# Steps
# 1: Define the modelling pipeline. Tip: This can simply be a LGBMRegressor based on X_train_t from before.
# 2. Make sure we are choosing the correct objective for our estimator.

# %%
# Train the initial LightGBM model

lgbm_base = LGBMRegressor(
    objective="tweedie",
    tweedie_variance_power=1.5,
    learning_rate=0.05,
    n_estimators=300,
    num_leaves=15,
    max_depth=6,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

lgbm_base.fit(
    X_train_t,
    y_train_t,
    sample_weight=w_train_t
)

df_train["pp_t_lgbm_base"] = lgbm_base.predict(X_train_t)
df_test["pp_t_lgbm_base"] = lgbm_base.predict(X_test_t)

train_deviance_lgbm_base = weighted_deviance(
    y_train_t,
    df_train["pp_t_lgbm_base"],
    w_train_t
)

test_deviance_lgbm_base = weighted_deviance(
    y_test_t,
    df_test["pp_t_lgbm_base"],
    w_test_t
)

print("Training loss LGBM:", train_deviance_lgbm_base)
print("Testing loss LGBM:", test_deviance_lgbm_base)

# %%
# TODO: Let's tune the LGBM to reduce overfitting.
# Steps:
# 1. Define a `GridSearchCV` object with our lgbm pipeline/estimator. Tip: Parameters for a specific step of the pipeline
# can be passed by <step_name>__param. 

# Note: Typically we tune many more parameters and larger grids,
# but to save compute time here, we focus on getting the learning rate
# and the number of estimators somewhat aligned -> tune learning_rate and n_estimators

weight_by_index = pd.Series(
    w_train_t,
    index=X_train_t.index
)


# Define a weighted Tweedie deviance scorer
def negative_weighted_tweedie_deviance(
    estimator,
    X_valid,
    y_valid
):
    y_pred = estimator.predict(X_valid)

    valid_weights = weight_by_index.loc[X_valid.index].to_numpy()

    return -weighted_deviance(
        y_valid,
        y_pred,
        valid_weights
    )


# Define the parameter grid
param_grid = {
    "n_estimators": [200, 400],
    "learning_rate": [0.03, 0.05],
    "num_leaves": [7, 15],
}


# Set up GridSearchCV
lgbm_search = GridSearchCV(
    estimator=lgbm_base,
    param_grid=param_grid,
    scoring=negative_weighted_tweedie_deviance,
    cv=KFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    ),
    n_jobs=1,
    refit=True,
    verbose=1,
    return_train_score=True
)


# Fit the grid search
lgbm_search.fit(
    X_train_t,
    y_train_t,
    sample_weight=w_train_t
)


# Display the best parameters
print("Best parameters:")
print(lgbm_search.best_params_)

print("Best CV deviance:")
print(-lgbm_search.best_score_)


# Get the best fitted model
best_lgbm = lgbm_search.best_estimator_


# Make predictions
df_train["pp_t_lgbm"] = best_lgbm.predict(X_train_t)
df_test["pp_t_lgbm"] = best_lgbm.predict(X_test_t)


# Calculate train and test deviance
train_deviance_lgbm = weighted_deviance(
    y_train_t,
    df_train["pp_t_lgbm"],
    w_train_t
)

test_deviance_lgbm = weighted_deviance(
    y_test_t,
    df_test["pp_t_lgbm"],
    w_test_t
)

print("Best LGBM training deviance:", train_deviance_lgbm)
print("Best LGBM testing deviance:", test_deviance_lgbm)


# Compare total claim amounts
print(
    "Total claim amount on test set:",
    "observed =", df_test["ClaimAmountCut"].sum(),
    "predicted =",
    np.sum(df_test["Exposure"] * df_test["pp_t_lgbm"])
)


# Display cross-validation results
cv_results = pd.DataFrame(lgbm_search.cv_results_)

# Convert negative scores back to positive Deviance
cv_results["mean_train_deviance"] = -cv_results["mean_train_score"]
cv_results["mean_validation_deviance"] = -cv_results["mean_test_score"]

display(
    cv_results[
        [
            "param_n_estimators",
            "param_learning_rate",
            "param_num_leaves",
            "mean_train_deviance",
            "mean_validation_deviance",
            "rank_test_score"
        ]
    ]
)

# %%
# Let's compare the sorting of the pure premium predictions
# %%
# Compare all models
results = pd.DataFrame(
    [
        {
            "Model": "Tweedie GLM Benchmark",
            "Train Deviance": train_deviance_glm1,
            "Test Deviance": test_deviance_glm1,
        },
        {
            "Model": "Tweedie GLM Splines",
            "Train Deviance": train_deviance_glm2,
            "Test Deviance": test_deviance_glm2,
        },
        {
            "Model": "LightGBM Base",
            "Train Deviance": train_deviance_lgbm_base,
            "Test Deviance": test_deviance_lgbm_base,
        },
        {
            "Model": "LightGBM Tuned",
            "Train Deviance": train_deviance_lgbm,
            "Test Deviance": test_deviance_lgbm,
        },
    ]
)

results["Train-Test Gap"] = (
    results["Test Deviance"] - results["Train Deviance"]
)

results["Test Improvement (%)"] = (
    (test_deviance_glm1 - results["Test Deviance"])
    / test_deviance_glm1 * 100
)

results = results.sort_values("Test Deviance")

display(results)

# Source: https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
def lorenz_curve(y_true, y_pred, exposure):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount


fig, ax = plt.subplots(figsize=(8, 8))

for label, y_pred in [
    ("LGBM Tuned", df_test["pp_t_lgbm"]),
    ("GLM Benchmark", df_test["pp_t_glm1"]),
    ("GLM Splines", df_test["pp_t_glm2"]),
]:
    ordered_samples, cum_claims = lorenz_curve(
        df_test["PurePremium"], y_pred, df_test["Exposure"]
    )
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    label += f" (Gini index: {gini: .3f})"
    ax.plot(ordered_samples, cum_claims, linestyle="-", label=label)

# Oracle model: y_pred == y_test
ordered_samples, cum_claims = lorenz_curve(
    df_test["PurePremium"], df_test["PurePremium"], df_test["Exposure"]
)
gini = 1 - 2 * auc(ordered_samples, cum_claims)
label = f"Oracle (Gini index: {gini: .3f})"
ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray", label=label)

# Random baseline
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
ax.set(
    title="Lorenz Curves",
    xlabel="Fraction of policyholders\n(ordered by model from safest to riskiest)",
    ylabel="Fraction of total claim amount",
)
ax.legend(loc="upper left")
plt.show()

# %%
