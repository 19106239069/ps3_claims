# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from dask_ml.preprocessing import Categorizer
from glum import GeneralizedLinearRegressor, TweedieDistribution
import lightgbm as lgb
from lightgbm import LGBMRegressor
import dalex as dx
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from evaluation import evaluate_predictions
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
# ============================================================
# PS 4 - Exercise 1: Monotonicity constraints
# ============================================================

# Step 1: Plot exposure-weighted average pure premium
# by BonusMalus group

bm_summary = (
    df_train.loc[df_train["Exposure"] > 0]
    .groupby("BonusMalus", as_index=False)
    .agg(
        total_claim_amount=("ClaimAmountCut", "sum"),
        total_exposure=("Exposure", "sum"),
        n_policies=("IDpol", "count")
    )
)

# Calculate exposure-weighted average pure premium
bm_summary["weighted_avg_pure_premium"] = (
    bm_summary["total_claim_amount"]
    / bm_summary["total_exposure"]
)

# Plot
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    bm_summary["BonusMalus"],
    bm_summary["weighted_avg_pure_premium"],
    marker="o",
    linewidth=1.5
)

ax.set(
    title="Exposure-weighted Average Pure Premium by BonusMalus",
    xlabel="BonusMalus",
    ylabel="Exposure-weighted Average Pure Premium"
)

ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Step 2: Define monotonicity constraints

feature_names = X_train_t.columns.tolist()

monotone_constraints = [
    1 if feature == "BonusMalus" else 0
    for feature in feature_names
]

# Check the feature order and constraints
print("Feature order and monotonicity constraints:")

print(
    dict(zip(feature_names, monotone_constraints))
)

# Check that the constraint vector has the correct length
assert len(monotone_constraints) == X_train_t.shape[1]

# Check that only BonusMalus has an increasing constraint
assert monotone_constraints.count(1) == 1
assert monotone_constraints[feature_names.index("BonusMalus")] == 1

# Create LightGBM with monotonicity constraint

constrained_lgbm = LGBMRegressor(
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
    verbosity=-1,

    # Monotonicity constraint
    monotone_constraints=monotone_constraints
)

# Step 3: Cross-validate the constrained LightGBM

constrained_lgbm_search = GridSearchCV(
    estimator=constrained_lgbm,
    param_grid=param_grid,
    scoring=negative_weighted_tweedie_deviance,
    cv=lgbm_search.cv,
    n_jobs=1,
    refit=True,
    verbose=1,
    return_train_score=True
)

# Fit the constrained model
constrained_lgbm_search.fit(
    X_train_t,
    y_train_t,
    sample_weight=w_train_t
)

# Display the best parameters
print("Best constrained LGBM parameters:")
print(constrained_lgbm_search.best_params_)

print("Best constrained CV deviance:")
print(-constrained_lgbm_search.best_score_)

# Get the best constrained estimator

best_constrained_lgbm = constrained_lgbm_search.best_estimator_

# Make predictions
df_train["pp_t_lgbm_constrained"] = (
    best_constrained_lgbm.predict(X_train_t)
)

df_test["pp_t_lgbm_constrained"] = (
    best_constrained_lgbm.predict(X_test_t)
)

# Save predictions in the original dataframe as well
df["pp_t_lgbm_constrained"] = np.nan

df.loc[df_train.index, "pp_t_lgbm_constrained"] = (
    df_train["pp_t_lgbm_constrained"]
)

df.loc[df_test.index, "pp_t_lgbm_constrained"] = (
    df_test["pp_t_lgbm_constrained"]
)

print(
    df[
        ["IDpol", "BonusMalus", "pp_t_lgbm_constrained"]
    ].head()
)


# Calculate train and test deviance

train_deviance_lgbm_constrained = weighted_deviance(
    y_train_t,
    df_train["pp_t_lgbm_constrained"],
    w_train_t
)

test_deviance_lgbm_constrained = weighted_deviance(
    y_test_t,
    df_test["pp_t_lgbm_constrained"],
    w_test_t
)

print(
    "Constrained LGBM training deviance:",
    train_deviance_lgbm_constrained
)

print(
    "Constrained LGBM testing deviance:",
    test_deviance_lgbm_constrained
)

# Compare unconstrained and constrained LightGBM

lgbm_comparison = pd.DataFrame(
    [
        {
            "Model": "Unconstrained LGBM",
            "CV Deviance": -lgbm_search.best_score_,
            "Train Deviance": train_deviance_lgbm,
            "Test Deviance": test_deviance_lgbm,
        },
        {
            "Model": "Constrained LGBM",
            "CV Deviance": -constrained_lgbm_search.best_score_,
            "Train Deviance": train_deviance_lgbm_constrained,
            "Test Deviance": test_deviance_lgbm_constrained,
        },
    ]
)

display(lgbm_comparison.round(5))

# %%
# ============================================================
# PS 4 - Exercise 2: Learning curve
# ============================================================

# Step 1: Refit the best constrained LightGBM
# with training and testing evaluation sets

best_constrained_lgbm = (
    constrained_lgbm_search.best_estimator_
)

best_constrained_lgbm.fit(
    X_train_t,
    y_train_t,

    # Exposure weights for training data
    sample_weight=w_train_t,

    # Evaluate on both training and testing datasets
    eval_set=[
        (X_train_t, y_train_t),
        (X_test_t, y_test_t)
    ],

    # Give names to the two evaluation datasets
    eval_names=["train", "test"],

    # Exposure weights for evaluation datasets
    eval_sample_weight=[
        w_train_t,
        w_test_t
    ],

    # Use Tweedie deviance as the evaluation metric
    eval_metric="tweedie"
)

# Check evaluation results

print("Evaluation datasets:")
print(best_constrained_lgbm.evals_result_.keys())

print("\nEvaluation metrics:")
print({
    dataset: list(metrics.keys())
    for dataset, metrics in best_constrained_lgbm.evals_result_.items()
})

# Step 2: Plot the learning curve

ax = lgb.plot_metric(
    best_constrained_lgbm,
    metric="tweedie",
    dataset_names=["train", "test"],
    title="Learning Curve - Constrained LightGBM",
    xlabel="Boosting Iterations",
    ylabel="Tweedie Deviance",
    figsize=(9, 5)
)

plt.tight_layout()
plt.show()

# Find the iteration with the lowest test deviance

eval_results = best_constrained_lgbm.evals_result_

train_curve = eval_results["train"]["tweedie"]
test_curve = eval_results["test"]["tweedie"]

best_iteration = np.argmin(test_curve) + 1

print("Best iteration:", best_iteration)
print("Minimum test deviance:", min(test_curve))
print("Train deviance at best iteration:", train_curve[best_iteration - 1])

# %%
# ============================================================
# PS 4 - Exercise 3: Metrics function
# ============================================================

# Step 1: Evaluate the unconstrained LGBM on the test set

unconstrained_metrics = evaluate_predictions(
    y_true=y_test_t,
    y_pred=df_test["pp_t_lgbm"],
    sample_weight=w_test_t
)

# Step 2: Evaluate the constrained LGBM on the test set

constrained_metrics = evaluate_predictions(
    y_true=y_test_t,
    y_pred=df_test["pp_t_lgbm_constrained"],
    sample_weight=w_test_t
)

# Step 3: Compare the metrics of both models

metrics_comparison = pd.concat(
    {
        "Unconstrained LGBM": unconstrained_metrics["Value"],
        "Constrained LGBM": constrained_metrics["Value"]
    },
    axis=1
)

display(metrics_comparison.round(5))

# %%
# ============================================================
# PS 4 - Exercise 4: Evaluation plots
# ============================================================

# Step 1: Define the prediction function

def dalex_predict(model, data):
    return model.predict(data)


# Create an explainer for the unconstrained LGBM

explainer_unconstrained = dx.Explainer(
    model=best_lgbm,
    data=X_test_t,
    y=y_test_t,
    predict_function=dalex_predict,
    label="Unconstrained LGBM",
    verbose=False
)


# Create an explainer for the constrained LGBM

explainer_constrained = dx.Explainer(
    model=best_constrained_lgbm,
    data=X_test_t,
    y=y_test_t,
    predict_function=dalex_predict,
    label="Constrained LGBM",
    verbose=False
)


# Identify numerical and categorical features

categorical_features = categoricals

numerical_features = [
    feature
    for feature in X_test_t.columns
    if feature not in categorical_features
]

print("Categorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)

# Step 2: Calculate PDPs for numerical features
# Unconstrained LGBM

pdp_unconstrained_num = explainer_unconstrained.model_profile(
    type="partial",
    variables=numerical_features,
    variable_type="numerical",
    N=300,
    grid_points=50,
    random_state=42,
    verbose=False
)

# Constrained LGBM

pdp_constrained_num = explainer_constrained.model_profile(
    type="partial",
    variables=numerical_features,
    variable_type="numerical",
    N=300,
    grid_points=50,
    random_state=42,
    verbose=False
)

# Step 3: Plot numerical PDPs
# Compare unconstrained and constrained LGBM

pdp_unconstrained_num.plot(
    objects=pdp_constrained_num,
    geom="aggregates",
    facet_ncol=3,
    title="PDPs - Numerical Features",
    y_title="Predicted Pure Premium"
)

# Step 4: Calculate PDPs for categorical features
# Unconstrained LGBM

pdp_unconstrained_cat = explainer_unconstrained.model_profile(
    type="partial",
    variables=categorical_features,
    variable_type="categorical",
    N=300,
    random_state=42,
    verbose=False
)

# Constrained LGBM

pdp_constrained_cat = explainer_constrained.model_profile(
    type="partial",
    variables=categorical_features,
    variable_type="categorical",
    N=300,
    random_state=42,
    verbose=False
)

# Step 5: Plot categorical PDPs
# Compare unconstrained and constrained LGBM

pdp_unconstrained_cat.plot(
    objects=pdp_constrained_cat,
    geom="bars",
    facet_ncol=3,
    title="PDPs - Categorical Features",
    y_title="Predicted Pure Premium"
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
