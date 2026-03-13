from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.ui_styles import apply_shared_styles

from utils.artifact_loader import (
    load_logistic_artifacts,
    load_logistic_reference_data,
)
from utils.feature_builder import (
    BusinessProfileInputs,
    baseline_new_business_profile,
    build_logistic_profile,
    category_display_to_column_map,
    complaint_display_to_column_map,
    prettify_feature_name,
)
from utils.location_utils import (
    NYC_LAT_MAX,
    NYC_LAT_MIN,
    NYC_LNG_MAX,
    NYC_LNG_MIN,
    clamp_to_nyc_bounds,
)
from utils.prediction_tools import (
    predict_logistic_profile,
    top_positive_negative,
)


LOGISTIC_LAT_COL = "business_latitude_first12m_first"
LOGISTIC_LNG_COL = "business_longitude_first12m_first"
LOGISTIC_LICENSE_COL = "active_license_count_first12m_mean"


def _format_metric_key(key: str) -> str:
    return key.replace("_", " ").title()


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_confusion_counts(
    metrics: dict[str, object],
) -> tuple[float, float, float, float] | None:
    confusion = metrics.get("confusion_matrix")

    if isinstance(confusion, dict):
        if all(k in confusion for k in ["tn", "fp", "fn", "tp"]):
            return (
                float(confusion["tn"]),
                float(confusion["fp"]),
                float(confusion["fn"]),
                float(confusion["tp"]),
            )

    if isinstance(confusion, list) and len(confusion) == 2:
        try:
            return (
                float(confusion[0][0]),
                float(confusion[0][1]),
                float(confusion[1][0]),
                float(confusion[1][1]),
            )
        except Exception:
            return None

    tn = metrics.get("tn")
    fp = metrics.get("fp")
    fn = metrics.get("fn")
    tp = metrics.get("tp")

    if all(v is not None for v in [tn, fp, fn, tp]):
        return (float(tn), float(fp), float(fn), float(tp))

    return None


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _build_class_metrics_df(metrics: dict[str, object]) -> pd.DataFrame:
    counts = _extract_confusion_counts(metrics)
    if counts is None:
        return pd.DataFrame()

    tn, fp, fn, tp = counts

    rows = [
        {
            "Class": "Did not survive 36 months (Class 0)",
            "Accuracy": _safe_divide(tn, tn + fn),
            "Recall": _safe_divide(tn, tn + fp),
        },
        {
            "Class": "Survived at least 36 months (Class 1)",
            "Accuracy": _safe_divide(tp, tp + fp),
            "Recall": _safe_divide(tp, tp + fn),
        },
    ]

    df = pd.DataFrame(rows)

    for col in ["Accuracy", "Recall"]:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")

    return df


def _extract_mean_predicted_probabilities(
    metrics: dict[str, object],
) -> tuple[float | None, float | None]:
    survivor_value = _safe_float(
        metrics.get("mean_predicted_probability_survivors")
    )
    nonsurvivor_value = _safe_float(
        metrics.get("mean_predicted_probability_nonsurvivors")
    )

    return survivor_value, nonsurvivor_value


def _prettify_coef_table(df: pd.DataFrame) -> pd.DataFrame:
    pretty_df = df.copy()
    if "feature" in pretty_df.columns:
        pretty_df["feature"] = pretty_df["feature"].map(prettify_feature_name)
    return pretty_df


def _feature_group(feature_name: str) -> str:
    if feature_name.startswith("business_category_"):
        return "category"
    if feature_name.startswith("complaint_type_"):
        return "complaint"
    return "other"


def _coef_row_style(feature_name: str) -> str:
    group = _feature_group(feature_name)

    if group == "category":
        return "background-color:#1d4ed8;color:white;"
    if group == "complaint":
        return "background-color:#ea580c;color:white;"
    return "background-color:#166534;color:white;"


def _style_coef_table(df: pd.DataFrame):
    styled_df = df.copy()

    if "feature" not in styled_df.columns:
        return styled_df

    raw_features = styled_df["feature"].copy()
    styled_df["feature"] = styled_df["feature"].map(prettify_feature_name)

    row_styles = raw_features.map(_coef_row_style)

    return styled_df.style.apply(
        lambda row: [row_styles.loc[row.name]] * len(row),
        axis=1,
    )


def _get_reference_median(
    df: pd.DataFrame,
    column_name: str,
    fallback: float,
) -> float:
    if column_name in df.columns and not df[column_name].dropna().empty:
        return float(df[column_name].median())
    return fallback


st.set_page_config(page_title="Logistic Regression", layout="wide")
apply_shared_styles()

st.title("Business Survival Predictor: Probability of 3-Year Survival")

artifacts = load_logistic_artifacts()
reference_data = load_logistic_reference_data()

pipeline = artifacts["pipeline"]
kept_columns = artifacts["kept_columns"]
coef_summary = artifacts["coef_summary"].copy()
metrics = artifacts["metrics"]
businesses_df = reference_data["businesses"]
x_train_df = reference_data["x_train"]
x_test_df = reference_data["x_test"]

positive_coef, negative_coef = top_positive_negative(
    coef_summary,
    coefficient_col="coefficient",
    top_n=20,
)

baseline_profile = baseline_new_business_profile(kept_columns, businesses_df)
baseline_prediction = predict_logistic_profile(pipeline, baseline_profile)

median_lat = _get_reference_median(businesses_df, LOGISTIC_LAT_COL, 40.7128)
median_lng = _get_reference_median(businesses_df, LOGISTIC_LNG_COL, -74.0060)

category_map = category_display_to_column_map(kept_columns)
display_categories = sorted(category_map.keys())

complaint_map = complaint_display_to_column_map(kept_columns)
display_complaints = sorted(complaint_map.keys())

if "logistic_selected_categories" not in st.session_state:
    st.session_state["logistic_selected_categories"] = []
if "logistic_selected_complaints" not in st.session_state:
    st.session_state["logistic_selected_complaints"] = []

st.markdown(
    """
    <style>
    .info-card {
        background: linear-gradient(
            135deg,
            rgba(127, 255, 212, 0.22) 0%,   /* aquamarine */
            rgba(255, 255, 120, 0.18) 45%,  /* soft yellow */
            rgba(11, 102, 35, 0.20) 100%    /* green (#0B6623 family) */
        );

        border: 1px solid rgba(255,255,255,0.28);
        border-radius: 18px;
        padding: 18px 18px 16px 18px;
        min-height: 220px;

        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);

        box-shadow: 0 12px 35px rgba(0,0,0,0.25);
    }

    .info-card h4 {
        margin: 0 0 10px 0;
        font-size: 1.02rem;
        font-weight: 500;
        color: white;
    }

    .info-card p {
        margin: 0;
        font-size: 0.95rem;
        line-height: 1.55;
        color: rgba(255,255,255,0.92);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

card1, card2, card3 = st.columns(3)

with card1:
    st.markdown(
        """
        <div class="info-card">
            <h4>What the Model Predicts</h4>
            <p>
                This logistic regression model estimates the probability that an NYC business
                will survive for at least 3 years <b>(36 months)</b> using features summarized from its
                first year <b>(12 observed months)</b>. Probabilities near <b>1.0</b> indicate higher predicted
                survival, while values near <b>0.0</b> indicate lower survival.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with card2:
    st.markdown(
        """
        <div class="info-card">
            <h4>How This Version Was Trained</h4>
            <p>
                To help the model learn patterns from both successful and struggling businesses,
                we <b>resampled the group with fewer observations</b> (businesses that closed early)
                so that it was represented more evenly alongside businesses that survived.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with card3:
    st.markdown(
        """
        <div class="info-card">
            <h4>What the Baseline Means</h4>
            <p>
                The <b>baseline</b> represents a simple reference business profile: 1
                average active license count, no selected categories, no complaint counts, and a
                median NYC location.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Survival Simulator", "Model Insights"])

with tab1:
    st.subheader("Hypothetical Business: Estimate 3-Year Survival")

    st.markdown(
        """
    Use the inputs below to create a first-year business profile:
    """
    )

    st.markdown("**Business categories in first 12 months**")
    selected_category_display = st.multiselect(
        "Business categories",
        options=display_categories,
        key="logistic_selected_categories",
        help="Multiple categories are allowed because a business may hold multiple active licenses.",
    )

    st.markdown("**Complaint types and counts in first 12 months**")
    selected_complaints = st.multiselect(
        "Complaint types",
        options=display_complaints,
        key="logistic_selected_complaints",
        help="Select any complaint types to include. Their count inputs will appear immediately below.",
    )

    complaint_counts: dict[str, float] = {}
    if selected_complaints:
        complaint_input_cols = st.columns(2)
        for idx, complaint_name in enumerate(selected_complaints):
            complaint_column = complaint_map[complaint_name]
            with complaint_input_cols[idx % 2]:
                st.markdown(f"**{complaint_name}**")
                count = st.number_input(
                    "Count",
                    min_value=0,
                    value=1,
                    step=1,
                    key=f"logistic_{complaint_column}",
                )
            complaint_counts[complaint_column] = float(count)

    with st.form("logistic_simulator_form"):
        active_license_count = st.number_input(
            "Average active license count in first 12 months",
            min_value=1,
            max_value=5,
            value=1,
            step=1,
        )

        col1, col2 = st.columns(2)
        business_latitude = col1.number_input(
            "Latitude",
            min_value=float(NYC_LAT_MIN),
            max_value=float(NYC_LAT_MAX),
            value=float(median_lat),
            format="%.6f",
            help=f"NYC latitude is roughly between {NYC_LAT_MIN} and {NYC_LAT_MAX}.",
        )
        business_longitude = col2.number_input(
            "Longitude",
            min_value=float(NYC_LNG_MIN),
            max_value=float(NYC_LNG_MAX),
            value=float(median_lng),
            format="%.6f",
            help=f"NYC longitude is roughly between {NYC_LNG_MIN} and {NYC_LNG_MAX}.",
        )

        submitted = st.form_submit_button("Predict 36-month survival probability")

    if submitted:
        business_latitude, business_longitude = clamp_to_nyc_bounds(
            float(business_latitude),
            float(business_longitude),
        )

        selected_category_columns = [
            category_map[name] for name in selected_category_display
        ]

        profile_inputs = BusinessProfileInputs(
            selected_category_columns=selected_category_columns,
            active_license_count=int(active_license_count),
            business_latitude=business_latitude,
            business_longitude=business_longitude,
            complaint_counts=complaint_counts,
        )

        profile = build_logistic_profile(
            kept_columns=kept_columns,
            reference_df=businesses_df,
            inputs=profile_inputs,
        )

        prediction = predict_logistic_profile(pipeline, profile)

        st.markdown("### Prediction Output")

        c1, c2, c3 = st.columns(3)
        prob = prediction["predicted_survival_probability"]
        baseline_prob = baseline_prediction["predicted_survival_probability"]

        display_prob = "> 0.9999" if prob > 0.9999 else f"{prob:.8f}"
        delta_value = prob - baseline_prob
        display_delta = "> 0.9999" if delta_value > 0.9999 else f"{delta_value:.8f}"

        c1.metric(
            "Predicted 36-month survival probability",
            display_prob,
            delta=display_delta,
        )
        c2.metric(
            "Predicted class",
            str(prediction["predicted_class"]),
        )
        c3.metric(
            "Baseline probability",
            f"{baseline_prediction['predicted_survival_probability']:.8f}",
        )

        st.markdown("### Baseline vs Hypothetical")
        results_df = pd.DataFrame(
            {
                "Profile": ["Baseline", "Hypothetical"],
                "Predicted 36-month survival probability": [
                    baseline_prediction["predicted_survival_probability"],
                    prediction["predicted_survival_probability"],
                ],
                "Predicted class": [
                    baseline_prediction["predicted_class"],
                    prediction["predicted_class"],
                ],
            }
        )
        st.dataframe(results_df, use_container_width=True)

        st.markdown("### Entered / Derived Feature Values Used")
        non_zero_features = profile.loc[0]
        non_zero_features = non_zero_features[non_zero_features != 0].sort_values(
            ascending=False
        )

        display_features = (
            non_zero_features.rename("value")
            .reset_index()
            .rename(columns={"index": "feature"})
        )
        display_features["feature"] = display_features["feature"].map(prettify_feature_name)

        st.dataframe(display_features, use_container_width=True)

with tab2:
    st.subheader("Top Factors Influencing Survival")
    st.markdown(
        """
    Each value represents the **marginal change in log-odds of surviving 3 years**
    associated with a one-unit increase in that feature, holding all other
    features constant.

    - **Positive coefficients** increase the predicted probability of survival.
    - **Negative coefficients** decrease the predicted probability of survival.

    The tables below show the predictors with the largest positive and negative
    effects on predicted survival.
    """
    )

    st.markdown(
        """
    **Predictor color legend**

    <span style="background:#1d4ed8;color:white;padding:6px 10px;border-radius:6px;margin-right:10px;">
    Business category predictor
    </span>

    <span style="background:#ea580c;color:white;padding:6px 10px;border-radius:6px;margin-right:10px;">
    Complaint type predictor
    </span>

    <span style="background:#166534;color:white;padding:6px 10px;border-radius:6px;">
    Other predictors (location, licenses)
    </span>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top positive coefficients**")
        pos_cols = [
            col
            for col in ["feature", "coefficient"]
            if col in positive_coef.columns
        ]
        st.dataframe(
            _style_coef_table(positive_coef[pos_cols]),
            use_container_width=True,
        )
    with col2:
        st.markdown("**Top negative coefficients**")
        neg_cols = [
            col
            for col in ["feature", "coefficient"]
            if col in negative_coef.columns
        ]
        st.dataframe(
            _style_coef_table(negative_coef[neg_cols]),
            use_container_width=True,
        )

    st.subheader("Evaluation Summary")

    class_metrics_df = _build_class_metrics_df(metrics)
    survivor_mean_prob, nonsurvivor_mean_prob = _extract_mean_predicted_probabilities(
        metrics
    )

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("**Class-specific performance**")
        if not class_metrics_df.empty:
            st.dataframe(class_metrics_df, use_container_width=True)
        else:
            st.info(
                "Class-specific metrics are unavailable because confusion-matrix counts were not found in the saved metrics."
            )

        st.markdown("**Dataset sizes**")
        size_df = pd.DataFrame(
            {
                "Split": ["Business reference rows", "X_train rows", "X_test rows"],
                "Rows": [businesses_df.shape[0], x_train_df.shape[0], x_test_df.shape[0]],
                "Columns": [businesses_df.shape[1], x_train_df.shape[1], x_test_df.shape[1]],
            }
        )
        st.dataframe(size_df, use_container_width=True)

    with right_col:
        st.markdown("**Mean predicted probabilities by true outcome**")

        prob_summary_df = pd.DataFrame(
            {
                "Group": [
                    "Actually survived at least 36 months",
                    "Actually did not survive 36 months",
                ],
                "Mean predicted survival probability": [
                    f"{survivor_mean_prob:.4f}" if survivor_mean_prob is not None else "N/A",
                    f"{nonsurvivor_mean_prob:.4f}" if nonsurvivor_mean_prob is not None else "N/A",
                ],
            }
        )
        st.dataframe(prob_summary_df, use_container_width=True)

        st.markdown(
            f"""
        Businesses that **actually survived at least 36 months** have an average predicted survival probability of **{survivor_mean_prob:.4f}**.
        Businesses that **actually closed before 36 months** have an average predicted survival probability of **{nonsurvivor_mean_prob:.4f}**.

        This gap shows that the model assigns **higher probabilities to businesses that truly survived** and **lower probabilities to those that closed earlier**.  
        The difference between these averages indicates that the model is able to meaningfully separate survivors from non-survivors in the test data.
        """
        )