from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.ui_styles import apply_shared_styles

from utils.artifact_loader import (
    load_logistic_reference_data,
    load_standard_cox_artifacts,
    load_time_varying_cox_artifacts,
)
from utils.cox_feature_builder import (
    CoxProfileInputs,
    baseline_standard_cox_profile,
    build_standard_cox_profile,
    build_time_varying_cox_profiles_over_time,
    cox_category_display_to_column_map,
    cox_complaint_display_to_column_map,
    generate_time_varying_example_timelines,
    get_reference_median_lat_lng,
    prettify_cox_feature_name,
    summarize_generated_time_varying_timelines,
)
from utils.location_utils import (
    DEFAULT_SURVIVAL_MONTHS,
    NYC_LAT_MAX,
    NYC_LAT_MIN,
    NYC_LNG_MAX,
    NYC_LNG_MIN,
    clamp_to_nyc_bounds,
)
from utils.prediction_tools import (
    predict_standard_cox_profile,
    predict_time_varying_cox_profiles,
    top_positive_negative,
)


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


def _style_cox_summary_table(df: pd.DataFrame):
    styled_df = df.copy()

    if "feature" not in styled_df.columns:
        return styled_df

    raw_features = styled_df["feature"].copy()
    styled_df["feature"] = styled_df["feature"].map(prettify_cox_feature_name)

    row_styles = raw_features.map(_coef_row_style)

    return styled_df.style.apply(
        lambda row: [row_styles.loc[row.name]] * len(row),
        axis=1,
    )


st.set_page_config(page_title="Cox Models", layout="wide")
apply_shared_styles()

st.markdown(
    """
    <style>
    .info-card {
        background: linear-gradient(
            135deg,
            rgba(127, 255, 212, 0.22) 0%,
            rgba(255, 255, 120, 0.18) 45%,
            rgba(11, 102, 35, 0.20) 100%
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

    .glass-banner {
        margin-top: 8px;
        margin-bottom: 18px;
        padding: 14px 18px;
        border-radius: 14px;
        background: linear-gradient(
            135deg,
            rgba(127,255,212,0.30) 0%,
            rgba(255,255,140,0.25) 50%,
            rgba(11,102,35,0.30) 100%
        );
        border: 1px solid rgba(255,255,255,0.35);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        color: white;
        font-size: 15px;
        font-weight: 500;
        box-shadow: 0 10px 25px rgba(0,0,0,0.18);
    }

    .limitation-banner {
        margin-top: 4px;
        margin-bottom: 18px;
        padding: 14px 18px;
        border-radius: 14px;
        background: linear-gradient(
            135deg,
            rgba(255,80,40,0.28) 0%,
            rgba(255,150,40,0.25) 45%,
            rgba(255,210,80,0.28) 100%
        );
        border: 1px solid rgba(255,200,120,0.35);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        color: white;
        font-size: 14.5px;
        line-height: 1.5;
        box-shadow: 0 10px 25px rgba(0,0,0,0.18);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Cox Proportional Hazards Survival Analysis")

standard_artifacts = load_standard_cox_artifacts()
time_varying_artifacts = load_time_varying_cox_artifacts()
reference_data = load_logistic_reference_data()
reference_df = reference_data["businesses"]

standard_summary = standard_artifacts["summary"].copy()
tv_summary = time_varying_artifacts["summary"].copy()

standard_positive, standard_negative = top_positive_negative(
    standard_summary,
    coefficient_col="coef",
    top_n=10,
)
tv_positive, tv_negative = top_positive_negative(
    tv_summary,
    coefficient_col="coef",
    top_n=10,
)

standard_tab, tv_tab = st.tabs(["Standard Cox Model", "Time-Varying Cox Model"])

with standard_tab:
    card1, card2, card3 = st.columns(3)

    with card1:
        st.markdown(
            """
            <div class="info-card">
                <h4>What This Model Predicts</h4>
                <p>
                    This standard Cox proportional hazards model compares hypothetical businesses
                    using their baseline feature values, meaning the business characteristics observed
                    at the beginning of the timeline (the first month). The simulator below shows
                    predicted survival at 1, 3, 5, and 10 years.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with card2:
        st.markdown(
            """
            <div class="info-card">
                <h4>How the Standard Cox Dataset Looks</h4>
                <p>
                    The standard Cox model uses <b>one row per business</b>. Each row summarizes a business
                    using baseline features such as category indicators, complaint history, active license count,
                    and location, then relates those features to how long the business stayed open.
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
                    The <b>baseline</b> is a neutral reference business: one active license,
                    no complaint history, no selected categories, and a median NYC location
                    assigned to its nearest learned location cluster.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="limitation-banner">
        ⚠ <b>Model limitations:</b> This standard Cox model uses only baseline (initial) business characteristics
        and assumes those effects remain constant over time. It does not capture changes in complaints,
        licenses, or neighborhood conditions as a business evolves. Predictions reflect patterns learned
        from historical data and represent relative risk estimates rather than exact forecasts of closure.
        </div>
        """,
        unsafe_allow_html=True,
    )

    baseline_profile = baseline_standard_cox_profile(
        standard_artifacts["kept_columns"],
        reference_df,
    )
    baseline_prediction = predict_standard_cox_profile(
        model=standard_artifacts["model"],
        scaler=standard_artifacts["scaler"],
        kept_columns=standard_artifacts["kept_columns"],
        profile_df=baseline_profile,
        survival_times=DEFAULT_SURVIVAL_MONTHS,
    )

    median_lat, median_lng = get_reference_median_lat_lng(reference_df)

    standard_category_map = cox_category_display_to_column_map(
        standard_artifacts["kept_columns"]
    )
    complaint_map = cox_complaint_display_to_column_map(
        standard_artifacts["kept_columns"]
    )

    st.subheader("Hypothetical Business Simulator")

    num_businesses = st.number_input(
        "Number of hypothetical businesses to compare",
        min_value=1,
        max_value=5,
        value=1,
        step=1,
        key="standard_num_businesses",
    )

    business_specs: list[dict[str, object]] = []

    for idx in range(int(num_businesses)):
        category_key = f"standard_cats_{idx}"
        complaint_key = f"standard_complaints_{idx}"

        if category_key not in st.session_state:
            st.session_state[category_key] = []
        if complaint_key not in st.session_state:
            st.session_state[complaint_key] = []

        with st.expander(f"Hypothetical Business {idx + 1}", expanded=(idx == 0)):
            st.markdown("**Business categories**")
            selected_category_names = st.multiselect(
                f"Business categories for Business {idx + 1}",
                options=sorted(standard_category_map.keys()),
                key=category_key,
                help="A business may hold multiple active licenses and therefore multiple categories.",
            )

            st.markdown("**Complaint types and counts**")
            selected_complaint_names = st.multiselect(
                f"Complaint types for Business {idx + 1}",
                options=sorted(complaint_map.keys()),
                key=complaint_key,
                help="Select any complaint types to include. Count inputs will appear immediately below.",
            )

            complaint_counts: dict[str, float] = {}
            complaint_input_cols = st.columns(2)
            for complaint_idx, complaint_name in enumerate(selected_complaint_names):
                complaint_column = complaint_map[complaint_name]
                with complaint_input_cols[complaint_idx % 2]:
                    st.markdown(f"**{complaint_name}**")
                    count = st.number_input(
                        f"Count for {complaint_name} (Business {idx + 1})",
                        min_value=0,
                        value=1,
                        step=1,
                        key=f"standard_complaint_count_{idx}_{complaint_column}",
                    )
                complaint_counts[complaint_column] = float(count)

            active_license_count = st.number_input(
                f"Active license count for Business {idx + 1}",
                min_value=1,
                max_value=5,
                value=1,
                step=1,
                key=f"standard_license_{idx}",
            )

            col1, col2 = st.columns(2)
            latitude = col1.number_input(
                f"Latitude for Business {idx + 1}",
                min_value=float(NYC_LAT_MIN),
                max_value=float(NYC_LAT_MAX),
                value=float(median_lat),
                format="%.6f",
                key=f"standard_lat_{idx}",
                help=f"NYC latitude is roughly between {NYC_LAT_MIN} and {NYC_LAT_MAX}.",
            )
            longitude = col2.number_input(
                f"Longitude for Business {idx + 1}",
                min_value=float(NYC_LNG_MIN),
                max_value=float(NYC_LNG_MAX),
                value=float(median_lng),
                format="%.6f",
                key=f"standard_lng_{idx}",
                help=f"NYC longitude is roughly between {NYC_LNG_MIN} and {NYC_LNG_MAX}.",
            )

            business_specs.append(
                {
                    "label": f"Business {idx + 1}",
                    "category_columns": [
                        standard_category_map[name] for name in selected_category_names
                    ],
                    "active_license_count": int(active_license_count),
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                    "complaint_counts": complaint_counts,
                }
            )

    submitted = st.button("Compare survival trajectories", use_container_width=True)

    if submitted:
        chart_rows: list[dict[str, object]] = []
        summary_rows: list[dict[str, object]] = []

        baseline_line = {
            "1 year": baseline_prediction["survival_prob_12m"],
            "3 years": baseline_prediction["survival_prob_36m"],
            "5 years": baseline_prediction["survival_prob_60m"],
            "10 years": baseline_prediction["survival_prob_120m"],
        }

        chart_rows.extend(
            [
                {
                    "Horizon": horizon,
                    "Business": "Baseline",
                    "Survival Probability": value,
                }
                for horizon, value in baseline_line.items()
            ]
        )

        summary_rows.append(
            {
                "Business": "Baseline",
                "Partial hazard": baseline_prediction["partial_hazard"],
                "Risk vs baseline": 1.0,
                "1-year survival": baseline_prediction["survival_prob_12m"],
                "3-year survival": baseline_prediction["survival_prob_36m"],
                "5-year survival": baseline_prediction["survival_prob_60m"],
                "10-year survival": baseline_prediction["survival_prob_120m"],
            }
        )

        for spec in business_specs:
            lat, lng = clamp_to_nyc_bounds(
                float(spec["latitude"]),
                float(spec["longitude"]),
            )

            inputs = CoxProfileInputs(
                selected_category_columns=list(spec["category_columns"]),
                active_license_count=int(spec["active_license_count"]),
                business_latitude=lat,
                business_longitude=lng,
                complaint_counts=dict(spec["complaint_counts"]),
            )

            profile = build_standard_cox_profile(
                kept_columns=standard_artifacts["kept_columns"],
                reference_df=reference_df,
                inputs=inputs,
            )

            prediction = predict_standard_cox_profile(
                model=standard_artifacts["model"],
                scaler=standard_artifacts["scaler"],
                kept_columns=standard_artifacts["kept_columns"],
                profile_df=profile,
                survival_times=DEFAULT_SURVIVAL_MONTHS,
            )

            line = {
                "1 year": prediction["survival_prob_12m"],
                "3 years": prediction["survival_prob_36m"],
                "5 years": prediction["survival_prob_60m"],
                "10 years": prediction["survival_prob_120m"],
            }

            chart_rows.extend(
                [
                    {
                        "Horizon": horizon,
                        "Business": str(spec["label"]),
                        "Survival Probability": value,
                    }
                    for horizon, value in line.items()
                ]
            )

            summary_rows.append(
                {
                    "Business": str(spec["label"]),
                    "Partial hazard": prediction["partial_hazard"],
                    "Risk vs baseline": prediction["partial_hazard"]
                    / baseline_prediction["partial_hazard"],
                    "1-year survival": prediction["survival_prob_12m"],
                    "3-year survival": prediction["survival_prob_36m"],
                    "5-year survival": prediction["survival_prob_60m"],
                    "10-year survival": prediction["survival_prob_120m"],
                }
            )

        chart_df = pd.DataFrame(chart_rows)
        horizon_order = ["1 year", "3 years", "5 years", "10 years"]
        chart_df["Horizon"] = pd.Categorical(
            chart_df["Horizon"],
            categories=horizon_order,
            ordered=True,
        )
        chart_df = chart_df.sort_values(["Horizon", "Business"])

        st.markdown("#### Survival Probability Trajectory Comparison")
        st.line_chart(
            chart_df.pivot(
                index="Horizon",
                columns="Business",
                values="Survival Probability",
            ),
            use_container_width=True,
        )

        st.markdown(
            """
            In the table below, **partial hazard** is the model's relative risk score, and
            **risk vs baseline** compares each hypothetical business against the neutral baseline business.
            Values above 1 indicate higher relative closure risk than baseline, while values below 1
            indicate lower relative closure risk than baseline.
            """
        )

        st.markdown("#### Risk Summary")
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(
            summary_df.style.format(
                {
                    "Partial hazard": "{:.4f}",
                    "Risk vs baseline": "{:.3f}",
                    "1-year survival": "{:.3f}",
                    "3-year survival": "{:.3f}",
                    "5-year survival": "{:.3f}",
                    "10-year survival": "{:.3f}",
                }
            ),
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("#### Features Most Associated with Higher vs Lower Hazard")
    st.markdown(
        """
        Hazard represents a business’s relative closure risk based on its baseline characteristics, compared to the baseline business in the dataset.

        - **Positive coefficients** are associated with **higher hazard**, meaning higher relative closure risk.
        - **Negative coefficients** are associated with **lower hazard**, meaning lower relative closure risk.
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
        st.markdown("**Top features increasing hazard**")
        display_cols = [
            c for c in ["feature", "coef", "exp(coef)"] if c in standard_positive.columns
        ]
        st.dataframe(
            _style_cox_summary_table(standard_positive[display_cols]),
            use_container_width=True,
        )
    with col2:
        st.markdown("**Top features decreasing hazard**")
        display_cols = [
            c for c in ["feature", "coef", "exp(coef)"] if c in standard_negative.columns
        ]
        st.dataframe(
            _style_cox_summary_table(standard_negative[display_cols]),
            use_container_width=True,
        )

with tv_tab:
    card1, card2, card3 = st.columns(3)

    with card1:
        st.markdown(
            """
            <div class="info-card">
                <h4>What This Model Predicts</h4>
                <p>
                    The time-varying Cox model estimates a business's current relative closure risk at each time point using features that can change over time, such as complaints, active license count, categories, and location. Unlike the standard Cox model, it does not simulate future survival probability trajectories.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with card2:
        st.markdown(
            """
            <div class="info-card">
                <h4>How the Time-Varying Dataset Looks</h4>
                <p>
                    The time-varying Cox model uses <b>multiple rows per business</b>, where each row
                    represents the business at a different time point. This allows the model to learn
                    how evolving business conditions relate to changing closure risk over time.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with card3:
        st.markdown(
            """
            <div class="info-card">
                <h4>What the Generated Timelines Mean</h4>
                <p>
                    The example timelines below are synthetic but realistic business histories.
                    They show how a business's state can evolve across time points while the model
                    updates its relative closure risk using the changing inputs.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    st.subheader("Generated Example Business Timelines")

    tv_num_businesses = st.number_input(
        "Number of hypothetical businesses",
        min_value=1,
        max_value=4,
        value=2,
        step=1,
        key="tv_num_businesses",
    )
    tv_num_timepoints = st.number_input(
        "Number of time points per business",
        min_value=2,
        max_value=6,
        value=4,
        step=1,
        key="tv_num_timepoints",
    )

    if "tv_generated_specs" not in st.session_state:
        st.session_state["tv_generated_specs"] = []

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Generate example timelines", use_container_width=True):
            st.session_state["tv_generated_specs"] = generate_time_varying_example_timelines(
                kept_columns=time_varying_artifacts["kept_columns"],
                reference_df=reference_df,
                num_businesses=int(tv_num_businesses),
                num_timepoints=int(tv_num_timepoints),
                random_state=42,
            )
    with col_b:
        if st.button("Regenerate with new random seed", use_container_width=True):
            seed = int(pd.Timestamp.now().timestamp()) % 1_000_000
            st.session_state["tv_generated_specs"] = generate_time_varying_example_timelines(
                kept_columns=time_varying_artifacts["kept_columns"],
                reference_df=reference_df,
                num_businesses=int(tv_num_businesses),
                num_timepoints=int(tv_num_timepoints),
                random_state=seed,
            )

    generated_specs = list(st.session_state.get("tv_generated_specs", []))

    if generated_specs:
        generated_summary = summarize_generated_time_varying_timelines(generated_specs)

        st.markdown("### Generated Business States")
        st.dataframe(
            generated_summary.style.format(
                {
                    "Latitude": "{:.5f}",
                    "Longitude": "{:.5f}",
                }
            ),
            use_container_width=True,
        )

        chart_rows: list[dict[str, object]] = []
        summary_rows: list[dict[str, object]] = []

        for business_spec in generated_specs:
            normalized_timepoints: list[dict[str, object]] = []

            for tp in list(business_spec["timepoints"]):
                lat, lng = clamp_to_nyc_bounds(
                    float(tp["business_latitude"]),
                    float(tp["business_longitude"]),
                )

                normalized_timepoints.append(
                    {
                        "month": int(tp["month"]),
                        "selected_category_columns": list(tp["selected_category_columns"]),
                        "active_license_count": int(tp["active_license_count"]),
                        "business_latitude": lat,
                        "business_longitude": lng,
                        "complaint_counts": dict(tp["complaint_counts"]),
                    }
                )

            profiles_df = build_time_varying_cox_profiles_over_time(
                kept_columns=time_varying_artifacts["kept_columns"],
                reference_df=reference_df,
                timepoint_specs=normalized_timepoints,
            )

            predictions_df = predict_time_varying_cox_profiles(
                model=time_varying_artifacts["model"],
                scaler=time_varying_artifacts["scaler"],
                kept_columns=time_varying_artifacts["kept_columns"],
                profiles_df=profiles_df,
            )

            predictions_df["Business"] = str(business_spec["label"])
            predictions_df["Risk score"] = predictions_df["partial_hazard"]

            for _, row in predictions_df.iterrows():
                chart_rows.append(
                    {
                        "Month": int(row["month"]),
                        "Business": str(row["Business"]),
                        "Risk score": float(row["Risk score"]),
                    }
                )

                summary_rows.append(
                    {
                        "Business": str(row["Business"]),
                        "Month": int(row["month"]),
                        "Partial hazard": float(row["partial_hazard"]),
                        "Risk score": float(row["Risk score"]),
                    }
                )

        if chart_rows:
            chart_df = pd.DataFrame(chart_rows).sort_values(["Month", "Business"])

            st.markdown("### Relative Closure Risk Across Time Points")
            st.markdown(
                """
                This chart shows the model's **partial hazard / risk score** at each time point.
                Higher values indicate higher current relative closure risk under the business state
                shown for that month.
                """
            )
            st.line_chart(
                chart_df.pivot(
                    index="Month",
                    columns="Business",
                    values="Risk score",
                ),
                use_container_width=True,
            )

            st.markdown("### Time-Point Risk Summary")

            summary_df = (
                pd.DataFrame(summary_rows)
                .drop(columns=["Partial hazard"])
                .sort_values(["Business", "Month"])
            )

            st.dataframe(
                summary_df.style.format(
                    {
                        "Risk score": "{:.4f}",
                    }
                ),
                use_container_width=True,
            )
    else:
        st.markdown(
            """
            <div class="glass-banner">
            Click <b>Generate example timelines</b> to create realistic evolving business states
            for the time-varying Cox model. These timelines are <b>randomly generated</b> from the
            structure of the training data (categories, complaints, licenses, and locations).
            The generator uses a <b>random seed</b> so results can be reproduced, while
            <b>Regenerate with new random seed</b> creates a different set of example timelines.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### Features Most Associated with Higher vs Lower Hazard")
    st.markdown(
        """
        Hazard is the instantaneous risk that a business will close at a given time, assuming it has remained open up to that point.
        
        - **Positive coefficients** are associated with **higher instantaneous hazard**,
          meaning higher current relative closure risk.
        - **Negative coefficients** are associated with **lower instantaneous hazard**,
          meaning lower current relative closure risk.
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

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top features increasing hazard**")
        display_cols = [
            c for c in ["feature", "coef", "exp(coef)"] if c in tv_positive.columns
        ]
        st.dataframe(
            _style_cox_summary_table(tv_positive[display_cols]),
            use_container_width=True,
        )
    with col2:
        st.markdown("**Top features decreasing hazard**")
        display_cols = [
            c for c in ["feature", "coef", "exp(coef)"] if c in tv_negative.columns
        ]
        st.dataframe(
            _style_cox_summary_table(tv_negative[display_cols]),
            use_container_width=True,
        )
