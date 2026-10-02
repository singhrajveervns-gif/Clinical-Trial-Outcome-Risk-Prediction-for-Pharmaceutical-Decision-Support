"""
Clinical Trial Outcome Risk Prediction — Streamlit App

Loads the fitted model produced in Phase 8.5 of the notebook
(models/final_model.pkl) and scores a single, user-described trial:
predicted operational completion probability, a risk tier, and a
local SHAP explanation of the underlying tree-model prediction.

Run locally:
    streamlit run app.py

The app expects models/final_model.pkl to sit in the same folder
this script is launched from (see the folder layout in the README).
"""

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt


# --------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------

st.set_page_config(
    page_title="Clinical Trial Outcome Risk Predictor",
    page_icon="🧪",
    layout="wide",
)

MODEL_PATH = "models/final_model.pkl"


# --------------------------------------------------------------------
# Feature definitions
# --------------------------------------------------------------------

NUMERIC_FEATURES = [
    "num_conditions",
    "num_interventions",
    "brief_title_word_count",
    "full_title_word_count",
    "intervention_description_word_count",
    "start_year",
    "trial_complexity_index",
]

BINARY_FEATURES = [
    "is_multi_condition",
    "is_multi_intervention",
    "includes_child",
    "includes_adult",
    "includes_older_adult",
]

CATEGORICAL_FEATURES = [
    "sponsor_type_grouped",
    "Responsible Party",
    "Primary Purpose",
    "Study Type",
    "Phases",
]

MODEL_FEATURES = (
    NUMERIC_FEATURES
    + BINARY_FEATURES
    + CATEGORICAL_FEATURES
)


# --------------------------------------------------------------------
# Sponsor grouping
# --------------------------------------------------------------------

SPONSOR_GROUP_MAP = {
    "INDUSTRY": "INDUSTRY",
    "NIH": "GOVERNMENT",
    "FED": "GOVERNMENT",
    "US_FED": "GOVERNMENT",
    "OTHER_GOV": "GOVERNMENT",
    "NETWORK": "ACADEMIC_OR_NETWORK",
    "OTHER": "ACADEMIC_OR_OTHER",
    "INDIV": "ACADEMIC_OR_OTHER",
}


# --------------------------------------------------------------------
# Dropdown options
# --------------------------------------------------------------------

PHASE_OPTIONS = [
    "NOT_APPLICABLE",
    "EARLY_PHASE1",
    "PHASE1",
    "PHASE1, PHASE2",
    "PHASE2",
    "PHASE2, PHASE3",
    "PHASE3",
    "PHASE4",
    "Unknown",
]

PURPOSE_OPTIONS = [
    "TREATMENT",
    "PREVENTION",
    "DIAGNOSTIC",
    "SUPPORTIVE_CARE",
    "SCREENING",
    "HEALTH_SERVICES_RESEARCH",
    "BASIC_SCIENCE",
    "DEVICE_FEASIBILITY",
    "ECT",
    "OTHER",
    "Unknown",
]

RESPONSIBLE_PARTY_OPTIONS = [
    "SPONSOR",
    "PRINCIPAL_INVESTIGATOR",
    "Unknown",
]

ORG_CLASS_OPTIONS = (
    list(SPONSOR_GROUP_MAP.keys())
    + ["UNKNOWN"]
)


# --------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# --------------------------------------------------------------------
# Risk tier
# --------------------------------------------------------------------

def risk_tier(p_success: float) -> str:
    if p_success < 0.40:
        return "High Risk (<40%)"

    elif p_success < 0.70:
        return "Medium Risk (40-70%)"

    return "Low Risk (>70%)"


# --------------------------------------------------------------------
# Build model input
# --------------------------------------------------------------------

def build_input_row(inputs: dict) -> pd.DataFrame:

    num_conditions = inputs["num_conditions"]
    num_interventions = inputs["num_interventions"]

    row = {
        "num_conditions": num_conditions,

        "num_interventions": num_interventions,

        "brief_title_word_count": (
            len(inputs["brief_title"].split())
        ),

        "full_title_word_count": (
            len(inputs["full_title"].split())
        ),

        "intervention_description_word_count": (
            len(inputs["intervention_description"].split())
        ),

        "start_year": inputs["start_year"],

        "trial_complexity_index": (
            num_conditions + num_interventions
        ),

        "is_multi_condition": int(
            num_conditions > 1
        ),

        "is_multi_intervention": int(
            num_interventions > 1
        ),

        "includes_child": int(
            inputs["includes_child"]
        ),

        "includes_adult": int(
            inputs["includes_adult"]
        ),

        "includes_older_adult": int(
            inputs["includes_older_adult"]
        ),

        "sponsor_type_grouped": (
            SPONSOR_GROUP_MAP.get(
                inputs["org_class"],
                "OTHER"
            )
        ),

        "Responsible Party": (
            inputs["responsible_party"]
        ),

        "Primary Purpose": (
            inputs["primary_purpose"]
        ),

        "Study Type": (
            inputs["study_type"]
        ),

        "Phases": (
            inputs["phase"]
        ),
    }

    return pd.DataFrame(
        [row],
        columns=MODEL_FEATURES
    )


# --------------------------------------------------------------------
# SHAP explanation
# --------------------------------------------------------------------

def explain_prediction(
    model,
    input_row: pd.DataFrame
):
    """
    Generate a local SHAP explanation.

    The saved model is a CalibratedClassifierCV. For SHAP, we
    extract the underlying fitted pipeline(s), transform the
    input with the same preprocessor used during training, and
    explain the underlying tree classifier.

    The displayed probability remains the calibrated probability
    returned by model.predict_proba().
    """

    try:

        import shap

        # ------------------------------------------------------------
        # Find fitted base estimator(s)
        # ------------------------------------------------------------

        base_models = []

        # Case 1:
        # Normal sklearn Pipeline
        if hasattr(model, "named_steps"):

            base_models.append(model)

        # Case 2:
        # CalibratedClassifierCV
        elif hasattr(
            model,
            "calibrated_classifiers_"
        ):

            for calibrated_classifier in (
                model.calibrated_classifiers_
            ):

                estimator = getattr(
                    calibrated_classifier,
                    "estimator",
                    None
                )

                # Compatibility with older sklearn
                if estimator is None:

                    estimator = getattr(
                        calibrated_classifier,
                        "base_estimator",
                        None
                    )

                if estimator is not None:

                    base_models.append(
                        estimator
                    )

        # Case 3:
        # Direct estimator
        else:

            base_models.append(model)

        if not base_models:

            raise ValueError(
                "Could not find fitted base estimator(s) "
                "inside the saved model."
            )

        # ------------------------------------------------------------
        # Calculate SHAP values
        # ------------------------------------------------------------

        all_shap_values = []

        feature_names = None

        for base_model in base_models:

            # --------------------------------------------------------
            # Extract preprocessing + classifier
            # --------------------------------------------------------

            if hasattr(
                base_model,
                "named_steps"
            ):

                if "preprocessor" not in (
                    base_model.named_steps
                ):

                    raise ValueError(
                        "The fitted pipeline does not contain "
                        "a 'preprocessor' step."
                    )

                preprocessor = (
                    base_model.named_steps[
                        "preprocessor"
                    ]
                )

                if "classifier" not in (
                    base_model.named_steps
                ):

                    raise ValueError(
                        "The fitted pipeline does not contain "
                        "a 'classifier' step."
                    )

                estimator = (
                    base_model.named_steps[
                        "classifier"
                    ]
                )

                # Transform exactly as during model training
                transformed = (
                    preprocessor.transform(
                        input_row
                    )
                )

                # Convert sparse matrix to dense
                if hasattr(
                    transformed,
                    "toarray"
                ):

                    transformed = (
                        transformed.toarray()
                    )

                feature_names = (
                    preprocessor
                    .get_feature_names_out()
                )

            else:

                # Direct tree estimator
                estimator = base_model

                transformed = (
                    input_row.to_numpy()
                )

                feature_names = np.asarray(
                    input_row.columns
                )

            # --------------------------------------------------------
            # SHAP TreeExplainer
            # --------------------------------------------------------

            explainer = shap.TreeExplainer(
                estimator
            )

            raw_shap = explainer.shap_values(
                transformed,
                check_additivity=False
            )

            # --------------------------------------------------------
            # Handle SHAP output formats
            # --------------------------------------------------------

            if isinstance(
                raw_shap,
                list
            ):

                # Older SHAP:
                # [class_0_values, class_1_values]
                shap_row = raw_shap[1][0]

            elif getattr(
                raw_shap,
                "ndim",
                0
            ) == 3:

                # Newer SHAP:
                # samples x features x classes
                shap_row = raw_shap[
                    0,
                    :,
                    1
                ]

            else:

                # Binary output:
                # samples x features
                shap_row = raw_shap[0]

            all_shap_values.append(
                np.asarray(
                    shap_row,
                    dtype=float
                )
            )

        # ------------------------------------------------------------
        # Average across calibrated estimators
        # ------------------------------------------------------------

        shap_values = np.mean(
            np.vstack(
                all_shap_values
            ),
            axis=0
        )

        return (
            pd.Series(
                shap_values,
                index=feature_names
            )
            .sort_values(
                key=np.abs,
                ascending=False
            )
        )

    except Exception as e:

        st.error(
            "SHAP explanation error: "
            f"{type(e).__name__}: {e}"
        )

        return None


# --------------------------------------------------------------------
# Page title
# --------------------------------------------------------------------

st.title(
    "🧪 Clinical Trial Outcome Risk Predictor"
)

st.caption(
    "Portfolio demo — predicts an **operational completion** "
    "probability (Completed vs. Terminated/Withdrawn/Suspended) "
    "from design-time trial characteristics only. "
    "Not a clinical, efficacy, or investment recommendation."
)


# --------------------------------------------------------------------
# Sidebar — trial inputs
# --------------------------------------------------------------------

with st.sidebar:

    st.header(
        "Describe the trial"
    )

    brief_title = st.text_input(
        "Brief title",
        "A Study of Drug X in Adult Patients"
    )

    full_title = st.text_input(
        "Full title",
        "A Randomized, Double-Blind Study of Drug X "
        "in Adult Patients With Condition Y"
    )

    intervention_description = st.text_area(
        "Intervention description",
        "Participants receive Drug X or placebo "
        "once daily for 12 weeks."
    )

    # --------------------------------------------------------------
    # Design complexity
    # --------------------------------------------------------------

    st.subheader(
        "Design complexity"
    )

    conditions_text = st.text_input(
        "Conditions studied (comma-separated)",
        "Type 2 Diabetes"
    )

    interventions_text = st.text_input(
        "Interventions tested (comma-separated)",
        "Drug X"
    )

    num_conditions = max(
        1,
        len(
            [
                c
                for c
                in conditions_text.split(",")
                if c.strip()
            ]
        )
    )

    num_interventions = max(
        1,
        len(
            [
                i
                for i
                in interventions_text.split(",")
                if i.strip()
            ]
        )
    )

    st.caption(
        f"→ {num_conditions} condition(s), "
        f"{num_interventions} intervention(s)"
    )

    # --------------------------------------------------------------
    # Eligibility
    # --------------------------------------------------------------

    st.subheader(
        "Eligibility"
    )

    includes_child = st.checkbox(
        "Includes pediatric participants (Child)"
    )

    includes_adult = st.checkbox(
        "Includes adult participants",
        value=True
    )

    includes_older_adult = st.checkbox(
        "Includes older-adult participants"
    )

    # --------------------------------------------------------------
    # Sponsor and design
    # --------------------------------------------------------------

    st.subheader(
        "Sponsor & design"
    )

    org_class = st.selectbox(
        "Sponsor organization class",
        ORG_CLASS_OPTIONS,
        index=0
    )

    responsible_party = st.selectbox(
        "Responsible party",
        RESPONSIBLE_PARTY_OPTIONS
    )

    primary_purpose = st.selectbox(
        "Primary purpose",
        PURPOSE_OPTIONS
    )

    study_type = st.selectbox(
        "Study type",
        [
            "INTERVENTIONAL",
            "OBSERVATIONAL"
        ]
    )

    phase = st.selectbox(
        "Phase",
        PHASE_OPTIONS
    )

    start_year = st.number_input(
        "Planned start year",
        min_value=1990,
        max_value=2035,
        value=2024
    )

    predict_clicked = st.button(
        "Predict trial outcome",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------------------------
# Load model
# --------------------------------------------------------------------

try:

    model = load_model()

except FileNotFoundError:

    st.error(
        f"Couldn't find `{MODEL_PATH}`. "
        "Place the saved model at that path and rerun."
    )

    st.stop()

except Exception as e:

    st.error(
        f"Could not load the model: "
        f"{type(e).__name__}: {e}"
    )

    st.stop()


# --------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------

if predict_clicked:

    inputs = {
        "brief_title": brief_title,
        "full_title": full_title,
        "intervention_description": (
            intervention_description
        ),
        "num_conditions": num_conditions,
        "num_interventions": num_interventions,
        "includes_child": includes_child,
        "includes_adult": includes_adult,
        "includes_older_adult": includes_older_adult,
        "org_class": org_class,
        "responsible_party": responsible_party,
        "primary_purpose": primary_purpose,
        "study_type": study_type,
        "phase": phase,
        "start_year": start_year,
    }

    # --------------------------------------------------------------
    # Build input row
    # --------------------------------------------------------------

    input_row = build_input_row(
        inputs
    )

    # --------------------------------------------------------------
    # Predict calibrated probability
    # --------------------------------------------------------------

    try:

        p_success = model.predict_proba(
            input_row
        )[0, 1]

    except Exception as e:

        st.error(
            f"Prediction error: "
            f"{type(e).__name__}: {e}"
        )

        st.stop()

    tier = risk_tier(
        p_success
    )

    # --------------------------------------------------------------
    # Prediction metrics
    # --------------------------------------------------------------

    col1, col2 = st.columns(2)

    col1.metric(
        "Predicted completion probability",
        f"{p_success:.1%}"
    )

    col2.metric(
        "Risk tier",
        tier
    )

    # --------------------------------------------------------------
    # SHAP explanation
    # --------------------------------------------------------------

    st.divider()

    st.subheader(
        "Why the model landed here"
    )

    shap_contributions = explain_prediction(
        model,
        input_row
    )

    if shap_contributions is not None:

        top = (
            shap_contributions
            .head(10)
            .iloc[::-1]
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        bar_colors = [
            "#d62728"
            if value < 0
            else "#2ca02c"
            for value
            in top.values
        ]

        ax.barh(
            top.index,
            top.values,
            color=bar_colors
        )

        ax.axvline(
            0,
            linewidth=0.8
        )

        ax.set_xlabel(
            "SHAP value "
            "(pushes prediction toward success / failure)"
        )

        ax.set_title(
            "Top feature contributions for this trial"
        )

        plt.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(
            fig
        )

        st.caption(
            "Green bars push the underlying tree model "
            "toward **success**; red bars push it toward "
            "**failure**. SHAP describes model behavior, "
            "not real-world causality."
        )

    else:

        st.info(
            "The prediction above is valid, but a local "
            "SHAP feature breakdown could not be generated."
        )

    # --------------------------------------------------------------
    # Exact feature row
    # --------------------------------------------------------------

    with st.expander(
        "Show the exact feature row sent to the model"
    ):

        st.dataframe(
            input_row.T.rename(
                columns={0: "value"}
            )
        )

else:

    st.info(
        "Fill in the trial details in the sidebar, "
        "then click **Predict trial outcome**."
    )


# --------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------

st.divider()

st.caption(
    "This tool reflects a model trained on historical "
    "ClinicalTrials.gov registry data. It reports statistical "
    "association, not clinical efficacy or causality, and should "
    "support — not replace — human portfolio review."
)
