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
            len(
                inputs["intervention_description"].split()
            )
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
# SHAP feature display names
# --------------------------------------------------------------------

DISPLAY_NAME_MAP = {
    "num__num_conditions":
        "Number of Conditions",

    "num__num_interventions":
        "Number of Interventions",

    "num__brief_title_word_count":
        "Brief Title Length",

    "num__full_title_word_count":
        "Full Title Length",

    "num__intervention_description_word_count":
        "Intervention Description Length",

    "num__start_year":
        "Start Year",

    "num__trial_complexity_index":
        "Trial Complexity",

    "bin__is_multi_condition":
        "Multiple Conditions",

    "bin__is_multi_intervention":
        "Multiple Interventions",

    "bin__includes_child":
        "Includes Child",

    "bin__includes_adult":
        "Includes Adult",

    "bin__includes_older_adult":
        "Includes Older Adult",

    "cat__sponsor_type_grouped_INDUSTRY":
        "Sponsor: Industry",

    "cat__sponsor_type_grouped_GOVERNMENT":
        "Sponsor: Government",

    "cat__sponsor_type_grouped_ACADEMIC_OR_NETWORK":
        "Sponsor: Academic / Network",

    "cat__sponsor_type_grouped_ACADEMIC_OR_OTHER":
        "Sponsor: Academic / Other",

    "cat__Responsible Party_SPONSOR":
        "Responsible Party: Sponsor",

    "cat__Responsible Party_PRINCIPAL_INVESTIGATOR":
        "Responsible Party: Principal Investigator",

    "cat__Responsible Party_Unknown":
        "Responsible Party: Unknown",

    "cat__Primary Purpose_TREATMENT":
        "Purpose: Treatment",

    "cat__Primary Purpose_PREVENTION":
        "Purpose: Prevention",

    "cat__Primary Purpose_DIAGNOSTIC":
        "Purpose: Diagnostic",

    "cat__Primary Purpose_SUPPORTIVE_CARE":
        "Purpose: Supportive Care",

    "cat__Primary Purpose_SCREENING":
        "Purpose: Screening",

    "cat__Primary Purpose_HEALTH_SERVICES_RESEARCH":
        "Purpose: Health Services Research",

    "cat__Primary Purpose_BASIC_SCIENCE":
        "Purpose: Basic Science",

    "cat__Primary Purpose_DEVICE_FEASIBILITY":
        "Purpose: Device Feasibility",

    "cat__Primary Purpose_ECT":
        "Purpose: ECT",

    "cat__Primary Purpose_OTHER":
        "Purpose: Other",

    "cat__Primary Purpose_Unknown":
        "Purpose: Unknown",

    "cat__Study Type_INTERVENTIONAL":
        "Study Type: Interventional",

    "cat__Study Type_OBSERVATIONAL":
        "Study Type: Observational",

    "cat__Phases_NOT_APPLICABLE":
        "Phase: Not Applicable",

    "cat__Phases_EARLY_PHASE1":
        "Phase: Early Phase 1",

    "cat__Phases_PHASE1":
        "Phase: Phase 1",

    "cat__Phases_PHASE1, PHASE2":
        "Phase: Phase 1 / 2",

    "cat__Phases_PHASE2":
        "Phase: Phase 2",

    "cat__Phases_PHASE2, PHASE3":
        "Phase: Phase 2 / 3",

    "cat__Phases_PHASE3":
        "Phase: Phase 3",

    "cat__Phases_PHASE4":
        "Phase: Phase 4",

    "cat__Phases_Unknown":
        "Phase: Unknown",
}


def clean_feature_name(feature_name: str) -> str:
    """
    Convert transformed feature names into recruiter-friendly labels.
    Uses an explicit mapping where available and a generic fallback
    for any unexpected feature names.
    """

    if feature_name in DISPLAY_NAME_MAP:
        return DISPLAY_NAME_MAP[feature_name]

    cleaned = feature_name

    cleaned = cleaned.replace("num__", "")
    cleaned = cleaned.replace("bin__", "")
    cleaned = cleaned.replace("cat__", "")

    cleaned = cleaned.replace("_", " ")

    return cleaned.title()


# --------------------------------------------------------------------
# SHAP explanation
# --------------------------------------------------------------------

def explain_prediction(
    model,
    input_row: pd.DataFrame
):
    """
    Generate a local SHAP explanation.

    The saved model is a CalibratedClassifierCV. For SHAP, the
    underlying fitted tree estimator(s) are extracted and explained.

    The displayed prediction remains the calibrated probability
    returned by model.predict_proba().
    """

    try:

        import shap

        # ------------------------------------------------------------
        # Find fitted base estimator(s)
        # ------------------------------------------------------------

        base_models = []

        # Normal sklearn Pipeline
        if hasattr(
            model,
            "named_steps"
        ):

            base_models.append(model)

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

        # Direct estimator
        else:

            base_models.append(model)

        if not base_models:

            raise ValueError(
                "Could not find fitted base estimator(s) "
                "inside the saved model."
            )

        # ------------------------------------------------------------
        # SHAP for each underlying estimator
        # ------------------------------------------------------------

        all_shap_values = []

        feature_names = None

        for base_model in base_models:

            # --------------------------------------------------------
            # Pipeline containing preprocessor + classifier
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

                if "classifier" not in (
                    base_model.named_steps
                ):

                    raise ValueError(
                        "The fitted pipeline does not contain "
                        "a 'classifier' step."
                    )

                preprocessor = (
                    base_model.named_steps[
                        "preprocessor"
                    ]
                )

                estimator = (
                    base_model.named_steps[
                        "classifier"
                    ]
                )

                # Same preprocessing as training
                transformed = (
                    preprocessor.transform(
                        input_row
                    )
                )

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

            # --------------------------------------------------------
            # Direct estimator
            # --------------------------------------------------------

            else:

                estimator = base_model

                transformed = (
                    input_row.to_numpy()
                )

                feature_names = np.asarray(
                    input_row.columns
                )

            # --------------------------------------------------------
            # Tree SHAP
            # --------------------------------------------------------

            explainer = shap.TreeExplainer(
                estimator
            )

            raw_shap = explainer.shap_values(
                transformed,
                check_additivity=False
            )

            # --------------------------------------------------------
            # SHAP output formats
            # --------------------------------------------------------

            if isinstance(
                raw_shap,
                list
            ):

                # Older SHAP versions
                shap_row = raw_shap[1][0]

            elif getattr(
                raw_shap,
                "ndim",
                0
            ) == 3:

                # Newer SHAP versions
                shap_row = raw_shap[
                    0,
                    :,
                    1
                ]

            else:

                # Binary classification
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

        shap_series = (
            pd.Series(
                shap_values,
                index=feature_names
            )
            .sort_values(
                key=np.abs,
                ascending=False
            )
        )

        return shap_series

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
                for c in conditions_text.split(",")
                if c.strip()
            ]
        )
    )

    num_interventions = max(
        1,
        len(
            [
                i
                for i in interventions_text.split(",")
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
    # Build input
    # --------------------------------------------------------------

    input_row = build_input_row(
        inputs
    )

    # --------------------------------------------------------------
    # Prediction
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

        # Top 10 absolute SHAP contributors
        top = (
            shap_contributions
            .head(10)
            .iloc[::-1]
            .copy()
        )

        # Convert technical feature names into readable labels
        top.index = [
            clean_feature_name(feature)
            for feature in top.index
        ]

        # ----------------------------------------------------------
        # Plot
        # ----------------------------------------------------------

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
            "Top Feature Contributions for This Trial"
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
