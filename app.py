"""
Clinical Trial Success Prediction — Streamlit App
==================================================
Loads the fitted pipeline produced in Phase 8.5 of the notebook
(models/final_model.pkl — a single sklearn Pipeline bundling the
ColumnTransformer preprocessor + the tuned classifier) and scores a
single, user-described trial: predicted P(success), a risk tier
(High / Medium / Low, same cut points as notebook Phase 7.3), and a
local SHAP explanation of why the model landed on that number.

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

# The exact 17 raw columns the fitted pipeline's ColumnTransformer expects,
# in the same names used during training (Phase 5A.1 of the notebook).
NUMERIC_FEATURES = [
    "num_conditions", "num_interventions", "brief_title_word_count",
    "full_title_word_count", "intervention_description_word_count",
    "start_year", "trial_complexity_index",
]
BINARY_FEATURES = [
    "is_multi_condition", "is_multi_intervention",
    "includes_child", "includes_adult", "includes_older_adult",
]
CATEGORICAL_FEATURES = [
    "sponsor_type_grouped", "Responsible Party", "Primary Purpose", "Study Type", "Phases",
]
MODEL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Same grouping rule used in notebook Phase 3.5, applied here to the raw
# "Organization Class" value so the form asks for something a real user
# would actually know, and the app derives the engineered category.
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

PHASE_OPTIONS = [
    "NOT_APPLICABLE", "EARLY_PHASE1", "PHASE1", "PHASE1, PHASE2",
    "PHASE2", "PHASE2, PHASE3", "PHASE3", "PHASE4", "Unknown",
]
PURPOSE_OPTIONS = [
    "TREATMENT", "PREVENTION", "DIAGNOSTIC", "SUPPORTIVE_CARE", "SCREENING",
    "HEALTH_SERVICES_RESEARCH", "BASIC_SCIENCE", "DEVICE_FEASIBILITY", "ECT",
    "OTHER", "Unknown",
]
RESPONSIBLE_PARTY_OPTIONS = ["SPONSOR", "PRINCIPAL_INVESTIGATOR", "Unknown"]
ORG_CLASS_OPTIONS = list(SPONSOR_GROUP_MAP.keys()) + ["UNKNOWN"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def risk_tier(p_success: float) -> str:
    # Same cut points as notebook Phase 7.3
    if p_success < 0.40:
        return "High Risk (<40%)"
    elif p_success < 0.70:
        return "Medium Risk (40-70%)"
    return "Low Risk (>70%)"


def build_input_row(inputs: dict) -> pd.DataFrame:
    num_conditions = inputs["num_conditions"]
    num_interventions = inputs["num_interventions"]

    row = {
        "num_conditions": num_conditions,
        "num_interventions": num_interventions,
        "brief_title_word_count": len(inputs["brief_title"].split()),
        "full_title_word_count": len(inputs["full_title"].split()),
        "intervention_description_word_count": len(inputs["intervention_description"].split()),
        "start_year": inputs["start_year"],
        "trial_complexity_index": num_conditions + num_interventions,
        "is_multi_condition": int(num_conditions > 1),
        "is_multi_intervention": int(num_interventions > 1),
        "includes_child": int(inputs["includes_child"]),
        "includes_adult": int(inputs["includes_adult"]),
        "includes_older_adult": int(inputs["includes_older_adult"]),
        "sponsor_type_grouped": SPONSOR_GROUP_MAP.get(inputs["org_class"], "OTHER"),
        "Responsible Party": inputs["responsible_party"],
        "Primary Purpose": inputs["primary_purpose"],
        "Study Type": inputs["study_type"],
        "Phases": inputs["phase"],
    }
    return pd.DataFrame([row], columns=MODEL_FEATURES)


def explain_prediction(pipeline, input_row: pd.DataFrame):
    """Best-effort local SHAP explanation. Returns None if the final
    classifier isn't tree-based (TreeExplainer doesn't apply)."""
    try:
        import shap
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        transformed = preprocessor.transform(input_row)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.TreeExplainer(classifier)
        raw_shap = explainer.shap_values(transformed)

        if isinstance(raw_shap, list):
            # Older SHAP: list of per-class arrays, shape (n_samples, n_features)
            shap_row = raw_shap[1][0]
        elif raw_shap.ndim == 3:
            # Newer SHAP: single array, shape (n_samples, n_features, n_classes)
            shap_row = raw_shap[0, :, 1]
        else:
            # Binary-only output, shape (n_samples, n_features)
            shap_row = raw_shap[0]

        return pd.Series(shap_row, index=feature_names).sort_values(key=abs, ascending=False)
    except Exception:
        return None


# --------------------------------------------------------------------
# Sidebar — trial inputs
# --------------------------------------------------------------------
st.title("🧪 Clinical Trial Outcome Risk Predictor")
st.caption(
    "Portfolio demo — predicts an **operational completion** probability "
    "(Completed vs. Terminated/Withdrawn/Suspended) from design-time trial "
    "characteristics only. Not a clinical, efficacy, or investment recommendation."
)

with st.sidebar:
    st.header("Describe the trial")

    brief_title = st.text_input("Brief title", "A Study of Drug X in Adult Patients")
    full_title = st.text_input(
        "Full title",
        "A Randomized, Double-Blind Study of Drug X in Adult Patients With Condition Y",
    )
    intervention_description = st.text_area(
        "Intervention description",
        "Participants receive Drug X or placebo once daily for 12 weeks.",
    )

    st.subheader("Design complexity")
    conditions_text = st.text_input("Conditions studied (comma-separated)", "Type 2 Diabetes")
    interventions_text = st.text_input("Interventions tested (comma-separated)", "Drug X")
    num_conditions = max(1, len([c for c in conditions_text.split(",") if c.strip()]))
    num_interventions = max(1, len([i for i in interventions_text.split(",") if i.strip()]))
    st.caption(f"→ {num_conditions} condition(s), {num_interventions} intervention(s)")

    st.subheader("Eligibility")
    includes_child = st.checkbox("Includes pediatric participants (Child)")
    includes_adult = st.checkbox("Includes adult participants", value=True)
    includes_older_adult = st.checkbox("Includes older-adult participants")

    st.subheader("Sponsor & design")
    org_class = st.selectbox("Sponsor organization class", ORG_CLASS_OPTIONS, index=0)
    responsible_party = st.selectbox("Responsible party", RESPONSIBLE_PARTY_OPTIONS)
    primary_purpose = st.selectbox("Primary purpose", PURPOSE_OPTIONS)
    study_type = st.selectbox("Study type", ["INTERVENTIONAL", "OBSERVATIONAL"])
    phase = st.selectbox("Phase", PHASE_OPTIONS)
    start_year = st.number_input("Planned start year", min_value=1990, max_value=2035, value=2024)

    predict_clicked = st.button("Predict trial outcome", type="primary", use_container_width=True)

# --------------------------------------------------------------------
# Main panel — prediction + explanation
# --------------------------------------------------------------------
try:
    pipeline = load_model()
except FileNotFoundError:
    st.error(
        f"Couldn't find `{MODEL_PATH}`. Place the saved pipeline from notebook "
        "Phase 8.5 at that path (see the README's folder layout) and rerun."
    )
    st.stop()

if predict_clicked:
    inputs = dict(
        brief_title=brief_title, full_title=full_title,
        intervention_description=intervention_description,
        num_conditions=num_conditions, num_interventions=num_interventions,
        includes_child=includes_child, includes_adult=includes_adult,
        includes_older_adult=includes_older_adult, org_class=org_class,
        responsible_party=responsible_party, primary_purpose=primary_purpose,
        study_type=study_type, phase=phase, start_year=start_year,
    )
    input_row = build_input_row(inputs)

    p_success = pipeline.predict_proba(input_row)[0, 1]
    tier = risk_tier(p_success)

    col1, col2 = st.columns(2)
col1.metric("Predicted completion probability", f"{p_success:.1%}")
col2.metric("Risk tier", tier)


    st.divider()
    st.subheader("Why the model landed here")
    shap_contributions = explain_prediction(pipeline, input_row)

    if shap_contributions is not None:
        top = shap_contributions.head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        colors = ["#d62728" if v < 0 else "#2ca02c" for v in top.values]
        ax.barh(top.index, top.values, color=colors)
        ax.set_xlabel("SHAP value (→ pushes toward success / away from it)")
        ax.set_title("Top feature contributions for this trial")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption(
            "Green bars push the prediction toward **success**; red bars push it "
            "toward **failure**. This explains the model's reasoning, not a claim "
            "about real-world causality."
        )
    else:
        st.info(
            "Local SHAP explanation isn't available for this model type — the "
            "prediction above is still valid, just without the feature breakdown."
        )

    with st.expander("Show the exact feature row sent to the model"):
        st.dataframe(input_row.T.rename(columns={0: "value"}))
else:
    st.info("Fill in the trial details in the sidebar, then click **Predict trial outcome**.")

st.divider()
st.caption(
    "This tool reflects a model trained on historical ClinicalTrials.gov registry "
    "data. It reports statistical association, not clinical efficacy or causality, "
    "and should support — not replace — human portfolio review."
)
