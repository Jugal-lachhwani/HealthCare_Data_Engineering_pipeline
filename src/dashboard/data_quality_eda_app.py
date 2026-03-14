from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split


DEFAULT_CONFIG_PATH = Path("config/dashboard_config.json")

NUMERIC_COLS = [
    "age",
    "hospitalid",
    "wardid",
    "los",
    "LACE_Score",
    "bp_systolic",
    "bp_diastolic",
    "pulse",
    "respirations",
    "temperature",
    "bmi",
    "glucose",
    "creatinine",
    "wbc",
    "hemoglobin",
    "potassium",
    "sodium",
    "calcium",
    "ed_visits",
    "ip_visits",
    "chronic_conditions",
    "cost_of_initial_stay",
    "care_plan_costs",
    "days_until_readmission",
    "readmitted",
    "readmitted_under_30_days",
]

MODEL_FEATURE_NUMERIC = [
    "age",
    "los",
    "LACE_Score",
    "glucose",
    "creatinine",
    "wbc",
    "hemoglobin",
    "chronic_conditions",
    "ed_visits",
    "ip_visits",
    "bp_systolic",
    "bp_diastolic",
]

MODEL_FEATURE_CATEGORICAL = ["gender", "ethnic_group_c", "insurance_provider", "Condition", "unittype"]


@st.cache_data(show_spinner=False)
def load_dashboard_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=True)
def load_csv(path: str, mtime_ns: int) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def get_file_mtime_ns(path: str) -> int:
    csv_path = Path(path)
    if not csv_path.exists():
        return -1
    return csv_path.stat().st_mtime_ns


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()

    for col in NUMERIC_COLS:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")

    for col in ["readmitted", "readmitted_under_30_days"]:
        if col in working.columns:
            working[col] = working[col].fillna(0).astype(int)

    for col in ["admit_day", "discharge_day", "readmission_admit_day", "readmission_discharge_day"]:
        if col in working.columns:
            working[col] = pd.to_datetime(working[col], errors="coerce", dayfirst=True)

    # Build a robust monthly time key for trend charts.
    if "admit_day" not in working.columns:
        working["admit_day"] = pd.NaT

    if working["admit_day"].isna().all():
        fallback_date_cols = ["processed_at_utc", "_airbyte_emitted_at", "discharge_day", "readmission_admit_day"]
        for fallback_col in fallback_date_cols:
            if fallback_col in working.columns:
                fallback_ts = pd.to_datetime(working[fallback_col], errors="coerce", utc=True)
                if hasattr(fallback_ts.dt, "tz") and fallback_ts.dt.tz is not None:
                    fallback_ts = fallback_ts.dt.tz_localize(None)
                working["admit_day"] = working["admit_day"].fillna(fallback_ts)
                if working["admit_day"].notna().any():
                    break

    if "admit_day" in working.columns and working["admit_day"].notna().any():
        working["admit_month"] = working["admit_day"].dt.to_period("M").dt.to_timestamp()

    if "age" in working.columns:
        working["age_band"] = pd.cut(
            working["age"],
            bins=[0, 30, 45, 60, 75, 90, 120],
            labels=["0-30", "31-45", "46-60", "61-75", "76-90", "90+"],
            include_lowest=True,
        )

    lace = working.get("LACE_Score", pd.Series(index=working.index, dtype=float)).fillna(0)
    los = working.get("los", pd.Series(index=working.index, dtype=float)).fillna(0)
    chronic = working.get("chronic_conditions", pd.Series(index=working.index, dtype=float)).fillna(0)
    age = working.get("age", pd.Series(index=working.index, dtype=float)).fillna(0)

    risk = 0.35 * np.clip(lace / 19, 0, 1) + 0.2 * np.clip(los / 30, 0, 1) + 0.2 * np.clip(chronic / 15, 0, 1) + 0.25 * np.clip(age / 100, 0, 1)
    working["risk_score_live"] = np.clip(risk, 0, 1)

    def _bucket(v: float) -> str:
        if v >= 0.75:
            return "Intensive Care Transition"
        if v >= 0.5:
            return "Enhanced Follow-up"
        if v >= 0.3:
            return "Standard Follow-up"
        return "Routine Monitoring"

    working["suggested_intervention_bucket"] = working["risk_score_live"].map(_bucket)
    return working


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.subheader("Filters")
    filtered = df.copy()

    if "hospitalid" in filtered.columns and filtered["hospitalid"].notna().any():
        hospitals = sorted([int(x) for x in filtered["hospitalid"].dropna().unique().tolist()])
        selected_hospitals = st.sidebar.multiselect("Hospital IDs", options=hospitals, default=hospitals)
        if selected_hospitals:
            filtered = filtered[filtered["hospitalid"].isin(selected_hospitals)]

    if "unittype" in filtered.columns:
        units = sorted(filtered["unittype"].fillna("unknown").astype(str).unique().tolist())
        selected_units = st.sidebar.multiselect("Unit Types", options=units, default=units)
        if selected_units:
            filtered = filtered[filtered["unittype"].fillna("unknown").isin(selected_units)]

    if "gender" in filtered.columns:
        genders = sorted(filtered["gender"].fillna("unknown").astype(str).unique().tolist())
        selected_genders = st.sidebar.multiselect("Gender", options=genders, default=genders)
        if selected_genders:
            filtered = filtered[filtered["gender"].fillna("unknown").isin(selected_genders)]

    if "readmitted_under_30_days" in filtered.columns:
        outcome = st.sidebar.selectbox("30-Day Readmission", options=["All", "Readmitted", "Not Readmitted"], index=0)
        if outcome == "Readmitted":
            filtered = filtered[filtered["readmitted_under_30_days"] == 1]
        if outcome == "Not Readmitted":
            filtered = filtered[filtered["readmitted_under_30_days"] == 0]

    if "age" in filtered.columns and filtered["age"].notna().any():
        min_age = int(filtered["age"].min())
        max_age = int(filtered["age"].max())
        age_range = st.sidebar.slider("Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))
        filtered = filtered[(filtered["age"] >= age_range[0]) & (filtered["age"] <= age_range[1])]

    return filtered


def render_header() -> None:
    st.set_page_config(
        page_title="Readmission Intelligence Dashboard",
        page_icon="HEALTH",
        layout="wide",
    )

    st.title("Readmission Intelligence Dashboard")
    st.caption("Model-ready analytics on EHR_FInal data for readmission risk and reason analysis.")


def render_empty_state() -> None:
    st.warning("No final CSV dataset found yet. Run final output export first.")


def render_executive_kpi_cards(df: pd.DataFrame) -> None:
    st.subheader("1) Executive KPI Cards")
    total = len(df)
    readmit_any = int(df["readmitted"].sum()) if "readmitted" in df.columns else 0
    readmit_30 = int(df["readmitted_under_30_days"].sum()) if "readmitted_under_30_days" in df.columns else 0
    readmit_any_rate = (readmit_any / total * 100) if total else 0.0
    readmit_30_rate = (readmit_30 / total * 100) if total else 0.0
    avg_days = float(df.loc[df["readmitted"] == 1, "days_until_readmission"].dropna().mean()) if {"readmitted", "days_until_readmission"}.issubset(df.columns) else 0.0
    high_risk = int((df["risk_score_live"] >= 0.75).sum()) if "risk_score_live" in df.columns else 0
    high_risk_rate = (high_risk / total * 100) if total else 0.0

    avg_los = float(pd.to_numeric(df["los"], errors="coerce").mean()) if "los" in df.columns else 0.0
    median_los = float(pd.to_numeric(df["los"], errors="coerce").median()) if "los" in df.columns else 0.0
    avg_lace = float(pd.to_numeric(df["LACE_Score"], errors="coerce").mean()) if "LACE_Score" in df.columns else 0.0
    avg_chronic = float(pd.to_numeric(df["chronic_conditions"], errors="coerce").mean()) if "chronic_conditions" in df.columns else 0.0
    avg_initial_cost = float(pd.to_numeric(df["cost_of_initial_stay"], errors="coerce").mean()) if "cost_of_initial_stay" in df.columns else 0.0
    avg_care_plan_cost = float(pd.to_numeric(df["care_plan_costs"], errors="coerce").mean()) if "care_plan_costs" in df.columns else 0.0

    multi_chronic_count = (
        int((pd.to_numeric(df["chronic_conditions"], errors="coerce").fillna(0) >= 2).sum())
        if "chronic_conditions" in df.columns
        else 0
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Admissions", total)
    c2.metric("Total Readmissions (Any)", readmit_any)
    c3.metric("30-Day Readmission Count", readmit_30)
    c4.metric("30-Day Readmission Rate", f"{readmit_30_rate:.2f}%")
    c5.metric("Avg Days to Readmission", f"{avg_days:.2f}")
    c6.metric("High-Risk Patients Flagged", high_risk)

    c7, c8, c9, c10, c11, c12 = st.columns(6)
    c7.metric("Any Readmission Rate", f"{readmit_any_rate:.2f}%")
    c8.metric("High-Risk Rate", f"{high_risk_rate:.2f}%")
    c9.metric("Avg LOS (Days)", f"{avg_los:.2f}")
    c10.metric("Median LOS (Days)", f"{median_los:.2f}")
    c11.metric("Avg LACE Score", f"{avg_lace:.2f}")
    c12.metric("Avg Chronic Conditions", f"{avg_chronic:.2f}")

    c13, c14, c15 = st.columns(3)
    c13.metric("Patients with >=2 Chronic", multi_chronic_count)
    c14.metric("Avg Initial Stay Cost", f"{avg_initial_cost:.2f}")
    c15.metric("Avg Care Plan Cost", f"{avg_care_plan_cost:.2f}")


def render_readmission_trend(df: pd.DataFrame) -> None:
    st.subheader("2) Readmission Trend")
    if "admit_month" not in df.columns or df["admit_month"].isna().all():
        st.info("No usable timestamp found for trend charts (checked admit_day and fallback timestamp fields).")
        return

    trend = (
        df.groupby("admit_month", dropna=False)
        .agg(
            admissions=("patientunitstayid", "count"),
            readmissions_any=("readmitted", "sum"),
            readmissions_30d=("readmitted_under_30_days", "sum"),
        )
        .reset_index()
        .sort_values("admit_month")
    )
    trend["readmission_30d_rate_pct"] = trend["readmissions_30d"] / trend["admissions"] * 100
    trend["readmission_30d_rate_ma"] = trend["readmission_30d_rate_pct"].rolling(window=3, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend["admit_month"], y=trend["admissions"], mode="lines+markers", name="Admissions"))
    fig.add_trace(go.Scatter(x=trend["admit_month"], y=trend["readmissions_any"], mode="lines+markers", name="Readmissions (Any)"))
    fig.update_layout(title="Monthly Admissions vs Monthly Readmissions")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=trend["admit_month"], y=trend["readmission_30d_rate_pct"], mode="lines+markers", name="30-Day Rate"))
    fig2.add_trace(go.Scatter(x=trend["admit_month"], y=trend["readmission_30d_rate_ma"], mode="lines", name="3-Month Moving Avg"))
    fig2.update_layout(title="Monthly 30-Day Readmission Rate (%)")
    st.plotly_chart(fig2, use_container_width=True)


def render_cohort_segmentation(df: pd.DataFrame) -> None:
    st.subheader("3) Cohort and Segmentation")

    if "age_band" in df.columns:
        age_seg = (
            df.groupby("age_band", observed=False)["readmitted_under_30_days"]
            .mean()
            .mul(100)
            .reset_index(name="readmission_rate_pct")
        )
        fig = px.bar(age_seg, x="age_band", y="readmission_rate_pct", title="Readmission Rate by Age Band")
        st.plotly_chart(fig, use_container_width=True)

    seg_frames = []
    for col in ["gender", "ethnic_group_c", "insurance_provider"]:
        if col in df.columns:
            tmp = df.groupby(col, dropna=False)["readmitted_under_30_days"].mean().mul(100).reset_index()
            tmp.columns = ["segment_value", "readmission_rate_pct"]
            tmp["segment"] = col
            seg_frames.append(tmp)
    if seg_frames:
        seg_df = pd.concat(seg_frames, ignore_index=True)
        fig = px.bar(seg_df, x="segment_value", y="readmission_rate_pct", color="segment", barmode="group", title="Readmission Rate by Gender / Ethnic Group / Insurance")
        st.plotly_chart(fig, use_container_width=True)

    if "Condition" in df.columns:
        cond = (
            df.groupby("Condition", dropna=False)["readmitted_under_30_days"].mean().mul(100).reset_index(name="readmission_rate_pct")
            .sort_values("readmission_rate_pct", ascending=False)
            .head(10)
        )
        fig = px.bar(cond, x="readmission_rate_pct", y="Condition", orientation="h", title="Top 10 Conditions by 30-Day Readmission Rate")
        st.plotly_chart(fig, use_container_width=True)

    if {"hospitalid", "unittype", "readmitted_under_30_days"}.issubset(df.columns):
        heat = (
            df.groupby(["hospitalid", "unittype"], dropna=False)["readmitted_under_30_days"].mean().mul(100).reset_index(name="rate")
        )
        fig = px.density_heatmap(heat, x="hospitalid", y="unittype", z="rate", histfunc="avg", title="Readmission Rate Heatmap by Hospital / Unit Type")
        st.plotly_chart(fig, use_container_width=True)


def render_clinical_risk_drivers(df: pd.DataFrame) -> None:
    st.subheader("4) Clinical Risk Drivers")

    if {"LACE_Score", "readmitted_under_30_days"}.issubset(df.columns):
        fig = px.histogram(df, x="LACE_Score", color=df["readmitted_under_30_days"].map({0: "No", 1: "Yes"}), barmode="overlay", nbins=30, title="LACE Score Distribution by 30-Day Readmission")
        st.plotly_chart(fig, use_container_width=True)

    lab_cols = ["glucose", "creatinine", "wbc", "hemoglobin"]
    labs_present = [c for c in lab_cols if c in df.columns]
    if labs_present and "readmitted_under_30_days" in df.columns:
        melted = df.melt(id_vars=["readmitted_under_30_days"], value_vars=labs_present, var_name="lab", value_name="value")
        fig = px.box(melted, x="lab", y="value", color=melted["readmitted_under_30_days"].map({0: "No", 1: "Yes"}), title="Top Lab Values vs Readmission Class")
        st.plotly_chart(fig, use_container_width=True)

    if {"chronic_conditions", "readmitted_under_30_days"}.issubset(df.columns):
        cc = (
            df.groupby("chronic_conditions", dropna=False)["readmitted_under_30_days"].mean().mul(100).reset_index(name="readmission_rate_pct")
        )
        fig = px.line(cc.sort_values("chronic_conditions"), x="chronic_conditions", y="readmission_rate_pct", markers=True, title="Comorbidity Burden vs Readmission Probability")
        st.plotly_chart(fig, use_container_width=True)

    if {"los", "readmitted_under_30_days"}.issubset(df.columns):
        fig = px.box(df, x=df["readmitted_under_30_days"].map({0: "No", 1: "Yes"}), y="los", title="Length of Stay vs Readmission")
        st.plotly_chart(fig, use_container_width=True)


def render_data_quality(df: pd.DataFrame) -> None:
    st.subheader("5) Data Quality")

    missing = pd.DataFrame(
        {
            "feature": df.columns,
            "missing_pct": [df[col].isna().mean() * 100 for col in df.columns],
        }
    ).sort_values("missing_pct", ascending=False)
    fig = px.bar(missing.head(30), x="feature", y="missing_pct", title="Missing Value % by Feature")
    st.plotly_chart(fig, use_container_width=True)

    if "admit_month" in df.columns and df["admit_month"].notna().any():
        key_features = [c for c in ["LACE_Score", "los", "glucose", "creatinine", "wbc", "hemoglobin", "readmitted_under_30_days"] if c in df.columns]
        comp = df.groupby("admit_month", dropna=False)[key_features].apply(lambda x: x.notna().mean().mean() * 100).reset_index(name="completeness_pct")
        fig = px.line(comp.sort_values("admit_month"), x="admit_month", y="completeness_pct", markers=True, title="Completeness Over Time")
        st.plotly_chart(fig, use_container_width=True)

    dup_count = int(df.duplicated(subset=[c for c in ["patientunitstayid", "admit_day"] if c in df.columns]).sum())
    c1, c2 = st.columns(2)
    c1.metric("Duplicate Records", dup_count)
    c2.metric("Duplicate %", f"{(dup_count / len(df) * 100 if len(df) else 0):.2f}%")

    if "admit_month" in df.columns and "patientunitstayid" in df.columns:
        is_dup = df.duplicated(subset=["patientunitstayid", "admit_day"], keep=False) if "admit_day" in df.columns else df.duplicated(subset=["patientunitstayid"], keep=False)
        dup_month = df.assign(is_dup=is_dup).groupby("admit_month", dropna=False)["is_dup"].sum().reset_index(name="duplicate_rows")
        fig = px.line(dup_month.sort_values("admit_month"), x="admit_month", y="duplicate_rows", markers=True, title="Duplicate Records Trend")
        st.plotly_chart(fig, use_container_width=True)

    lb1, lb2 = st.columns(2)
    if "readmitted" in df.columns:
        class_any = df["readmitted"].value_counts().rename(index={0: "No", 1: "Yes"}).reset_index()
        class_any.columns = ["class", "count"]
        fig = px.bar(class_any, x="class", y="count", title="Label Balance: readmitted")
        lb1.plotly_chart(fig, use_container_width=True)
    if "readmitted_under_30_days" in df.columns:
        class_30 = df["readmitted_under_30_days"].value_counts().rename(index={0: "No", 1: "Yes"}).reset_index()
        class_30.columns = ["class", "count"]
        fig = px.bar(class_30, x="class", y="count", title="Label Balance: readmitted_under_30_days")
        lb2.plotly_chart(fig, use_container_width=True)


def _fit_model(df: pd.DataFrame) -> dict[str, Any] | None:
    if "readmitted_under_30_days" not in df.columns:
        return None

    target = df["readmitted_under_30_days"].fillna(0).astype(int)
    if target.nunique() < 2 or len(df) < 50:
        return None

    feature_cols_num = [c for c in MODEL_FEATURE_NUMERIC if c in df.columns]
    feature_cols_cat = [c for c in MODEL_FEATURE_CATEGORICAL if c in df.columns]
    if not feature_cols_num and not feature_cols_cat:
        return None

    X = df[feature_cols_num + feature_cols_cat].copy()
    for c in feature_cols_num:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(pd.to_numeric(X[c], errors="coerce").median())
    for c in feature_cols_cat:
        X[c] = X[c].fillna("unknown").astype(str)

    X_enc = pd.get_dummies(X, columns=feature_cols_cat, dummy_na=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, target, test_size=0.3, random_state=42, stratify=target
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_score >= 0.5).astype(int)
    all_score = model.predict_proba(X_enc)[:, 1]

    feature_importance = pd.DataFrame(
        {
            "feature": X_enc.columns,
            "importance": np.abs(model.coef_[0]),
        }
    ).sort_values("importance", ascending=False)

    return {
        "y_test": y_test,
        "y_score": y_score,
        "y_pred_default": y_pred_default,
        "feature_importance": feature_importance,
        "all_score": all_score,
    }


def render_model_performance(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("6) Model Performance")
    model_out = _fit_model(df)
    if model_out is None:
        st.info("Not enough class diversity or required features to compute model performance.")
        return df

    y_test = model_out["y_test"]
    y_score = model_out["y_score"]

    threshold = st.slider("Decision Threshold", min_value=0.05, max_value=0.95, value=0.5, step=0.01)
    y_pred = (y_score >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig_cm = px.imshow(cm, text_auto=True, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"], title="Confusion Matrix")
    st.plotly_chart(fig_cm, use_container_width=True)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC AUC={roc_auc:.3f}"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline", line=dict(dash="dash")))
    fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig_roc, use_container_width=True)

    precision, recall, _ = precision_recall_curve(y_test, y_score)
    pr_auc = auc(recall, precision)
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"PR AUC={pr_auc:.3f}"))
    fig_pr.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(fig_pr, use_container_width=True)

    thresholds = np.arange(0.05, 0.96, 0.05)
    rows = []
    for th in thresholds:
        yp = (y_score >= th).astype(int)
        tp = int(((yp == 1) & (y_test == 1)).sum())
        fp = int(((yp == 1) & (y_test == 0)).sum())
        fn = int(((yp == 0) & (y_test == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        rows.append({"threshold": th, "precision": p, "recall": r, "f1": f1})
    tune_df = pd.DataFrame(rows)
    fig_thr = px.line(tune_df, x="threshold", y=["precision", "recall", "f1"], markers=True, title="Threshold Tuning: Precision / Recall / F1")
    st.plotly_chart(fig_thr, use_container_width=True)

    imp = model_out["feature_importance"].head(20)
    fig_imp = px.bar(imp, x="importance", y="feature", orientation="h", title="Feature Importance (Logistic Coefficient Magnitude)")
    st.plotly_chart(fig_imp, use_container_width=True)

    scored = df.copy()
    scored["predicted_readmission_probability"] = model_out["all_score"]
    return scored


def render_operational_action_view(df: pd.DataFrame) -> None:
    st.subheader("7) Operational Action View")

    risk_col = "predicted_readmission_probability" if "predicted_readmission_probability" in df.columns else "risk_score_live"

    fig = px.histogram(df, x=risk_col, nbins=30, title="Risk Distribution of Current Patients")
    st.plotly_chart(fig, use_container_width=True)

    top_cols = [
        "patientunitstayid",
        risk_col,
        "LACE_Score",
        "los",
        "chronic_conditions",
        "suggested_intervention_bucket",
    ]
    available = [c for c in top_cols if c in df.columns]
    top = df.sort_values(risk_col, ascending=False).head(30)
    st.markdown("### Top High-Risk Patients")
    st.dataframe(top[available], use_container_width=True)

    efficacy = st.slider("Intervention Effectiveness (%)", min_value=5, max_value=80, value=30, step=5) / 100.0
    n = len(df)
    capacities = sorted(set([50, 100, 200, 500, min(1000, n)]))
    scenario_rows = []
    ranked = df.sort_values(risk_col, ascending=False)
    for cap in capacities:
        k = min(cap, n)
        prevented = float(ranked.head(k)[risk_col].sum() * efficacy)
        scenario_rows.append({"intervention_capacity": k, "expected_preventable_readmissions": prevented})
    scen = pd.DataFrame(scenario_rows)
    fig = px.bar(scen, x="intervention_capacity", y="expected_preventable_readmissions", title="Expected Preventable Readmissions by Intervention Capacity")
    st.plotly_chart(fig, use_container_width=True)


def render_reason_tables(reason_df: pd.DataFrame, driver_df: pd.DataFrame) -> None:
    st.subheader("Readmission Reason Outputs")
    c1, c2 = st.columns(2)
    if not reason_df.empty:
        c1.dataframe(reason_df.head(20), use_container_width=True)
    else:
        c1.info("Reason summary CSV not available.")

    if not driver_df.empty:
        c2.dataframe(driver_df, use_container_width=True)
    else:
        c2.info("Driver summary CSV not available.")


def main() -> None:
    render_header()

    cfg_path = st.sidebar.text_input("Dashboard Config", str(DEFAULT_CONFIG_PATH))
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    try:
        cfg = load_dashboard_config(cfg_path)
    except FileNotFoundError:
        st.error(f"Config not found: {cfg_path}")
        return

    final_dataset_path = cfg.get("final_model_dataset_csv_path", "Data/final/readmission_model_dataset.csv")
    final_reason_path = cfg.get("final_reason_summary_csv_path", "Data/final/readmission_reason_summary.csv")
    final_driver_path = cfg.get("final_driver_summary_csv_path", "Data/final/readmission_driver_summary.csv")

    st.sidebar.write("Final Model Dataset:", final_dataset_path)
    st.sidebar.write("Reason Summary:", final_reason_path)

    final_df = load_csv(final_dataset_path, get_file_mtime_ns(final_dataset_path))
    reason_df = load_csv(final_reason_path, get_file_mtime_ns(final_reason_path))
    driver_df = load_csv(final_driver_path, get_file_mtime_ns(final_driver_path))

    if final_df.empty:
        render_empty_state()
        return

    prepared_df = _prepare_dataframe(final_df)
    filtered_df = apply_filters(prepared_df)

    if filtered_df.empty:
        st.warning("No rows match current filters. Adjust selections in the sidebar.")
        return

    render_executive_kpi_cards(filtered_df)
    render_readmission_trend(filtered_df)
    render_cohort_segmentation(filtered_df)
    render_clinical_risk_drivers(filtered_df)
    render_data_quality(filtered_df)
    scored_df = render_model_performance(filtered_df)
    render_operational_action_view(scored_df)
    render_reason_tables(reason_df, driver_df)


if __name__ == "__main__":
    main()
