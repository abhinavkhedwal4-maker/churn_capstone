"""Streamlit dashboard for ConnectTel Customer Churn Radar."""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from api_client import ApiError, ChurnApiClient

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY") or None

ADDONS = {
    "online_security": "Online security",
    "online_backup": "Online backup",
    "device_protection": "Device protection",
    "tech_support": "Tech support",
    "streaming_tv": "Streaming TV",
    "streaming_movies": "Streaming movies",
}


def apply_theme() -> None:
    """Apply HackerRank-style terminal dark theme CSS with green accents."""
    st.markdown(
        """
        <style>
          /* Base typography and container spacing */
          html, body, [class*="css"] {
            font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', Courier, monospace;
          }
          .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1100px;
          }

          /* Hide default Streamlit chrome */
          #MainMenu { visibility: hidden; }
          footer { visibility: hidden; }
          header { visibility: hidden; }
          .stDeployButton { display: none; }

          /* Rounded terminal cards */
          .terminal-card {
            background-color: #161B22;
            border: 1px solid #30363D;
            border-radius: 8px;
            padding: 18px 22px;
            margin-bottom: 1.25rem;
          }

          /* Metric cards */
          div[data-testid="stMetric"] {
            background-color: #161B22 !important;
            border: 1px solid #30363D !important;
            border-radius: 8px !important;
            padding: 14px 18px !important;
          }
          div[data-testid="stMetric"] label {
            color: #8B949E !important;
            font-size: 0.85rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
          }
          div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #E6EDF3 !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 1.7rem !important;
          }

          /* Green accent buttons with subtle hover glow */
          div.stButton > button,
          div.stButton > button[kind="primary"] {
            background-color: #1BA94C !important;
            color: #0E1116 !important;
            font-weight: 700 !important;
            border: 1px solid #1BA94C !important;
            border-radius: 6px !important;
            padding: 0.5rem 1rem !important;
            font-family: 'JetBrains Mono', monospace !important;
            transition: all 0.2s ease-in-out !important;
          }
          div.stButton > button:hover {
            background-color: #39D353 !important;
            border-color: #39D353 !important;
            box-shadow: 0 0 14px rgba(57, 211, 83, 0.45) !important;
            color: #0E1116 !important;
          }

          /* Green active-tab underline */
          button[data-baseweb="tab"] {
            font-family: 'JetBrains Mono', monospace !important;
            color: #8B949E !important;
            border-bottom: 2px solid transparent !important;
            padding-bottom: 8px !important;
          }
          button[data-baseweb="tab"][aria-selected="true"] {
            color: #39D353 !important;
            border-bottom: 2px solid #39D353 !important;
            font-weight: 700 !important;
          }

          /* Risk badges */
          .risk-badge {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
          }
          .risk-low {
            background-color: rgba(57, 211, 83, 0.15);
            color: #39D353;
            border: 1px solid #39D353;
          }
          .risk-medium {
            background-color: rgba(234, 179, 8, 0.15);
            color: #F0883E;
            border: 1px solid #F0883E;
          }
          .risk-high {
            background-color: rgba(248, 81, 73, 0.15);
            color: #F85149;
            border: 1px solid #F85149;
          }

          /* Override Streamlit info / alert boxes into dark terminal cards */
          div[data-testid="stAlert"] {
            background-color: #161B22 !important;
            border: 1px solid #30363D !important;
            border-left: 3px solid #1BA94C !important;
            color: #E6EDF3 !important;
            border-radius: 6px !important;
          }

          /* Expander containers in sidebar */
          div[data-testid="stExpander"] {
            background-color: #161B22 !important;
            border: 1px solid #30363D !important;
            border-radius: 6px !important;
            margin-bottom: 0.75rem !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_client() -> ChurnApiClient:
    return ChurnApiClient(API_URL, API_KEY)


def api_is_up() -> bool:
    try:
        return get_client().health()
    except Exception:
        return False


def load_model_info() -> dict:
    return get_client().model_info()


def yes_no(val: bool) -> str:
    return "Yes" if val else "No"


def customer_form() -> tuple[dict | None, bool]:
    """Sidebar form with grouped sections, input validation, and sensible defaults."""
    with st.sidebar:
        st.markdown("### `> CONNECT_TEL // RADAR`")
        is_online = api_is_up()
        if is_online:
            st.caption("🟢 `API: ONLINE`")
        else:
            st.caption("🔴 `API: OFFLINE`")
        st.divider()

        st.subheader("Customer Profile")

        # Group 1: Account
        with st.expander("01 // Account", expanded=True):
            senior = st.toggle("Senior citizen", value=False)
            partner = st.toggle("Has partner", value=True)
            dependents = st.toggle("Has dependents", value=False)
            tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)

        # Group 2: Services
        with st.expander("02 // Services", expanded=True):
            has_phone = st.toggle("Phone service", value=True)
            multiple_lines = (
                yes_no(st.toggle("Multiple lines", value=False))
                if has_phone
                else "No phone service"
            )
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"], index=0)
            if internet == "No":
                addons = {name: "No internet service" for name in ADDONS}
                st.caption("Add-on services inactive (no internet).")
            else:
                addons = {
                    name: yes_no(
                        st.toggle(label, value=(name in {"online_backup", "streaming_tv"}))
                    )
                    for name, label in ADDONS.items()
                }

        # Group 3: Billing
        with st.expander("03 // Billing", expanded=True):
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=0)
            paperless = st.toggle("Paperless billing", value=True)
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                index=0,
            )
            monthly = st.number_input(
                "Monthly charges ($)",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                step=0.5,
                format="%.2f",
            )
            st.caption("Estimated total charges: tenure × monthly charges.")

        submitted = st.button("Predict churn risk", type="primary", use_container_width=True)

    if monthly < 0:
        st.sidebar.error("Monthly charges must be non-negative.")
        return None, False

    if tenure < 0:
        st.sidebar.error("Tenure must be non-negative.")
        return None, False

    total_est = round(max(0.0, float(tenure) * float(monthly)), 2)
    if tenure == 0 and total_est == 0.0:
        total_est = round(float(monthly), 2)

    payload = {
        "senior_citizen": int(senior),
        "partner": yes_no(partner),
        "dependents": yes_no(dependents),
        "tenure": int(tenure),
        "phone_service": yes_no(has_phone),
        "multiple_lines": multiple_lines,
        "internet_service": internet,
        **addons,
        "contract": contract,
        "paperless_billing": yes_no(paperless),
        "payment_method": payment,
        "monthly_charges": float(monthly),
        "total_charges": total_est,
    }
    return payload, submitted


def reasons_chart(reasons: list[dict]) -> go.Figure:
    """Horizontal bar chart showing top SHAP impacts in terminal green/red palette."""
    top_reasons = reasons[:5]
    df = pd.DataFrame(top_reasons).sort_values("impact", ascending=True)
    colours = ["#F85149" if val > 0 else "#39D353" for val in df["impact"]]

    fig = go.Figure(
        go.Bar(
            x=df["impact"],
            y=df["feature"],
            orientation="h",
            marker_color=colours,
            text=[f"{val:+.3f}" for val in df["impact"]],
            textposition="auto",
            textfont={"family": "monospace", "color": "#E6EDF3", "size": 11},
        )
    )
    fig.update_layout(
        height=260,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "monospace", "color": "#E6EDF3"},
        xaxis={
            "title": "Impact on churn log-odds (SHAP)",
            "gridcolor": "#30363D",
            "zerolinecolor": "#8B949E",
            "zerolinewidth": 1,
        },
        yaxis={"gridcolor": "#30363D"},
    )
    return fig


def render_prediction(result: dict) -> None:
    """Render the prediction result card and top SHAP reasons bar chart."""
    prob = result.get("churn_probability", 0.0)
    risk = result.get("risk_level", "Low")
    threshold = result.get("threshold", 0.5)
    reasons = result.get("top_reasons", [])

    badge_class = f"risk-{risk.lower()}"
    prob_str = f"{prob * 100:.1f}%"
    action_text = result.get("recommended_action", "")
    outcome_str = "Likely to Churn" if result.get("will_churn") else "Likely to Stay"

    st.markdown(
        f"""
        <div class="terminal-card">
          <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
              <div style="color:#8B949E; font-size:0.8rem; text-transform:uppercase;">
                Predicted Churn Probability
              </div>
              <div style="font-size:3.2rem; font-weight:800; color:#E6EDF3; line-height:1.1;">
                {prob_str}
              </div>
            </div>
            <div>
              <span class="risk-badge {badge_class}">{risk} Risk</span>
            </div>
          </div>
          <div style="color:#8B949E; font-size:0.85rem; border-top:1px solid #30363D;
                      padding-top:0.6rem; margin-top:0.6rem;">
            Decision threshold: <strong style="color:#E6EDF3;">{threshold:.3f}</strong> | 
            Outcome: <strong style="color:#E6EDF3;">{outcome_str}</strong>
          </div>
          <div style="margin-top:0.4rem; color:#8B949E; font-size:0.85rem;">
            Recommended action: <span style="color:#E6EDF3;">{action_text}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Top Contributing Factors")
    if reasons:
        st.plotly_chart(reasons_chart(reasons), use_container_width=True)
        st.caption(
            "Green bars push toward staying (lower risk); red bars push toward churn (higher risk)."
        )
    else:
        st.caption("No factor breakdown available.")


def confusion_chart(matrix: dict) -> go.Figure:
    """Confusion matrix heatmap with dark green terminal palette."""
    z = [[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=["Predicted: Stay", "Predicted: Churn"],
            y=["Actual: Stay", "Actual: Churn"],
            text=z,
            texttemplate="<b>%{text}</b>",
            textfont={"color": "#E6EDF3", "family": "monospace", "size": 15},
            colorscale=[[0.0, "#161B22"], [0.4, "#1B4728"], [1.0, "#2EA043"]],
            showscale=False,
        )
    )
    fig.update_layout(
        height=280,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "monospace", "color": "#E6EDF3"},
        yaxis_autorange="reversed",
    )
    return fig


def render_performance() -> None:
    """Render 4 metric cards, model comparison table, and confusion matrix."""
    try:
        info = load_model_info()
    except ApiError as error:
        st.error(f"Model details unavailable: {error}")
        return
    except Exception as error:
        st.error(f"Could not load model details: {error}")
        return

    if not isinstance(info, dict) or "metrics" not in info or "baseline_metrics" not in info:
        st.error("Received incomplete model performance data from the API.")
        return

    metrics = info["metrics"]
    baseline = info["baseline_metrics"]

    # 4 Metric Cards on top
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    col2.metric("Precision", f"{metrics['precision']:.3f}")
    col3.metric("Recall", f"{metrics['recall']:.3f}")
    col4.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    # Model comparison table with best model highlighted in green
    st.markdown("### Model Comparison")
    comparison_df = pd.DataFrame(
        {
            "Metric": ["ROC-AUC", "Recall", "F1 Score", "Precision", "Accuracy"],
            "Baseline (Logistic Regression)": [
                baseline["roc_auc"],
                baseline["recall"],
                baseline["f1"],
                baseline["precision"],
                baseline["accuracy"],
            ],
            "Final (Gradient Boosting)": [
                metrics["roc_auc"],
                metrics["recall"],
                metrics["f1"],
                metrics["precision"],
                metrics["accuracy"],
            ],
        }
    )

    styled_table = comparison_df.style.format(
        {
            "Baseline (Logistic Regression)": "{:.3f}",
            "Final (Gradient Boosting)": "{:.3f}",
        }
    ).map(
        lambda _: "color: #39D353; font-weight: bold; background-color: rgba(57, 211, 83, 0.08);",
        subset=["Final (Gradient Boosting)"],
    )

    st.dataframe(styled_table, use_container_width=True, hide_index=True)
    st.caption(
        f"Model Version: {info.get('model_version', '1.0')} · "
        f"Optimal F1 Threshold: {info.get('threshold', 0.5):.3f} · "
        f"Training Samples: {info.get('train_rows', 0):,} · "
        f"Test Samples: {info.get('test_rows', 0):,}"
    )

    # Confusion matrix styled to match theme
    st.markdown("### Test Set Confusion Matrix")
    st.plotly_chart(confusion_chart(metrics["confusion_matrix"]), use_container_width=True)


def render_about() -> None:
    """Render short description, dataset info, tech stack, and methodology in concise sections."""
    st.markdown(
        """
        <div class="terminal-card">
          <h4 style="color:#39D353; margin-top:0;">01 // Project Overview</h4>
          <p style="color:#E6EDF3; margin-bottom:0;">
            Customer Churn Radar is an end-to-end machine learning service designed for ConnectTel.
            It evaluates individual customer retention risk and surfaces interpretable SHAP feature
            contributions so customer success teams can intervene before contract cancellations.
          </p>
        </div>

        <div class="terminal-card">
          <h4 style="color:#39D353; margin-top:0;">02 // Dataset & Features</h4>
          <p style="color:#E6EDF3; margin-bottom:0;">
            Trained on the IBM Telco Customer Churn dataset comprising 7,043 customer accounts
            with an ~26% historical churn rate. Input dimensions encompass account demographics,
            enrolled telecom services (phone, fiber optic, security add-ons), and billing parameters
            (contract length, payment method, monthly billing). Sensitive attributes like gender
            are strictly omitted.
          </p>
        </div>

        <div class="terminal-card">
          <h4 style="color:#39D353; margin-top:0;">03 // System Architecture & Tech Stack</h4>
          <p style="color:#E6EDF3; margin-bottom:0;">
            • <strong>Backend:</strong> FastAPI with Pydantic request validation, rate limiting,
            and API key authentication.<br>
            • <strong>Model Pipeline:</strong> Scikit-learn Pipeline with ColumnTransformer and
            a tuned GradientBoostingClassifier.<br>
            • <strong>Explainability:</strong> TreeSHAP local attribution generating per-feature
            log-odds impact.<br>
            • <strong>Frontend:</strong> Streamlit interactive dashboard with terminal theme
            and responsive Plotly visualisations.
          </p>
        </div>

        <div class="terminal-card">
          <h4 style="color:#39D353; margin-top:0;">04 // Threshold & Imbalance Handling</h4>
          <p style="color:#E6EDF3; margin-bottom:0;">
            Because churn is an imbalanced event (~26% positive class), the decision boundary
            is tuned to maximize F1 score on out-of-fold validation rather than using a default
            0.5 probability cut-off. This yields high recall on at-risk accounts while avoiding
            excessive false alarms.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Churn Radar | ConnectTel", page_icon="📡", layout="wide")
    apply_theme()

    st.title("Customer Churn Radar")
    st.caption("ConnectTel churn risk scoring and SHAP explainability console.")

    payload, submitted = customer_form()

    if submitted and payload:
        try:
            with st.spinner("Executing model inference..."):
                st.session_state["result"] = get_client().predict(payload)
        except ApiError as error:
            st.session_state.pop("result", None)
            st.error(f"Prediction error: {error}")
        except Exception as error:
            st.session_state.pop("result", None)
            st.error(f"Unexpected error communicating with API: {error}")

    tab_predict, tab_model, tab_about = st.tabs(["Prediction", "Model performance", "About"])

    with tab_predict:
        if "result" in st.session_state:
            render_prediction(st.session_state["result"])
        else:
            st.markdown(
                """
                <div class="terminal-card">
                  <span style="color:#39D353; font-weight:700;">$ SYSTEM READY</span>
                  <p style="margin-top:0.5rem; color:#8B949E; margin-bottom:0;">
                    Configure customer attributes in the sidebar and click
                    <strong style="color:#E6EDF3;">Predict churn risk</strong>
                    to score customer retention risk.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_model:
        render_performance()

    with tab_about:
        render_about()


if __name__ == "__main__":
    main()
