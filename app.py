import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

from src.model.predict import predict_transaction


st.set_page_config(
    page_title="Fraud Risk Analytics",
    page_icon="🔐",
    layout="wide"
)


st.title("🔐 Real-Time Fraud Detection & Risk Analytics")

st.write(
    "Analyze a transaction and estimate its fraud probability, "
    "risk score, and risk level."
)

st.divider()


col1, col2 = st.columns(2)


with col1:

    st.subheader("Transaction Details")

    amount = st.number_input(
        "Transaction Amount",
        min_value=50.0,
        value=500.0,
        step=50.0
    )

    hour = st.slider(
        "Transaction Hour",
        min_value=0,
        max_value=23,
        value=12
    )

    device_type = st.selectbox(
        "Device Type",
        [
            "Mobile",
            "Desktop",
            "Tablet"
        ]
    )

    location = st.selectbox(
        "Location",
        [
            "Delhi",
            "Mumbai",
            "Bangalore",
            "Hyderabad",
            "Chennai",
            "Pune",
            "Kolkata"
        ]
    )

    account_age_days = st.number_input(
        "Account Age (Days)",
        min_value=1,
        value=500,
        step=10
    )

    transaction_frequency = st.number_input(
        "Transaction Frequency",
        min_value=1,
        value=5,
        step=1
    )


with col2:

    st.subheader("Risk Indicators")

    failed_attempts = st.number_input(
        "Failed Attempts",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    distance = st.number_input(
        "Distance From Previous Transaction",
        min_value=0.0,
        value=10.0,
        step=5.0
    )

    previous_amount = st.number_input(
        "Previous Transaction Amount",
        min_value=50.0,
        value=500.0,
        step=50.0
    )

    new_device = st.selectbox(
        "New Device?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    location_changed = st.selectbox(
        "Location Changed?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    night_transaction = int(
        hour <= 5 or hour >= 23
    )

    amount_ratio = (
        amount / (previous_amount + 1)
    )


st.divider()


if st.button(
    "Analyze Transaction",
    use_container_width=True
):

    transaction = {
        "amount": amount,
        "hour": hour,
        "device_type": device_type,
        "location": location,
        "account_age_days": account_age_days,
        "transaction_frequency": transaction_frequency,
        "failed_attempts": failed_attempts,
        "distance_from_last_transaction": distance,
        "previous_transaction_amount": previous_amount,
        "new_device": new_device,
        "location_changed": location_changed,
        "night_transaction": night_transaction,
        "amount_ratio": amount_ratio
    }

    try:

        result = predict_transaction(transaction)

        st.divider()

        st.subheader("Prediction Results")

        result_col1, result_col2, result_col3 = st.columns(3)

        result_col1.metric(
            "Fraud Probability",
            f"{result['fraud_probability'] * 100:.2f}%"
        )

        result_col2.metric(
            "Risk Score",
            f"{result['risk_score']}/100"
        )

        result_col3.metric(
            "Risk Level",
            result["risk_level"]
        )

        st.divider()

        if result["risk_level"] == "HIGH":

            st.error(
                "🚨 High-risk transaction detected. "
                "This transaction requires further investigation."
            )

        elif result["risk_level"] == "MEDIUM":

            st.warning(
                "⚠️ Medium-risk transaction detected. "
                "Additional verification may be required."
            )

        else:

            st.success(
                "✅ Low-risk transaction. "
                "The transaction appears to be normal."
            )

        st.subheader("Transaction Analysis")

        analysis_col1, analysis_col2 = st.columns(2)

        with analysis_col1:

            st.write(
                f"**Transaction Amount:** ₹{amount:,.2f}"
            )

            st.write(
                f"**Transaction Hour:** {hour}:00"
            )

            st.write(
                f"**Device:** {device_type}"
            )

            st.write(
                f"**Location:** {location}"
            )

        with analysis_col2:

            st.write(
                f"**Failed Attempts:** {failed_attempts}"
            )

            st.write(
                f"**Distance:** {distance:.2f}"
            )

            st.write(
                f"**New Device:** {'Yes' if new_device else 'No'}"
            )

            st.write(
                f"**Location Changed:** "
                f"{'Yes' if location_changed else 'No'}"
            )

    except Exception as e:

        st.error(
            "Prediction failed. Please check that the trained "
            "model files exist in the ml_assets folder."
        )

        st.exception(e)


st.divider()

st.caption(
    "Real-Time Fraud Detection & Risk Analytics System"
)