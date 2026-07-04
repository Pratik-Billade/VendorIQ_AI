
import streamlit as st
from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag

st.set_page_config(
    page_title="VendorIQ AI",
    page_icon="🚀",
    layout="wide",
)

st.markdown("""
<style>
.stApp{
    background:#F7FAFE;
}
.block-container{
    max-width:1400px;
    padding-top:2.8rem;
    padding-bottom:1rem;
}
.hero{
    background:linear-gradient(90deg,#0F4C81,#2563EB);
    border-radius:18px;
    padding:28px 34px;
    margin-bottom:22px;
}
.hero h1{
    color:white;
    margin:0;
    font-size:42px;
}
.hero p{
    color:#E7F0FB;
    margin-top:10px;
    font-size:17px;
    line-height:1.5;
}
.section-title{
    color:#0F4C81;
    font-weight:700;
}
.cardhead{
    background:#EDF5FF;
    padding:12px 16px;
    border-radius:10px;
    border-left:5px solid #2563EB;
    margin-bottom:15px;
}
.footer{
    text-align:center;
    color:#64748B;
    font-size:14px;
    margin-top:20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🚀 VendorIQ AI</h1>
<p><b>Intelligent Vendor Invoice Analytics Platform</b><br>
Forecast freight costs and identify high-risk vendor invoices using machine learning to support
smarter finance and procurement decisions.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="cardhead"><span class="section-title">🚚 Freight Cost Predictor</span><br><small>Estimate expected freight charges before invoice processing.</small></div>', unsafe_allow_html=True)

        with st.form("freight_form"):
            dollars = st.number_input(
                "Invoice Amount ($)",
                min_value=0.0,
                value=0.0,
                step=100.0
            )
            submit = st.form_submit_button(
                "Predict Freight Cost",
                use_container_width=True
            )

        if submit:
            prediction = predict_freight_cost(
                {"Dollars": [dollars]}
            )["Predicted_Freight"]

            st.success("Prediction completed successfully.")

            st.metric(
                "Estimated Freight Cost",
                f"${prediction[0]:,.2f}"
            )

with right:
    with st.container(border=True):
        st.markdown('<div class="cardhead"><span class="section-title">🛡 Invoice Risk Analyzer</span><br><small>Determine whether an invoice should be routed for manual approval.</small></div>', unsafe_allow_html=True)

        with st.form("risk_form"):

            c1, c2 = st.columns(2)

            with c1:
                invoice_quantity = st.number_input(
                    "Invoice Quantity",
                    min_value=0,
                    value=0
                )

                invoice_dollars = st.number_input(
                    "Invoice Amount ($)",
                    min_value=0.0,
                    value=0.0
                )

                freight = st.number_input(
                    "Freight Cost",
                    min_value=0.0,
                    value=0.0
                )

            with c2:
                total_item_quantity = st.number_input(
                    "Total Item Quantity",
                    min_value=0,
                    value=0
                )

                total_item_dollars = st.number_input(
                    "Total Item Value ($)",
                    min_value=0.0,
                    value=0.0
                )

            analyze = st.form_submit_button(
                "Analyze Invoice",
                use_container_width=True
            )

        if analyze:
            result = predict_invoice_flag({
                "invoice_quantity": [invoice_quantity],
                "invoice_dollars": [invoice_dollars],
                "Freight": [freight],
                "total_item_quantity": [total_item_quantity],
                "total_item_dollars": [total_item_dollars]
            })["Predicted_Flag"]

            if bool(result[0]):
                st.error("🚨 Manual Approval Recommended")
                st.markdown("""
**Recommended Actions**
- Verify invoice values.
- Review supporting documentation.
- Validate vendor transaction history.
- Complete manual approval before payment.
""")
            else:
                st.success("✅ Suitable for Standard Approval")
                st.markdown("""
**Recommended Action**

Proceed with the standard approval workflow.
""")

st.divider()
st.markdown(
    '<div class="footer"><b>VendorIQ AI</b> | Freight Cost Prediction & Invoice Risk Analysis</div>',
    unsafe_allow_html=True
)
