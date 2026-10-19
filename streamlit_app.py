import joblib, pandas as pd, numpy as np, streamlit as st
st.set_page_config(page_title="Demand Forecasting",page_icon="📦")
st.title("📦 Supply Chain Demand Forecasting")
st.caption("XGBoost + lag/rolling features")

bundle=joblib.load("artifacts/demand_model.joblib")
model,features=bundle["model"],bundle["features"]
hist=pd.read_csv("artifacts/sales_history.csv",parse_dates=["date"])

sku=st.selectbox("SKU",sorted(hist.sku.unique()))
days=st.slider("Forecast horizon (days)",1,30,7)

s=hist[hist.sku==sku].sort_values("date").copy()
# Recursive forecast.
for _ in range(days):
    next_date=s.date.max()+pd.Timedelta(days=1)
    vals=s.sales.tolist()
    row=pd.DataFrame([{
        "sku":sku,"lag_1":vals[-1],"lag_7":vals[-7],"lag_14":vals[-14],"lag_28":vals[-28],
        "rolling_7":np.mean(vals[-7:]),"rolling_28":np.mean(vals[-28:]),
        "dow":next_date.dayofweek,"month":next_date.month
    }])
    yhat=max(0,float(model.predict(row[features])[0]))
    s=pd.concat([s,pd.DataFrame({"date":[next_date],"sku":[sku],"sales":[yhat]})],ignore_index=True)

future=s.tail(days)[["date","sales"]]
st.line_chart(s.tail(60).set_index("date")["sales"])
st.subheader("Forecast")
st.dataframe(future,hide_index=True)
st.metric("Total forecast demand",f"{future.sales.sum():,.0f} units")
