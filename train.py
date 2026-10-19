import os, joblib
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

rng=np.random.default_rng(21)
dates=pd.date_range("2024-01-01",periods=520,freq="D")
rows=[]
for sku in range(1,6):
    base=rng.uniform(50,150)
    for i,d in enumerate(dates):
        season=15*np.sin(2*np.pi*i/7)+10*np.sin(2*np.pi*i/30)
        trend=.08*i
        promo=1.25 if d.day in [1,15] else 1.0
        demand=max(0,base+season+trend)*promo+rng.normal(0,8)
        rows.append([d,sku,max(0,demand)])
df=pd.DataFrame(rows,columns=["date","sku","sales"]).sort_values(["sku","date"])

g=df.groupby("sku")["sales"]
for lag in [1,7,14,28]:
    df[f"lag_{lag}"]=g.shift(lag)
df["rolling_7"]=g.transform(lambda s:s.shift(1).rolling(7).mean())
df["rolling_28"]=g.transform(lambda s:s.shift(1).rolling(28).mean())
df["dow"]=df.date.dt.dayofweek
df["month"]=df.date.dt.month
df=df.dropna()

cutoff=df["date"].max()-pd.Timedelta(days=60)
train=df[df.date<cutoff]
test=df[df.date>=cutoff]
features=["sku","lag_1","lag_7","lag_14","lag_28","rolling_7","rolling_28","dow","month"]
model=XGBRegressor(n_estimators=400,max_depth=6,learning_rate=.05,
                   subsample=.85,colsample_bytree=.9,objective="reg:squarederror",random_state=42)
model.fit(train[features],train.sales)
pred=np.maximum(0,model.predict(test[features]))
mae=mean_absolute_error(test.sales,pred)
rmse=mean_squared_error(test.sales,pred)**0.5
wmape=np.abs(test.sales-pred).sum()/np.abs(test.sales).sum()
print(f"MAE={mae:.2f} RMSE={rmse:.2f} WMAPE={wmape:.2%}")

os.makedirs("artifacts",exist_ok=True)
joblib.dump({"model":model,"features":features}, "artifacts/demand_model.joblib")
df.to_csv("artifacts/sales_history.csv",index=False)
print("Saved artifacts/demand_model.joblib and sales_history.csv")
