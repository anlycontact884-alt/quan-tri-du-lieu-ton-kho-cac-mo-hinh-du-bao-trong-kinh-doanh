import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from io import BytesIO

st.set_page_config(page_title="Inventory Decision Support", page_icon="📦", layout="wide")
st.title("📦 Inventory Decision Support System")
st.caption("Demand Forecasting | Inventory Monitoring | Replenishment | Agent Decision Support")

@st.cache_data
def load_data(blob=None):
    path=Path("data/sales_data.csv")
    data=pd.read_csv(BytesIO(blob)) if blob is not None else (pd.read_csv(path) if path.exists() else None)
    if data is None: return None
    data.columns=data.columns.astype(str).str.strip()
    required=["Date","Store ID","Product ID","Demand"]
    missing=[c for c in required if c not in data.columns]
    if missing: raise ValueError("Thiếu cột bắt buộc: "+", ".join(missing))
    data["Date"]=pd.to_datetime(data["Date"],errors="coerce")
    for c in ["Demand","Inventory Level","Units Sold","Units Ordered"]:
        if c in data: data[c]=pd.to_numeric(data[c],errors="coerce")
    data=data.dropna(subset=required).copy()
    data["Store ID"]=data["Store ID"].astype(str)
    data["Product ID"]=data["Product ID"].astype(str)
    data["Demand"]=data["Demand"].clip(lower=0)
    return data.sort_values(["Store ID","Product ID","Date"])

def calc_rop(data, lead, service):
    # ROP from empirical lead-time demand; history used for dashboard snapshot.
    z=data.sort_values(["Store ID","Product ID","Date"]).copy()
    z["LTD"]=z.groupby(["Store ID","Product ID"])["Demand"].transform(
        lambda s:s.rolling(lead,min_periods=lead).sum())
    rop=z.dropna(subset=["LTD"]).groupby(["Store ID","Product ID"])["LTD"].quantile(service).rename("ROP").reset_index()
    avg=z.groupby(["Store ID","Product ID"]).tail(28).groupby(["Store ID","Product ID"])["Demand"].mean().rename("Avg Daily Demand").reset_index()
    latest=z.groupby(["Store ID","Product ID"],as_index=False).tail(1)
    cols=["Store ID","Product ID","Date","Inventory Level"]
    if "Inventory Level" in latest.columns:
        snap=latest[cols].rename(columns={"Date":"Snapshot Date","Inventory Level":"Inventory Position"})
    else:
        snap=latest[["Store ID","Product ID","Date"]].rename(columns={"Date":"Snapshot Date"})
        snap["Inventory Position"]=np.nan
    out=snap.merge(rop,how="left").merge(avg,how="left")
    out["ROP"]=out["ROP"].fillna(out["Avg Daily Demand"]*lead).round().astype(int)
    out["Inventory Position"]=out["Inventory Position"].fillna(0)
    out["Target"]=out["ROP"]+np.ceil(out["Avg Daily Demand"]*lead)
    out["Suggested Qty"]=np.maximum(out["Target"]-out["Inventory Position"],0).round().astype(int)
    out["Status"]=np.where(out["Inventory Position"]<=out["ROP"],"PENDING_HUMAN_APPROVAL","MONITOR")
    out["Risk"]=np.where(out["Inventory Position"]<=out["ROP"],"Cần xem xét đặt hàng","Chưa chạm ROP")
    return out

def seasonal_naive(data,horizon):
    # Explicit baseline fallback; trained RF / quantile models are not silently replaced.
    result=[]
    for (s,p),g in data.groupby(["Store ID","Product ID"]):
        g=g.sort_values("Date"); y=g["Demand"].tail(28).to_numpy()
        if not len(y): continue
        for h in range(1,horizon+1):
            value=y[-7+((h-1)%7)] if len(y)>=7 else y.mean()
            result.append([s,p,g.Date.max()+pd.Timedelta(days=h),max(0,float(value))])
    return pd.DataFrame(result,columns=["Store ID","Product ID","Date","Forecast"])

st.sidebar.header("Cấu hình dashboard")
uploaded=st.sidebar.file_uploader("Tải sales_data.csv",type=["csv"])
try: df=load_data(uploaded.getvalue() if uploaded else None)
except Exception as e: st.error(str(e)); st.stop()
if df is None:
    st.info("Đặt sales_data.csv trong thư mục data/ hoặc tải file ở thanh bên.")
    st.write("Cột bắt buộc: Date, Store ID, Product ID, Demand.")
    st.stop()

dmin,dmax=df.Date.min().date(),df.Date.max().date()
period=st.sidebar.date_input("Khoảng thời gian",(dmin,dmax),min_value=dmin,max_value=dmax)
if not isinstance(period,tuple) or len(period)!=2: period=(dmin,dmax)
stores=sorted(df["Store ID"].unique().tolist()); products=sorted(df["Product ID"].unique().tolist())
sel_store=st.sidebar.multiselect("Cửa hàng",stores,default=stores)
sel_product=st.sidebar.multiselect("Sản phẩm",products,default=products)
lead=st.sidebar.number_input("Lead time giả định (ngày)",1,60,7)
service=st.sidebar.slider("Service level / quantile",0.80,0.99,0.95,0.01)
horizon=st.sidebar.slider("Horizon dự báo baseline",7,30,14,7)
f=df[(df.Date.dt.date>=period[0])&(df.Date.dt.date<=period[1])&
     df["Store ID"].isin(sel_store)&df["Product ID"].isin(sel_product)].copy()
if f.empty: st.warning("Không có dữ liệu phù hợp bộ lọc."); st.stop()

snap=calc_rop(df[df["Store ID"].isin(sel_store)&df["Product ID"].isin(sel_product)],lead,service)
forecast=seasonal_naive(df[df["Store ID"].isin(sel_store)&df["Product ID"].isin(sel_product)],horizon)
daily=f.groupby("Date",as_index=False).Demand.sum()
by_store=f.groupby("Store ID",as_index=False).Demand.sum()
by_product=f.groupby("Product ID",as_index=False).Demand.sum().sort_values("Demand",ascending=False)
pages=["Executive Overview","Demand Forecasting","Inventory Monitoring","Replenishment Decision","Agentic AI Decision Center","Inventory Policy Evaluation","Model & System Monitoring"]
page=st.sidebar.radio("Điều hướng",pages)

if page=="Executive Overview":
    st.subheader("Executive Overview")
    a,b,c,d=st.columns(4)
    a.metric("Total Demand",f"{f.Demand.sum():,.0f}")
    b.metric("Average Daily Demand",f"{f.Demand.mean():,.1f}")
    c.metric("Store–Product pairs",f"{len(snap):,}")
    c2=int((snap.Status=="PENDING_HUMAN_APPROVAL").sum())
    d.metric("Need human review",f"{c2:,}")
    x,y=st.columns(2)
    x.plotly_chart(px.line(daily,x="Date",y="Demand",title="Nhu cầu thực tế theo ngày"),use_container_width=True)
    y.plotly_chart(px.bar(by_store,x="Store ID",y="Demand",title="Nhu cầu theo cửa hàng"),use_container_width=True)
    x,y=st.columns(2)
    x.plotly_chart(px.bar(by_product.head(10),x="Demand",y="Product ID",orientation="h",title="Top 10 sản phẩm"),use_container_width=True)
    y.plotly_chart(px.pie(by_store,names="Store ID",values="Demand",hole=.45,title="Cơ cấu nhu cầu"),use_container_width=True)

elif page=="Demand Forecasting":
    st.subheader("Demand Forecasting")
    x,y=st.columns(2)
    s=x.selectbox("Store ID",sel_store); p=y.selectbox("Product ID",sel_product)
    hist=df[(df["Store ID"]==s)&(df["Product ID"]==p)].sort_values("Date").tail(90)
    pred=forecast[(forecast["Store ID"]==s)&(forecast["Product ID"]==p)]
    fig=go.Figure()
    fig.add_scatter(x=hist.Date,y=hist.Demand,mode="lines",name="Actual Demand")
    fig.add_scatter(x=pred.Date,y=pred.Forecast,mode="lines+markers",name="Seasonal Naive lag-7")
    fig.update_layout(title="Nhu cầu lịch sử và dự báo baseline",xaxis_title="Date",yaxis_title="Demand")
    st.plotly_chart(fig,use_container_width=True)
    st.plotly_chart(px.bar(by_store,x="Store ID",y="Demand",title="Demand by Store"),use_container_width=True)
    st.plotly_chart(px.bar(by_product.head(15),x="Demand",y="Product ID",orientation="h",title="Top Products by Demand"),use_container_width=True)
    st.dataframe(pred,use_container_width=True)
    st.warning("Dự báo hiển thị ở đây là Seasonal Naive lag-7 làm baseline chạy độc lập. Chưa phải Random Forest Global hay P10/P50/P90 đã huấn luyện trong notebook.")

elif page=="Inventory Monitoring":
    st.subheader("Inventory Monitoring")
    a,b,c=st.columns(3)
    a.metric("Below ROP",f"{(snap['Inventory Position']<=snap.ROP).sum():,}")
    b.metric("Mean ROP",f"{snap.ROP.mean():,.1f}")
    c.metric("Mean Inventory Position",f"{snap['Inventory Position'].mean():,.1f}")
    if "Inventory Level" not in df.columns: st.warning("Dataset không có Inventory Level; tồn kho sẽ không thể phản ánh thực tế.")
    top=snap.sort_values("Suggested Qty",ascending=False).head(20)
    fig=go.Figure()
    fig.add_bar(x=top["Product ID"],y=top["Inventory Position"],name="Inventory Position")
    fig.add_bar(x=top["Product ID"],y=top.ROP,name="ROP")
    fig.update_layout(barmode="group",title="Inventory Position vs ROP")
    st.plotly_chart(fig,use_container_width=True)
    st.plotly_chart(px.histogram(snap,x="Inventory Position",title="Phân phối Inventory Position"),use_container_width=True)
    st.dataframe(snap.sort_values("Suggested Qty",ascending=False),use_container_width=True)

elif page=="Replenishment Decision":
    st.subheader("Replenishment Decision")
    needs=snap[snap["Suggested Qty"]>0].copy()
    a,b,c=st.columns(3)
    a.metric("Pairs cần xem xét",f"{len(needs):,}")
    b.metric("Total Suggested Qty",f"{needs['Suggested Qty'].sum():,.0f}")
    c.metric("Maximum Suggested Qty",f"{needs['Suggested Qty'].max() if len(needs) else 0:,.0f}")
    st.plotly_chart(px.bar(needs.sort_values("Suggested Qty",ascending=False).head(20),x="Suggested Qty",y="Product ID",color="Store ID",orientation="h",title="Top replenishment recommendations"),use_container_width=True)
    st.plotly_chart(px.histogram(needs,x="Suggested Qty",title="Phân phối số lượng đề xuất"),use_container_width=True)
    st.dataframe(needs.sort_values("Suggested Qty",ascending=False),use_container_width=True)
    st.download_button("Tải đề xuất CSV",needs.to_csv(index=False).encode("utf-8-sig"),"replenishment_recommendations.csv","text/csv")

elif page=="Agentic AI Decision Center":
    st.subheader("Agentic AI Decision Center")
    st.caption("Nguyên mẫu Agent theo quy tắc: quan sát tồn kho/ROP, giải thích và chuyển khuyến nghị cho người quản lý.")
    counts=snap.Status.value_counts().rename_axis("Status").reset_index(name="Cases")
    a,b,c=st.columns(3)
    a.metric("Total cases",f"{len(snap):,}")
    b.metric("Review reorder",f"{(snap.Status=='PENDING_HUMAN_APPROVAL').sum():,}")
    c.metric("Monitor",f"{(snap.Status=='MONITOR').sum():,}")
    x,y=st.columns(2)
    x.plotly_chart(px.pie(counts,names="Status",values="Cases",hole=.45,title="Agent decision status"),use_container_width=True)
    y.plotly_chart(px.bar(snap.groupby("Store ID",as_index=False)["Suggested Qty"].sum(),x="Store ID",y="Suggested Qty",title="Suggested quantity by store"),use_container_width=True)
    if len(snap):
        ix=st.selectbox("Chọn trường hợp",snap.index,format_func=lambda i:f"{snap.loc[i,'Store ID']} – {snap.loc[i,'Product ID']}")
        r=snap.loc[ix]
        a,b,c=st.columns(3)
        a.metric("Inventory Position",f"{r['Inventory Position']:,.0f}")
        b.metric("ROP",f"{r.ROP:,.0f}")
        c.metric("Suggested Qty",f"{r['Suggested Qty']:,.0f}")
        if r.Status=="PENDING_HUMAN_APPROVAL":
            st.warning(f"Inventory Position ≤ ROP. Đề xuất xem xét bổ sung {r['Suggested Qty']:,.0f} đơn vị; cần người quản lý kiểm tra.")
        else: st.success("Chưa chạm ROP. Tiếp tục theo dõi.")
    st.dataframe(snap,use_container_width=True)
    st.info("Chưa có chức năng lưu approve/reject hoặc gửi PO. Đây là khuyến nghị có Human-in-the-Loop, không phải Agent tự đặt hàng.")

elif page=="Inventory Policy Evaluation":
    st.subheader("Inventory Policy Evaluation")
    p=Path("artifacts/policy_comparison.csv")
    if p.exists():
        sim=pd.read_csv(p); st.dataframe(sim,use_container_width=True)
        measures=[c for c in ["Stockout Rate","Fill Rate","Mean Inventory","Orders","Stockout Units"] if c in sim]
        if measures:
            m=st.selectbox("Chỉ tiêu so sánh",measures)
            st.plotly_chart(px.bar(sim,x="Policy",y=m,color="Policy",title=f"Comparison: {m}"),use_container_width=True)
    else:
        st.info("Thêm artifacts/policy_comparison.csv được xuất từ notebook để hiển thị kết quả mô phỏng chính xác.")
        st.code("Policy,Stockout Rate,Fill Rate,Mean Inventory,Orders,Stockout Units\\nBaseline ROP,5.77,94.23,485.44,1187,63345\\nQuantile ROP,0.095,99.91,720.28,1371,1060",language="csv")

else:
    st.subheader("Model & System Monitoring")
    q=pd.DataFrame({"Metric":["Rows","Stores","Products","First date","Last date"],"Value":[len(df),df["Store ID"].nunique(),df["Product ID"].nunique(),str(df.Date.min().date()),str(df.Date.max().date())]})
    st.dataframe(q,use_container_width=True)
    p=Path("artifacts/model_metrics.csv")
    if p.exists():
        m=pd.read_csv(p); st.dataframe(m,use_container_width=True)
        if {"Model","WAPE"}.issubset(m.columns):
            st.plotly_chart(px.bar(m,x="Model",y="WAPE",title="WAPE by model"),use_container_width=True)
    else: st.info("Thêm artifacts/model_metrics.csv xuất từ notebook. Không tự tạo metric mô hình.")
    st.plotly_chart(px.box(f,x="Store ID",y="Demand",title="Demand distribution by store"),use_container_width=True)

st.divider()
st.caption("Research prototype. Lead time và service level là giả định điều chỉnh được; kết quả phụ thuộc dữ liệu được nạp.")
