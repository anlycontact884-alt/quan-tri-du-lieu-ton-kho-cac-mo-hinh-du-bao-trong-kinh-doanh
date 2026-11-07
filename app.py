import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from io import BytesIO

# ============================================================
# INVENTORY DECISION SUPPORT DASHBOARD
# Reads the real outputs exported from the research notebook.
# Test-set forecasts are displayed as evaluation results only;
# they are NOT presented as future forecasts.
# ============================================================

st.set_page_config(
    page_title="Inventory Decision Support",
    page_icon="📦",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"

st.title("📦 Inventory Decision Support System")
st.caption(
    "Demand Forecasting | Inventory Monitoring | Replenishment | "
    "Agent Decision Support"
)


# ------------------------- DATA LOADERS -------------------------

@st.cache_data
def load_data(blob=None):
    path = DATA_DIR / "sales_data.csv"
    if blob is not None:
        data = pd.read_csv(BytesIO(blob))
    elif path.exists():
        data = pd.read_csv(path)
    else:
        return None

    data.columns = data.columns.astype(str).str.strip()
    required = ["Date", "Store ID", "Product ID", "Demand"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("Thiếu cột bắt buộc: " + ", ".join(missing))

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for c in ["Demand", "Inventory Level", "Units Sold", "Units Ordered"]:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")

    data = data.dropna(subset=required).copy()
    data["Store ID"] = data["Store ID"].astype(str)
    data["Product ID"] = data["Product ID"].astype(str)
    data["Demand"] = data["Demand"].clip(lower=0)
    return data.sort_values(["Store ID", "Product ID", "Date"])


# Các file kết quả có thể được tải trực tiếp trên giao diện.
# Không bắt buộc phải tạo hoặc upload thư mục artifacts lên GitHub.
UPLOADED_ARTIFACTS = {}


def read_csv_artifact(filename):
    try:
        if filename in UPLOADED_ARTIFACTS:
            return pd.read_csv(BytesIO(UPLOADED_ARTIFACTS[filename]))
        path = ARTIFACT_DIR / filename
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()
    except Exception as exc:
        st.warning(f"Không đọc được {filename}: {exc}")
        return pd.DataFrame()


def read_json_artifact(filename):
    import json
    try:
        if filename in UPLOADED_ARTIFACTS:
            return json.loads(UPLOADED_ARTIFACTS[filename].decode("utf-8"))
        path = ARTIFACT_DIR / filename
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    except Exception as exc:
        st.warning(f"Không đọc được {filename}: {exc}")
        return {}


def find_column(df, candidates):
    """Return first existing column, ignoring case and spaces."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def calc_rop_snapshot(data, lead, service):
    """
    Dashboard-only snapshot calculation.
    This is not a replacement for the notebook's pre-test calibrated
    inventory simulation. Missing inventory stays missing.
    """
    z = data.sort_values(["Store ID", "Product ID", "Date"]).copy()
    z["LTD"] = z.groupby(["Store ID", "Product ID"])["Demand"].transform(
        lambda s: s.rolling(int(lead), min_periods=int(lead)).sum()
    )
    rop = (
        z.dropna(subset=["LTD"])
        .groupby(["Store ID", "Product ID"])["LTD"]
        .quantile(service)
        .rename("Dashboard ROP")
        .reset_index()
    )
    avg = (
        z.groupby(["Store ID", "Product ID"])
        .tail(28)
        .groupby(["Store ID", "Product ID"])["Demand"]
        .mean()
        .rename("Avg Daily Demand")
        .reset_index()
    )
    latest = z.groupby(["Store ID", "Product ID"], as_index=False).tail(1)
    cols = ["Store ID", "Product ID", "Date"]
    snap = latest[cols].rename(columns={"Date": "Snapshot Date"})

    if "Inventory Level" in latest.columns:
        snap["Inventory Position"] = pd.to_numeric(
            latest["Inventory Level"], errors="coerce"
        ).to_numpy()
    else:
        snap["Inventory Position"] = np.nan

    out = snap.merge(rop, how="left").merge(avg, how="left")
    out["Dashboard ROP"] = out["Dashboard ROP"].fillna(
        out["Avg Daily Demand"] * lead
    )
    out["Dashboard ROP"] = out["Dashboard ROP"].fillna(0).round().astype(int)
    out["Target"] = out["Dashboard ROP"] + np.ceil(
        out["Avg Daily Demand"].fillna(0) * lead
    )
    out["Suggested Qty"] = np.where(
        out["Inventory Position"].notna(),
        np.maximum(out["Target"] - out["Inventory Position"], 0),
        np.nan
    )
    out["Status"] = np.where(
        out["Inventory Position"].isna(),
        "MISSING_INVENTORY_DATA",
        np.where(
            out["Inventory Position"] <= out["Dashboard ROP"],
            "PENDING_HUMAN_APPROVAL",
            "MONITOR"
        )
    )
    return out


def normalize_forecast_columns(data):
    """Map exported notebook columns to a consistent dashboard schema."""
    if data.empty:
        return data

    df = data.copy()
    rename_map = {}
    aliases = {
        "Actual Demand": ["Actual Demand", "Demand", "actual", "y_true"],
        "RF Prediction": [
            "RF Prediction", "rf_pred", "Random Forest Prediction",
            "Random Forest Global", "prediction"
        ],
        "P10": ["P10", "p10", "q10"],
        "P50": ["P50", "p50", "q50"],
        "P90": ["P90", "p90", "q90"],
    }
    for target, options in aliases.items():
        col = find_column(df, options)
        if col is not None and col != target:
            rename_map[col] = target
    df = df.rename(columns=rename_map)

    date_col = find_column(df, ["Date", "date"])
    if date_col:
        df["Date"] = pd.to_datetime(df[date_col], errors="coerce")

    for col in ["Store ID", "Product ID"]:
        actual = find_column(df, [col, col.lower()])
        if actual:
            df[col] = df[actual].astype(str)

    return df


def show_metric_table(data, title):
    st.markdown(f"**{title}**")
    st.dataframe(data, use_container_width=True)


# ------------------------- SIDEBAR / DATA -------------------------

st.sidebar.header("Cấu hình dashboard")

st.sidebar.markdown("**1. Dữ liệu bán hàng**")
uploaded = st.sidebar.file_uploader(
    "Tải sales_data.csv",
    type=["csv"],
    key="sales_data_upload",
    help="File dữ liệu gốc gồm Date, Store ID, Product ID và Demand."
)

st.sidebar.markdown("**2. Kết quả xuất từ Notebook**")
uploaded_artifact_files = st.sidebar.file_uploader(
    "Tải các file kết quả (chọn nhiều file cùng lúc)",
    type=["csv", "json"],
    accept_multiple_files=True,
    key="artifact_uploads",
    help=(
        "Có thể chọn nhiều file CSV/JSON cùng lúc. "
        "Không cần tải thư mục artifacts hoặc đưa các file này lên GitHub."
    )
)

UPLOADED_ARTIFACTS = {
    file.name: file.getvalue()
    for file in (uploaded_artifact_files or [])
}

with st.sidebar.expander("Các file kết quả nên tải lên", expanded=False):
    st.markdown(
        "- `forecast_test.csv` – dự báo trên tập Test\n"
        "- `policy_comparison.csv` – tổng hợp so sánh chính sách tồn kho\n"
        "- `inventory_simulation_detail.csv` – chi tiết mô phỏng\n"
        "- `model_metrics.csv` – chỉ số đánh giá mô hình\n"
        "- `inventory_recommendations.csv` – đề xuất tồn kho\n"
        "- `agent_recommendations.csv` – kết quả kiểm thử Agent\n"
        "- `research_summary.json` – thông tin thực nghiệm (nếu có)"
    )

try:
    df = load_data(uploaded.getvalue() if uploaded else None)
except Exception as exc:
    st.error(str(exc))
    st.stop()

if df is None:
    st.info("Hãy tải file sales_data.csv tại mục Dữ liệu bán hàng ở thanh bên trái.")
    st.write("Cột bắt buộc: Date, Store ID, Product ID, Demand.")
    st.stop()

forecast_test_all = normalize_forecast_columns(
    read_csv_artifact("forecast_test.csv")
)
policy_all = read_csv_artifact("policy_comparison.csv")
metrics_all = read_csv_artifact("model_metrics.csv")
inventory_all = read_csv_artifact("inventory_recommendations.csv")
agent_all = read_csv_artifact("agent_recommendations.csv")
simulation_all = read_csv_artifact("inventory_simulation_detail.csv")
summary = read_json_artifact("research_summary.json")

dmin, dmax = df["Date"].min().date(), df["Date"].max().date()
period = st.sidebar.date_input(
    "Khoảng thời gian dữ liệu",
    (dmin, dmax),
    min_value=dmin,
    max_value=dmax
)
if not isinstance(period, (tuple, list)) or len(period) != 2:
    period = (dmin, dmax)

stores = sorted(df["Store ID"].dropna().unique().tolist())
products = sorted(df["Product ID"].dropna().unique().tolist())
sel_store = st.sidebar.multiselect("Cửa hàng", stores, default=stores)
sel_product = st.sidebar.multiselect("Sản phẩm", products, default=products)
lead = st.sidebar.number_input("Lead time giả định (ngày)", 1, 60, 7)
service = st.sidebar.slider("Service level / quantile", 0.80, 0.99, 0.95, 0.01)

filtered_history = df[
    (df["Date"].dt.date >= period[0])
    & (df["Date"].dt.date <= period[1])
    & df["Store ID"].isin(sel_store)
    & df["Product ID"].isin(sel_product)
].copy()

if filtered_history.empty:
    st.warning("Không có dữ liệu phù hợp với bộ lọc.")
    st.stop()

selected_history = df[
    df["Store ID"].isin(sel_store)
    & df["Product ID"].isin(sel_product)
].copy()

snapshot = calc_rop_snapshot(selected_history, lead, service)

if not forecast_test_all.empty:
    forecast_test = forecast_test_all.copy()
    if "Store ID" in forecast_test.columns:
        forecast_test = forecast_test[
            forecast_test["Store ID"].astype(str).isin(sel_store)
        ]
    if "Product ID" in forecast_test.columns:
        forecast_test = forecast_test[
            forecast_test["Product ID"].astype(str).isin(sel_product)
        ]
    if "Date" in forecast_test.columns:
        forecast_test = forecast_test[
            forecast_test["Date"].dt.date.between(period[0], period[1])
        ]
else:
    forecast_test = pd.DataFrame()

daily = filtered_history.groupby("Date", as_index=False)["Demand"].sum()
by_store = filtered_history.groupby("Store ID", as_index=False)["Demand"].sum()
by_product = (
    filtered_history.groupby("Product ID", as_index=False)["Demand"]
    .sum()
    .sort_values("Demand", ascending=False)
)

pages = [
    "Executive Overview",
    "Demand Forecasting",
    "Inventory Monitoring",
    "Replenishment Decision",
    "Agentic AI Decision Center",
    "Inventory Policy Evaluation",
    "Model & System Monitoring"
]
page = st.sidebar.radio("Điều hướng", pages)


# ------------------------- PAGE 1 -------------------------

if page == "Executive Overview":
    st.subheader("Executive Overview")
    a, b, c, d = st.columns(4)
    a.metric("Total Demand", f"{filtered_history.Demand.sum():,.0f}")
    b.metric("Average Daily Demand", f"{filtered_history.Demand.mean():,.1f}")
    c.metric("Store–Product pairs", f"{len(snapshot):,}")
    review_count = int((snapshot["Status"] == "PENDING_HUMAN_APPROVAL").sum())
    d.metric("Need human review", f"{review_count:,}")

    x, y = st.columns(2)
    x.plotly_chart(
        px.line(daily, x="Date", y="Demand", title="Nhu cầu thực tế theo ngày"),
        use_container_width=True
    )
    y.plotly_chart(
        px.bar(by_store, x="Store ID", y="Demand", title="Nhu cầu theo cửa hàng"),
        use_container_width=True
    )
    x, y = st.columns(2)
    x.plotly_chart(
        px.bar(
            by_product.head(10), x="Demand", y="Product ID",
            orientation="h", title="Top 10 sản phẩm"
        ),
        use_container_width=True
    )
    y.plotly_chart(
        px.pie(by_store, names="Store ID", values="Demand", hole=0.45,
               title="Cơ cấu nhu cầu"),
        use_container_width=True
    )

    if summary:
        st.caption(
            f"Mô hình được chọn: {summary.get('selected_model', 'Chưa có thông tin')}. "
            f"Giai đoạn Test: {summary.get('test_start', 'N/A')} đến "
            f"{summary.get('test_end', 'N/A')}."
        )


# ------------------------- PAGE 2 -------------------------

elif page == "Demand Forecasting":
    st.subheader("Demand Forecasting – Kết quả đánh giá trên tập Test")
    st.caption(
        "Trang này hiển thị dự báo trên tập Test đã xuất từ Notebook. "
        "Đây là dữ liệu đánh giá mô hình, không phải dự báo cho các ngày tương lai."
    )

    if forecast_test.empty:
        st.warning(
            "Chưa tìm thấy dữ liệu dự báo. Hãy đặt forecast_test.csv vào thư mục artifacts/."
        )
    else:
        if not {"Store ID", "Product ID"}.issubset(forecast_test.columns):
            st.error(
                "forecast_test.csv cần có các cột Store ID và Product ID "
                "để lọc theo cửa hàng/sản phẩm."
            )
        else:
            stores_fc = sorted(forecast_test["Store ID"].dropna().unique())
            products_fc = sorted(forecast_test["Product ID"].dropna().unique())
            x, y = st.columns(2)
            store_fc = x.selectbox("Store ID", stores_fc, key="forecast_store")
            product_fc = y.selectbox("Product ID", products_fc, key="forecast_product")

            selected_fc = forecast_test[
                (forecast_test["Store ID"] == store_fc)
                & (forecast_test["Product ID"] == product_fc)
            ].sort_values("Date")

            fig = go.Figure()
            if "Actual Demand" in selected_fc.columns:
                fig.add_scatter(
                    x=selected_fc["Date"], y=selected_fc["Actual Demand"],
                    mode="lines", name="Actual Demand"
                )
            if "RF Prediction" in selected_fc.columns:
                fig.add_scatter(
                    x=selected_fc["Date"], y=selected_fc["RF Prediction"],
                    mode="lines", name="Random Forest Global"
                )
            if {"P10", "P90"}.issubset(selected_fc.columns):
                fig.add_trace(go.Scatter(
                    x=pd.concat([selected_fc["Date"], selected_fc["Date"][::-1]]),
                    y=pd.concat([selected_fc["P90"], selected_fc["P10"][::-1]]),
                    fill="toself", fillcolor="rgba(99,110,250,0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="P10–P90 interval"
                ))
            for qcol in ["P10", "P50", "P90"]:
                if qcol in selected_fc.columns:
                    fig.add_scatter(
                        x=selected_fc["Date"], y=selected_fc[qcol],
                        mode="lines", name=qcol
                    )
            fig.update_layout(
                title="Actual Demand vs Model Predictions",
                xaxis_title="Date", yaxis_title="Demand",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(selected_fc, use_container_width=True)
            st.download_button(
                "Tải dữ liệu dự báo đang xem",
                selected_fc.to_csv(index=False).encode("utf-8-sig"),
                "forecast_test_filtered.csv",
                "text/csv"
            )

    st.divider()
    st.markdown("**Nhu cầu thực tế theo cửa hàng và sản phẩm trong khoảng thời gian đã chọn**")
    x, y = st.columns(2)
    x.plotly_chart(
        px.bar(by_store, x="Store ID", y="Demand", title="Demand by Store"),
        use_container_width=True
    )
    y.plotly_chart(
        px.bar(
            by_product.head(15), x="Demand", y="Product ID",
            orientation="h", title="Top Products by Demand"
        ),
        use_container_width=True
    )


# ------------------------- PAGE 3 -------------------------

elif page == "Inventory Monitoring":
    st.subheader("Inventory Monitoring")
    known_inventory = snapshot["Inventory Position"].notna()
    a, b, c = st.columns(3)
    a.metric("Pairs có dữ liệu tồn kho", f"{known_inventory.sum():,}")
    b.metric("Mean ROP", f"{snapshot['Dashboard ROP'].mean():,.1f}")
    mean_inventory = snapshot.loc[known_inventory, "Inventory Position"].mean()
    c.metric(
        "Mean Inventory Position",
        f"{mean_inventory:,.1f}" if pd.notna(mean_inventory) else "N/A"
    )

    if not known_inventory.any():
        st.warning(
            "Không có Inventory Level hợp lệ. Dashboard không tự xem giá trị thiếu "
            "là tồn kho bằng 0."
        )
    else:
        view = snapshot[known_inventory].copy()
        top = view.sort_values("Suggested Qty", ascending=False).head(20)
        fig = go.Figure()
        fig.add_bar(
            x=top["Product ID"], y=top["Inventory Position"],
            name="Inventory Position"
        )
        fig.add_bar(x=top["Product ID"], y=top["Dashboard ROP"], name="Dashboard ROP")
        fig.update_layout(barmode="group", title="Inventory Position vs ROP")
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(
            px.histogram(view, x="Inventory Position",
                         title="Phân phối Inventory Position"),
            use_container_width=True
        )
    st.caption(
        "Dashboard ROP ở trang này được tính từ lịch sử dữ liệu đã nạp và tham số "
        "đang chọn. Nó không thay thế ROP đã hiệu chỉnh trước tập Test trong nghiên cứu."
    )
    st.dataframe(snapshot, use_container_width=True)


# ------------------------- PAGE 4 -------------------------

elif page == "Replenishment Decision":
    st.subheader("Replenishment Decision")
    needs = snapshot[
        (snapshot["Suggested Qty"].fillna(0) > 0)
        & snapshot["Inventory Position"].notna()
    ].copy()

    a, b, c = st.columns(3)
    a.metric("Pairs cần xem xét", f"{len(needs):,}")
    b.metric("Total Suggested Qty", f"{needs['Suggested Qty'].sum():,.0f}")
    max_qty = needs["Suggested Qty"].max() if len(needs) else 0
    c.metric("Maximum Suggested Qty", f"{max_qty:,.0f}")

    if needs.empty:
        st.info("Không có đề xuất bổ sung từ snapshot tồn kho hiện tại.")
    else:
        st.plotly_chart(
            px.bar(
                needs.sort_values("Suggested Qty", ascending=False).head(20),
                x="Suggested Qty", y="Product ID", color="Store ID",
                orientation="h", title="Top replenishment recommendations"
            ),
            use_container_width=True
        )
        st.dataframe(
            needs.sort_values("Suggested Qty", ascending=False),
            use_container_width=True
        )
        st.download_button(
            "Tải đề xuất CSV",
            needs.to_csv(index=False).encode("utf-8-sig"),
            "replenishment_recommendations.csv",
            "text/csv"
        )

    if not inventory_all.empty:
        st.divider()
        st.markdown("**Khuyến nghị tồn kho đã xuất từ Notebook**")
        st.dataframe(inventory_all, use_container_width=True)


# ------------------------- PAGE 5 -------------------------

elif page == "Agentic AI Decision Center":
    st.subheader("Agentic AI Decision Center")
    st.caption(
        "Nguyên mẫu Agent theo quy tắc: quan sát tồn kho/ROP, giải thích "
        "và chuyển khuyến nghị cho người quản lý."
    )

    if not agent_all.empty:
        st.markdown("**Kết quả kiểm thử Agent từ Notebook**")
        status_col = find_column(agent_all, ["Status", "status", "Decision"])
        if status_col:
            counts = (
                agent_all[status_col].astype(str)
                .value_counts()
                .rename_axis("Status")
                .reset_index(name="Cases")
            )
            a, b = st.columns(2)
            a.metric("Total test cases", f"{len(agent_all):,}")
            b.metric("Distinct statuses", f"{agent_all[status_col].nunique():,}")
            a.plotly_chart(
                px.pie(counts, names="Status", values="Cases", hole=0.45,
                       title="Agent test status"),
                use_container_width=True
            )
        st.dataframe(agent_all, use_container_width=True)
        st.download_button(
            "Tải kết quả kiểm thử Agent",
            agent_all.to_csv(index=False).encode("utf-8-sig"),
            "agent_recommendations.csv",
            "text/csv"
        )
    else:
        st.info(
            "Chưa có agent_recommendations.csv. Dashboard đang hiển thị "
            "khuyến nghị theo quy tắc từ snapshot dữ liệu bên dưới."
        )

    st.markdown("**Khuyến nghị theo snapshot tồn kho**")
    counts = (
        snapshot["Status"].value_counts()
        .rename_axis("Status").reset_index(name="Cases")
    )
    a, b, c = st.columns(3)
    a.metric("Total pairs", f"{len(snapshot):,}")
    b.metric(
        "Review reorder",
        f"{(snapshot['Status'] == 'PENDING_HUMAN_APPROVAL').sum():,}"
    )
    c.metric(
        "Monitor",
        f"{(snapshot['Status'] == 'MONITOR').sum():,}"
    )
    x, y = st.columns(2)
    x.plotly_chart(
        px.pie(counts, names="Status", values="Cases", hole=0.45,
               title="Snapshot decision status"),
        use_container_width=True
    )
    y.plotly_chart(
        px.bar(
            snapshot.groupby("Store ID", as_index=False)["Suggested Qty"].sum(min_count=1),
            x="Store ID", y="Suggested Qty", title="Suggested quantity by store"
        ),
        use_container_width=True
    )
    st.dataframe(snapshot, use_container_width=True)
    st.info(
        "Agent trong nghiên cứu là nguyên mẫu hỗ trợ quyết định. "
        "Chưa có chức năng lưu approve/reject hoặc gửi Purchase Order thực tế."
    )


# ------------------------- PAGE 6 -------------------------

elif page == "Inventory Policy Evaluation":
    st.subheader("Inventory Policy Evaluation")
    st.caption(
        "So sánh hai chính sách Baseline ROP và Quantile ROP trên tập Test "
        "(2023-10-13 đến 2024-01-30). Đây là kết quả mô phỏng, không phải "
        "số liệu vận hành thực tế."
    )

    if policy_all.empty:
        st.warning(
            "Chưa tìm thấy artifacts/policy_comparison.csv. Hãy chạy cell xuất "
            "kết quả mô phỏng ở cuối Notebook, sau đó chép file vào thư mục artifacts/."
        )
    else:
        sim = policy_all.copy()
        # Chuẩn hóa đúng tên cột được tạo từ sim_summary trong Notebook.
        column_aliases = {
            "Policy": ["Policy", "policy", "Strategy"],
            "Total Demand": ["Total_Demand", "Total Demand", "total_demand"],
            "Stockout Units": ["Stockout_Units", "Stockout Units"],
            "Stockout Rate": ["Mean_Stockout_Rate", "Stockout Rate", "Mean Stockout Rate"],
            "Fill Rate": ["Mean_Fill_Rate", "Fill Rate", "Mean Fill Rate"],
            "Mean Inventory": ["Mean_Inventory", "Mean Inventory"],
            "Orders": ["Orders", "Order Count"]
        }
        rename = {}
        for target, aliases in column_aliases.items():
            found = find_column(sim, aliases)
            if found is not None and found != target:
                rename[found] = target
        sim = sim.rename(columns=rename)

        if "Policy" not in sim.columns:
            st.error("File policy_comparison.csv thiếu cột Policy.")
        else:
            sim["Policy"] = sim["Policy"].astype(str).str.strip().str.lower().map(
                lambda x: "Baseline ROP" if "baseline" in x else (
                    "Quantile ROP" if "quantile" in x else x.title()
                )
            )
            for col in ["Total Demand", "Stockout Units", "Stockout Rate", "Fill Rate", "Mean Inventory", "Orders"]:
                if col in sim.columns:
                    sim[col] = pd.to_numeric(sim[col], errors="coerce")

            # Các chỉ số trong sim_summary là tỷ lệ 0-1; chỉ đổi sang % khi trình bày.
            baseline = sim[sim["Policy"] == "Baseline ROP"]
            quantile = sim[sim["Policy"] == "Quantile ROP"]

            if not baseline.empty and not quantile.empty:
                b = baseline.iloc[0]
                q = quantile.iloc[0]
                st.markdown("#### Tổng quan kết quả mô phỏng")
                c1, c2, c3, c4 = st.columns(4)
                if "Stockout Units" in sim.columns:
                    c1.metric(
                        "Stockout Units – Quantile",
                        f"{q['Stockout Units']:,.0f}",
                        delta=f"{(q['Stockout Units'] - b['Stockout Units']):,.0f} so với Baseline",
                        delta_color="inverse"
                    )
                if "Fill Rate" in sim.columns:
                    c2.metric(
                        "Fill Rate – Quantile",
                        f"{q['Fill Rate']:.2%}",
                        delta=f"{(q['Fill Rate'] - b['Fill Rate']):+.2%} so với Baseline"
                    )
                if "Mean Inventory" in sim.columns:
                    c3.metric(
                        "Mean Inventory – Quantile",
                        f"{q['Mean Inventory']:,.2f}",
                        delta=f"{(q['Mean Inventory'] - b['Mean Inventory']):+,.2f} so với Baseline",
                        delta_color="inverse"
                    )
                if "Orders" in sim.columns:
                    c4.metric(
                        "Orders – Quantile",
                        f"{q['Orders']:,.0f}",
                        delta=f"{(q['Orders'] - b['Orders']):+,.0f} so với Baseline",
                        delta_color="inverse"
                    )

                # Diễn giải trade-off, không quy đổi thành chi phí khi chưa có dữ liệu chi phí.
                if "Stockout Units" in sim.columns and b["Stockout Units"] > 0:
                    shortage_reduction = (b["Stockout Units"] - q["Stockout Units"]) / b["Stockout Units"]
                    st.write(
                        f"Số đơn vị thiếu hàng giảm **{shortage_reduction:.2%}** "
                        "so với Baseline ROP."
                    )
                if "Mean Inventory" in sim.columns and b["Mean Inventory"] > 0:
                    inventory_change = (q["Mean Inventory"] - b["Mean Inventory"]) / b["Mean Inventory"]
                    st.write(
                        f"Tồn kho bình quân thay đổi **{inventory_change:+.2%}** "
                        "so với Baseline ROP."
                    )

            st.divider()
            st.markdown("#### So sánh các chỉ tiêu giữa hai chính sách")
            chart_metrics = [
                c for c in ["Stockout Units", "Stockout Rate", "Fill Rate", "Mean Inventory", "Orders"]
                if c in sim.columns
            ]
            if chart_metrics:
                selected_metric = st.selectbox(
                    "Chọn chỉ tiêu", chart_metrics, key="policy_metric"
                )
                chart_data = sim[["Policy", selected_metric]].dropna().copy()
                if selected_metric in ["Stockout Rate", "Fill Rate"]:
                    chart_data[selected_metric] = chart_data[selected_metric] * 100
                    y_title = f"{selected_metric} (%)"
                else:
                    y_title = selected_metric
                fig = px.bar(
                    chart_data, x="Policy", y=selected_metric, color="Policy",
                    text_auto=".2f", title=f"So sánh {selected_metric}"
                )
                fig.update_layout(yaxis_title=y_title, xaxis_title="Inventory policy")
                st.plotly_chart(fig, use_container_width=True)

            display_sim = sim.copy()
            if "Stockout Rate" in display_sim.columns:
                display_sim["Stockout Rate (%)"] = display_sim["Stockout Rate"] * 100
            if "Fill Rate" in display_sim.columns:
                display_sim["Fill Rate (%)"] = display_sim["Fill Rate"] * 100
            st.markdown("#### Bảng kết quả tổng hợp")
            st.dataframe(display_sim, use_container_width=True)
            st.download_button(
                "Tải bảng so sánh chính sách",
                sim.to_csv(index=False).encode("utf-8-sig"),
                "inventory_policy_comparison.csv",
                "text/csv"
            )

    if not simulation_all.empty:
        st.divider()
        st.markdown("#### Chi tiết kết quả mô phỏng theo cửa hàng và sản phẩm")
        detail = simulation_all.copy()
        if "Policy" in detail.columns:
            policies = sorted(detail["Policy"].dropna().astype(str).unique())
            chosen_policy = st.multiselect(
                "Lọc chính sách", policies, default=policies, key="detail_policy"
            )
            detail = detail[detail["Policy"].astype(str).isin(chosen_policy)]
        if "Store ID" in detail.columns:
            available_stores = sorted(detail["Store ID"].dropna().astype(str).unique())
            chosen_stores = st.multiselect(
                "Lọc cửa hàng", available_stores, default=available_stores,
                key="detail_store"
            )
            detail = detail[detail["Store ID"].astype(str).isin(chosen_stores)]
        if not detail.empty:
            if {"Policy", "Stockout Units"}.issubset(detail.columns):
                st.plotly_chart(
                    px.bar(
                        detail, x="Store ID", y="Stockout Units", color="Policy",
                        barmode="group", title="Stockout Units theo cửa hàng và chính sách"
                    ),
                    use_container_width=True
                )
            st.dataframe(detail, use_container_width=True)
            st.download_button(
                "Tải chi tiết mô phỏng đã lọc",
                detail.to_csv(index=False).encode("utf-8-sig"),
                "inventory_simulation_detail_filtered.csv",
                "text/csv"
            )
        else:
            st.info("Không có dòng mô phỏng phù hợp với bộ lọc.")
    else:
        st.info(
            "Chưa có inventory_simulation_detail.csv. Bảng tổng hợp vẫn có thể "
            "hiển thị độc lập nếu policy_comparison.csv đã được xuất."
        )

    st.info(
        "Diễn giải: Quantile ROP được hiệu chỉnh bằng phân vị 95% của tổng nhu cầu "
        "trong lead time từ dữ liệu trước Test; Baseline ROP dựa trên nhu cầu trung "
        "bình gần đây. Mô phỏng giả định lead time 7 ngày và cùng mức tồn kho ban đầu. "
        "Kết quả cho thấy đánh đổi giữa giảm thiếu hàng và tăng tồn kho/số lần đặt hàng. "
        "Chưa có dữ liệu chi phí lưu kho, đặt hàng và thiếu hàng để kết luận chính sách "
        "nào tối ưu về chi phí."
    )


# ------------------------- PAGE 7 -------------------------

else:
    st.subheader("Model & System Monitoring")

    info = pd.DataFrame({
        "Metric": [
            "Rows", "Stores", "Products", "First date", "Last date",
            "Forecast test rows"
        ],
        "Value": [
            len(df),
            df["Store ID"].nunique(),
            df["Product ID"].nunique(),
            str(df["Date"].min().date()),
            str(df["Date"].max().date()),
            len(forecast_test_all)
        ]
    })
    st.dataframe(info, use_container_width=True)

    if summary:
        st.markdown("**Thông tin cấu hình thực nghiệm được xuất từ Notebook**")
        st.json(summary)

    if metrics_all.empty:
        st.info(
            "Chưa tìm thấy artifacts/model_metrics.csv. "
            "Hãy chạy cell xuất kết quả ở cuối Notebook."
        )
    else:
        st.markdown("**Model evaluation metrics**")
        st.dataframe(metrics_all, use_container_width=True)

        metric_name_col = find_column(metrics_all, ["Metric", "metric"])
        model_col = find_column(metrics_all, ["Model", "model"])
        value_col = find_column(metrics_all, ["Value", "value", "Score"])
        if metric_name_col and model_col and value_col:
            metric_names = sorted(metrics_all[metric_name_col].dropna().unique())
            selected_metric = st.selectbox(
                "Chọn metric để so sánh", metric_names
            )
            plot_data = metrics_all[
                metrics_all[metric_name_col] == selected_metric
            ]
            st.plotly_chart(
                px.bar(
                    plot_data, x=model_col, y=value_col,
                    color=model_col, title=f"{selected_metric} by model"
                ),
                use_container_width=True
            )

    st.plotly_chart(
        px.box(
            filtered_history, x="Store ID", y="Demand",
            title="Demand distribution by store"
        ),
        use_container_width=True
    )

    st.markdown("**Trạng thái file tích hợp**")
    artifact_status = pd.DataFrame({
        "Artifact": [
            "forecast_test.csv",
            "policy_comparison.csv",
            "model_metrics.csv",
            "inventory_recommendations.csv",
            "agent_recommendations.csv",
            "inventory_simulation_detail.csv",
            "research_summary.json"
        ],
        "Available": [
            (name in UPLOADED_ARTIFACTS) or (ARTIFACT_DIR / name).exists()
            for name in [
                "forecast_test.csv",
                "policy_comparison.csv",
                "model_metrics.csv",
                "inventory_recommendations.csv",
                "agent_recommendations.csv",
                "inventory_simulation_detail.csv",
                "research_summary.json"
            ]
        ]
    })
    st.dataframe(artifact_status, use_container_width=True)


# ------------------------- FOOTER -------------------------

st.divider()
st.caption(
    "Research prototype. Test-set predictions are for model evaluation, not future demand. "
    "Lead time and service level are assumptions. Dashboard snapshot calculations are "
    "separate from the notebook's pre-test inventory simulation. "
    "Notebook result files can be uploaded directly in the sidebar; GitHub folders are optional."
)
