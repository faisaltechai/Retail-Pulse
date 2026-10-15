
"""RetailPulse - Revenue & Customer Intelligence.

Run from the project root:
    python -m streamlit run app/streamlit_app.py
"""

import html
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.predict_churn import predict_customer_churn


# ============================================================================
# 1. Paths and constants
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
TX_PATH = PROCESSED_DIR / "cleaned_data.csv"
SEG_PATH = PROCESSED_DIR / "customer_segments.csv"

TX_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "CustomerID", "Country", "Revenue",
]

SEG_COLUMNS = [
    "CustomerID", "Recency", "Frequency", "Monetary",
    "R_Score", "F_Score", "M_Score", "RFM_Score",
    "Segment", "Cluster", "Cluster_Name",
]

CURRENCY = "£"
CHURN_RECENCY_DAYS = 90

BG = "#0B1411"
CARD = "#14251F"
BORDER = "#294A3D"
ACCENT = "#55C98B"
MINT = "#7DE0A3"
TEXT = "#F1F5F3"
MUTED = "#A7C4B5"
GRID = "#1E382E"
AMBER = "#E0B25A"


# ============================================================================
# 2. Logo and CSS
# ============================================================================

LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" '
    'width="190" height="36" viewBox="0 0 190 36">'
    f'<circle cx="18" cy="18" r="14" fill="none" '
    f'stroke="{ACCENT}" stroke-width="2.5"/>'
    f'<polyline points="9,19 14,19 17,11 21,25 24,18 28,18" '
    f'fill="none" stroke="{ACCENT}" stroke-width="2.5" '
    'stroke-linecap="round" stroke-linejoin="round"/>'
    '<text x="42" y="24" font-family="Inter,Segoe UI,Arial,sans-serif" '
    'font-size="19" font-weight="600" '
    f'fill="{TEXT}">RetailPulse</text></svg>'
)

CSS = """
<style>
[data-testid="stMainBlockContainer"] {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1320px;
}
[data-testid="stHeader"] { background: transparent; }
.stApp { background: __BG__; }
[data-testid="stSidebar"] { border-right: 1px solid __BORDER__; }

[class*="st-key-card"] {
    background: __CARD__;
    border: 1px solid __BORDER__;
    border-radius: 14px;
    padding: 1.15rem 1.25rem;
}

.rp-eyebrow {
    color: __ACCENT__;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-bottom: .3rem;
}
.rp-title {
    font-size: 1.65rem;
    font-weight: 650;
    letter-spacing: -.01em;
    color: __TEXT__;
    margin: 0;
}
.rp-desc {
    color: __MUTED__;
    font-size: .95rem;
    margin: .3rem 0 1.4rem;
    max-width: 780px;
}
.rp-h {
    font-size: 1rem;
    font-weight: 600;
    color: __TEXT__;
    margin: 0;
}
.rp-sub {
    font-size: .8rem;
    color: __MUTED__;
    margin: .15rem 0 .8rem;
}

.rp-kpi {
    background: __CARD__;
    border: 1px solid __BORDER__;
    border-radius: 14px;
    padding: 1rem 1.15rem;
    height: 100%;
}
.rp-kpi-label {
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: .07em;
    color: __MUTED__;
}
.rp-kpi-value {
    font-size: 1.65rem;
    font-weight: 650;
    color: __TEXT__;
    line-height: 1.2;
    margin: .3rem 0 .15rem;
    overflow-wrap: anywhere;
}
.rp-kpi-sub {
    font-size: .78rem;
    color: __MUTED__;
}

.rp-note {
    border-left: 3px solid __ACCENT__;
    background: rgba(85,201,139,.07);
    border-radius: 8px;
    padding: .8rem 1rem;
    color: __MUTED__;
    font-size: .88rem;
    margin: 1rem 0 1.2rem;
}
.rp-note b { color: __TEXT__; }

.rp-empty {
    border: 1px dashed __BORDER__;
    border-radius: 12px;
    padding: 2.2rem 1rem;
    text-align: center;
    color: __MUTED__;
    font-size: .9rem;
}
.rp-empty b { color: __TEXT__; }

.rp-pill {
    display: inline-block;
    border: 1px solid;
    border-radius: 999px;
    padding: .15rem .8rem;
    font-size: .78rem;
    font-weight: 600;
    letter-spacing: .04em;
}
.rp-big {
    font-size: 2.6rem;
    font-weight: 700;
    color: __TEXT__;
    line-height: 1.1;
    margin: .6rem 0 .1rem;
}
.rp-bar {
    position: relative;
    height: 10px;
    background: __GRID__;
    border-radius: 999px;
    margin: .9rem 0 .3rem;
    overflow: visible;
}
.rp-bar-fill { height: 100%; border-radius: 999px; }
.rp-bar-mark {
    position: absolute;
    top: -4px;
    width: 2px;
    height: 18px;
    background: __TEXT__;
}
.rp-bar-scale {
    display: flex;
    justify-content: space-between;
    font-size: .72rem;
    color: __MUTED__;
}
.rp-logo { margin: .5rem 0 1rem; max-width: 100%; }
.rp-logo svg { max-width: 100%; height: auto; }

@media (max-width: 768px) {
    .rp-title { font-size: 1.4rem; }
    .rp-kpi-value { font-size: 1.3rem; }
}
</style>
"""

for token, value in {
    "__BG__": BG,
    "__CARD__": CARD,
    "__BORDER__": BORDER,
    "__ACCENT__": ACCENT,
    "__TEXT__": TEXT,
    "__MUTED__": MUTED,
    "__GRID__": GRID,
}.items():
    CSS = CSS.replace(token, value)

esc = html.escape


# ============================================================================
# 3. Data loading
# ============================================================================

def check_columns(path: Path, required: list) -> None:
    """Check whether a CSV exists and contains required columns."""
    if not path.exists():
        raise FileNotFoundError(str(path))

    columns = pd.read_csv(path, nrows=0).columns
    missing = [column for column in required if column not in columns]

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {', '.join(missing)}"
        )


@st.cache_data(show_spinner="Loading transactions...")
def load_transactions() -> pd.DataFrame:
    check_columns(TX_PATH, TX_COLUMNS)

    df = pd.read_csv(
        TX_PATH,
        usecols=TX_COLUMNS,
        dtype={
            "InvoiceNo": str,
            "StockCode": str,
            "Description": str,
            "Country": str,
        },
    )

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )

    for column in ["Revenue", "Quantity", "CustomerID"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Description"] = df["Description"].fillna("").str.strip()

    return df.dropna(subset=["InvoiceDate", "Revenue"]).copy()


@st.cache_data(show_spinner="Loading customer segments...")
def load_segments() -> pd.DataFrame:
    check_columns(SEG_PATH, SEG_COLUMNS)

    df = pd.read_csv(SEG_PATH, usecols=SEG_COLUMNS)

    df["CustomerID"] = pd.to_numeric(
        df["CustomerID"], errors="coerce"
    )
    df = df.dropna(subset=["CustomerID"]).copy()
    df["CustomerID"] = df["CustomerID"].astype(int)

    df["Segment"] = df["Segment"].fillna("Unassigned").astype(str)
    df["Cluster_Name"] = (
        df["Cluster_Name"].fillna("Unassigned").astype(str)
    )

    return df


@st.cache_data
def load_product_names() -> pd.Series:
    tx = load_transactions()
    named = tx[tx["Description"] != ""].drop_duplicates("StockCode")
    return named.set_index("StockCode")["Description"]


def load_or_stop(loader, name: str):
    """Display a useful error if a required resource cannot be loaded."""
    try:
        return loader()
    except FileNotFoundError as error:
        st.error(
            f"**{name} not found.** Expected file: `{error.args[0]}`. "
            "Create the file, then select **Reload data** in the sidebar."
        )
    except ValueError as error:
        st.error(f"**{name} could not be used.** {error}")
    except Exception as error:
        st.error(f"**Unexpected problem while loading {name}:** {error}")

    st.stop()


# ============================================================================
# 4. Formatting and reusable UI helpers
# ============================================================================

def fmt_money(value: float) -> str:
    if pd.isna(value):
        return "—"
    if abs(value) >= 1_000_000:
        return f"{CURRENCY}{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{CURRENCY}{value / 1_000:.1f}K"
    return f"{CURRENCY}{value:,.0f}"


def fmt_int(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{int(value):,}"


def header(title: str, description: str) -> None:
    st.markdown(
        '<div class="rp-eyebrow">'
        "RetailPulse · Revenue &amp; Customer Intelligence"
        "</div>"
        f'<h1 class="rp-title">{esc(title)}</h1>'
        f'<p class="rp-desc">{esc(description)}</p>',
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<p class="rp-h">{esc(title)}</p>'
        f'<p class="rp-sub">{esc(subtitle)}</p>',
        unsafe_allow_html=True,
    )


def card(key: str):
    return st.container(key=f"card_{key}")


def kpi_row(items: list) -> None:
    columns = st.columns(len(items))

    for column, (label, value, subtitle) in zip(columns, items):
        column.markdown(
            f'<div class="rp-kpi">'
            f'<div class="rp-kpi-label">{esc(str(label))}</div>'
            f'<div class="rp-kpi-value">{esc(str(value))}</div>'
            f'<div class="rp-kpi-sub">{esc(str(subtitle))}</div>'
            "</div>",
            unsafe_allow_html=True,
        )


def note(text_html: str) -> None:
    st.markdown(
        f'<div class="rp-note">{text_html}</div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, height: int = 340, grid: str | None = "y"):
    fig.update_layout(
        height=height,
        margin=dict(l=4, r=4, t=8, b=4),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=12),
        hoverlabel=dict(
            bgcolor=CARD,
            bordercolor=BORDER,
            font=dict(color=TEXT),
        ),
    )

    fig.update_xaxes(
        title=None,
        zeroline=False,
        linecolor=BORDER,
        showgrid=(grid == "x"),
        gridcolor=GRID,
    )
    fig.update_yaxes(
        title=None,
        zeroline=False,
        linecolor=BORDER,
        showgrid=(grid == "y"),
        gridcolor=GRID,
    )

    return fig


def show_chart(fig, key: str) -> None:
    st.plotly_chart(
        fig,
        width="stretch",
        theme=None,
        key=key,
        config={"displayModeBar": False},
    )


def ranked_bar(
    labels,
    values,
    key: str,
    text_fmt,
    hover_fmt: str,
    name: str,
) -> None:
    data = pd.DataFrame({
        "label": list(labels),
        "value": list(values),
    })

    if data.empty:
        st.info("Nothing to show for the current selection.")
        return

    data["label"] = data["label"].astype(str)
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["value"])

    if data.empty:
        st.info("No valid values are available for this chart.")
        return

    fig = px.bar(data, x="value", y="label", orientation="h")

    fig.update_traces(
        marker_color=ACCENT,
        text=[text_fmt(value) for value in data["value"]],
        textposition="outside",
        cliponaxis=False,
        textfont=dict(color=TEXT, size=11),
        hovertemplate=f"%{{y}}<br>{name}: {hover_fmt}<extra></extra>",
    )

    style_fig(fig, height=max(230, 34 * len(data) + 40), grid=None)
    fig.update_yaxes(categoryorder="total ascending")

    maximum = float(data["value"].max())
    minimum = min(0, float(data["value"].min()))
    padding = max(abs(maximum) * 0.2, 1)

    fig.update_xaxes(
        range=[minimum, maximum + padding],
        showticklabels=False,
    )

    show_chart(fig, key)


# ============================================================================
# 5. Page: Business Overview
# ============================================================================

def page_overview() -> None:
    header(
        "Business Overview",
        "Explore revenue, invoices, customers and products across the "
        "cleaned transaction history. Filter by date and country.",
    )

    tx = load_or_stop(load_transactions, "Transaction data")

    if tx.empty:
        st.info("The transaction file contains no usable rows.")
        return

    min_date = tx["InvoiceDate"].min().date()
    max_date = tx["InvoiceDate"].max().date()
    countries = sorted(tx["Country"].dropna().unique())

    with card("filters"):
        col_date, col_country = st.columns([1, 2])

        selected_dates = col_date.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="ov_dates",
        )

        selected_countries = col_country.multiselect(
            "Countries",
            countries,
            placeholder="All countries",
            key="ov_countries",
        )

    if isinstance(selected_dates, (tuple, list)):
        if len(selected_dates) == 2:
            start_date, end_date = selected_dates
        elif len(selected_dates) == 1:
            start_date = end_date = selected_dates[0]
        else:
            start_date, end_date = min_date, max_date
    else:
        start_date = end_date = selected_dates

    mask = (
        (tx["InvoiceDate"] >= pd.Timestamp(start_date))
        & (
            tx["InvoiceDate"]
            < pd.Timestamp(end_date) + pd.Timedelta(days=1)
        )
    )

    if selected_countries:
        mask &= tx["Country"].isin(selected_countries)

    df = tx.loc[mask].copy()

    if df.empty:
        st.info(
            "No transactions match these filters. "
            "Widen the date range or clear the country filter."
        )
        return

    revenue = df["Revenue"].sum()
    invoices = df["InvoiceNo"].nunique()
    customers = df["CustomerID"].nunique()
    products = df["StockCode"].nunique()

    identified_revenue = df.loc[
        df["CustomerID"].notna(), "Revenue"
    ].sum()

    positive_units = df.loc[df["Quantity"] > 0, "Quantity"].sum()
    returned_units = abs(df.loc[df["Quantity"] < 0, "Quantity"].sum())

    kpi_row([
        (
            "Net transaction revenue",
            fmt_money(revenue),
            "Includes negative cancellation/return values",
        ),
        (
            "Unique invoices",
            fmt_int(invoices),
            "May include cancellation invoices",
        ),
        (
            "Identified customers",
            fmt_int(customers),
            (
                f"Avg recorded revenue {fmt_money(identified_revenue / customers)}"
                if customers else "No identified customers"
            ),
        ),
        (
            "Unique products",
            fmt_int(products),
            f"Positive units: {fmt_int(positive_units)}",
        ),
    ])

    st.caption(
        f"Negative-quantity rows total {fmt_int(returned_units)} units "
        "in absolute terms. Invoice and revenue totals include the rows "
        "present in the processed dataset."
    )
    st.write("")

    with card("trend"):
        section(
            "Monthly revenue",
            f"Net recorded revenue per month ({CURRENCY})",
        )

        monthly = (
            df.groupby(df["InvoiceDate"].dt.to_period("M"))["Revenue"]
            .sum()
            .rename("Revenue")
            .reset_index()
        )
        monthly["Month"] = monthly["InvoiceDate"].dt.to_timestamp()

        fig = px.line(monthly, x="Month", y="Revenue", markers=True)
        fig.update_traces(
            line=dict(color=ACCENT, width=2.5),
            marker=dict(size=7, color=MINT),
            fill="tozeroy",
            fillcolor="rgba(85,201,139,0.10)",
            hovertemplate=(
                "%{x|%b %Y}<br>"
                f"Revenue: {CURRENCY}%{{y:,.0f}}<extra></extra>"
            ),
        )

        style_fig(fig, height=340, grid="y")
        fig.update_yaxes(tickprefix=CURRENCY, tickformat="~s")
        fig.update_xaxes(tickformat="%b %Y")
        show_chart(fig, "ov_trend")

        first_date = df["InvoiceDate"].min()
        last_date = df["InvoiceDate"].max()
        partial_months = []

        if first_date.day > 1:
            partial_months.append(
                f"{first_date:%b %Y} starts on {first_date:%d %b}"
            )
        if last_date.day < last_date.days_in_month:
            partial_months.append(
                f"{last_date:%b %Y} ends on {last_date:%d %b}"
            )
        if partial_months:
            st.caption(
                "Partial months in the selected range: "
                + "; ".join(partial_months)
                + ". These are not full-month totals."
            )

    st.write("")
    left, right = st.columns(2)

    with left, card("countries"):
        section(
            "Revenue by country",
            f"Top markets by net recorded revenue ({CURRENCY})",
        )

        top_n = st.segmented_control(
            "Countries shown",
            [5, 10, 15],
            default=10,
            format_func=lambda number: f"Top {number}",
            key="ov_topn",
            label_visibility="collapsed",
        ) or 10

        by_country = (
            df.groupby("Country")["Revenue"].sum().sort_values(
                ascending=False
            )
        )
        top_countries = by_country.head(top_n)

        ranked_bar(
            top_countries.index,
            top_countries.values,
            "ov_country_bar",
            fmt_money,
            f"{CURRENCY}%{{x:,.0f}}",
            "Revenue",
        )

        if len(by_country) > 0 and revenue != 0:
            st.caption(
                f"{by_country.index[0]} contributes "
                f"{by_country.iloc[0] / revenue:.0%} of net recorded "
                "revenue in this selection."
            )

    with right, card("products"):
        section(
            "Top products",
            "Ranked by revenue or positive units sold",
        )

        measure = st.segmented_control(
            "Product measure",
            ["Revenue", "Units sold"],
            default="Revenue",
            key="ov_measure",
            label_visibility="collapsed",
        ) or "Revenue"

        df_product = df.copy()
        df_product["PositiveUnits"] = df_product["Quantity"].clip(lower=0)

        product_summary = (
            df_product.groupby("StockCode")
            .agg(
                Revenue=("Revenue", "sum"),
                Units=("PositiveUnits", "sum"),
            )
        )

        metric = "Revenue" if measure == "Revenue" else "Units"
        product_summary = product_summary.sort_values(
            metric, ascending=False
        ).head(10)

        names = load_product_names()
        labels = pd.Series(
            [names.get(code, code) for code in product_summary.index],
            index=product_summary.index,
        ).fillna("Unknown product")

        labels = labels.str.title().str.slice(0, 30)
        duplicates = labels.duplicated(keep=False)
        labels.loc[duplicates] = (
            labels.loc[duplicates]
            + " ("
            + labels.loc[duplicates].index.astype(str)
            + ")"
        )

        ranked_bar(
            labels.values,
            product_summary[metric].values,
            "ov_product_bar",
            fmt_money if metric == "Revenue" else fmt_int,
            (
                f"{CURRENCY}%{{x:,.0f}}"
                if metric == "Revenue" else "%{x:,.0f}"
            ),
            metric,
        )

    st.write("")

    with card("tables"):
        section(
            "Detailed values",
            "Use the tabs to inspect countries and products.",
        )
        country_tab, product_tab = st.tabs(["Countries", "Products"])

        with country_tab:
            country_table = (
                df.groupby("Country")
                .agg(
                    Revenue=("Revenue", "sum"),
                    Invoices=("InvoiceNo", "nunique"),
                    Customers=("CustomerID", "nunique"),
                )
                .sort_values("Revenue", ascending=False)
                .reset_index()
            )

            country_table["Share"] = (
                country_table["Revenue"] / revenue * 100
                if revenue != 0 else 0
            )

            st.dataframe(
                country_table,
                hide_index=True,
                width="stretch",
                height=360,
                column_config={
                    "Revenue": st.column_config.NumberColumn(
                        "Net revenue", format=f"{CURRENCY}%.0f"
                    ),
                    "Invoices": st.column_config.NumberColumn(
                        "Unique invoices", format="%d"
                    ),
                    "Customers": st.column_config.NumberColumn(
                        "Identified customers", format="%d"
                    ),
                    "Share": st.column_config.NumberColumn(
                        "% of net revenue", format="%.1f%%"
                    ),
                },
            )

        with product_tab:
            query = st.text_input(
                "Search products",
                placeholder="Description or stock code",
                key="ov_product_search",
            )

            product_table = (
                df.groupby("StockCode")
                .agg(
                    Revenue=("Revenue", "sum"),
                    Quantity=("Quantity", "sum"),
                    Invoices=("InvoiceNo", "nunique"),
                )
                .sort_values("Revenue", ascending=False)
                .reset_index()
            )

            product_names = load_product_names()
            product_table["Product"] = (
                product_table["StockCode"]
                .map(product_names)
                .fillna("")
                .str.title()
            )

            if query.strip():
                q = query.strip().lower()
                matches = (
                    product_table["Product"].str.lower().str.contains(
                        q, regex=False
                    )
                    | product_table["StockCode"].str.lower().str.contains(
                        q, regex=False
                    )
                )
                product_table = product_table.loc[matches]

            if product_table.empty:
                st.info("No products match your search.")
            else:
                st.dataframe(
                    product_table[
                        [
                            "StockCode", "Product",
                            "Quantity", "Invoices", "Revenue",
                        ]
                    ],
                    hide_index=True,
                    width="stretch",
                    height=360,
                    column_config={
                        "StockCode": st.column_config.TextColumn(
                            "Stock code"
                        ),
                        "Quantity": st.column_config.NumberColumn(
                            "Net quantity", format="%d"
                        ),
                        "Invoices": st.column_config.NumberColumn(
                            "Unique invoices", format="%d"
                        ),
                        "Revenue": st.column_config.NumberColumn(
                            "Net revenue", format=f"{CURRENCY}%.0f"
                        ),
                    },
                )


# ============================================================================
# 6. Page: Customer Segmentation
# ============================================================================

def summarise(seg: pd.DataFrame, by: str) -> pd.DataFrame:
    summary = (
        seg.groupby(by)
        .agg(
            Customers=("CustomerID", "nunique"),
            Revenue=("Monetary", "sum"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean"),
        )
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )

    total_customers = summary["Customers"].sum()
    total_revenue = summary["Revenue"].sum()

    summary["CustomerShare"] = (
        summary["Customers"] / total_customers * 100
        if total_customers else 0
    )
    summary["RevenueShare"] = (
        summary["Revenue"] / total_revenue * 100
        if total_revenue else 0
    )

    return summary


def summary_table(
    summary: pd.DataFrame,
    label_col: str,
    label_title: str,
) -> None:
    columns = [
        label_col, "Customers", "CustomerShare", "Revenue",
        "RevenueShare", "AvgRecency", "AvgFrequency", "AvgMonetary",
    ]

    st.dataframe(
        summary[columns],
        hide_index=True,
        width="stretch",
        column_config={
            label_col: st.column_config.TextColumn(label_title),
            "Customers": st.column_config.NumberColumn(
                "Customers", format="%d"
            ),
            "CustomerShare": st.column_config.NumberColumn(
                "% customers", format="%.1f%%"
            ),
            "Revenue": st.column_config.NumberColumn(
                "Monetary total", format=f"{CURRENCY}%.0f"
            ),
            "RevenueShare": st.column_config.NumberColumn(
                "% monetary total", format="%.1f%%"
            ),
            "AvgRecency": st.column_config.NumberColumn(
                "Avg recency (days)", format="%.0f"
            ),
            "AvgFrequency": st.column_config.NumberColumn(
                "Avg frequency", format="%.1f"
            ),
            "AvgMonetary": st.column_config.NumberColumn(
                "Avg spend", format=f"{CURRENCY}%.0f"
            ),
        },
    )


def group_tab(
    seg: pd.DataFrame,
    by: str,
    label: str,
    prefix: str,
) -> None:
    summary = summarise(seg, by)
    left, right = st.columns(2)

    with left, card(f"{prefix}_count"):
        section(f"Customers by {label}", "Number of customers")
        ranked_bar(
            summary[by],
            summary["Customers"],
            f"{prefix}_count_bar",
            fmt_int,
            "%{x:,.0f}",
            "Customers",
        )

    with right, card(f"{prefix}_revenue"):
        section(
            f"Monetary value by {label}",
            f"Sum of customer Monetary values ({CURRENCY})",
        )
        ranked_bar(
            summary[by],
            summary["Revenue"],
            f"{prefix}_revenue_bar",
            fmt_money,
            f"{CURRENCY}%{{x:,.0f}}",
            "Monetary value",
        )

    st.write("")
    with card(f"{prefix}_table"):
        section(
            f"{label.capitalize()} summary",
            "Averages are calculated per customer.",
        )
        summary_table(summary, by, label.capitalize())


def overlap_heatmap(seg: pd.DataFrame) -> None:
    counts = pd.crosstab(seg["Segment"], seg["Cluster_Name"])

    fig = px.imshow(
        counts,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=[
            [0, "#10201A"],
            [1, ACCENT],
        ],
    )
    fig.update_traces(
        hovertemplate=(
            "RFM segment: %{y}<br>"
            "K-Means cluster: %{x}<br>"
            "Customers: %{z:,}<extra></extra>"
        )
    )

    style_fig(fig, height=max(260, 44 * len(counts) + 60), grid=None)
    fig.update_layout(coloraxis_showscale=False)
    show_chart(fig, "seg_overlap")


def customer_detail(view: pd.DataFrame, search: str) -> None:
    section(
        "Customer detail",
        "Choose a customer from the highest-spending 1,000 "
        "in the filtered list.",
    )

    options = (
        view.sort_values("Monetary", ascending=False)["CustomerID"]
        .head(1000)
        .tolist()
    )

    if not options:
        st.info("No customer is available for the current selection.")
        return

    default_index = (
        options.index(int(search))
        if search.strip().isdigit() and int(search) in options
        else 0
    )

    chosen = st.selectbox(
        "Customer",
        options,
        index=default_index,
        format_func=lambda customer_id: f"Customer {customer_id}",
        key="seg_customer_detail",
    )

    row = view.loc[view["CustomerID"] == chosen].iloc[0]

    kpi_row([
        (
            "RFM segment",
            str(row["Segment"]),
            f"RFM score {row['RFM_Score']}",
        ),
        (
            "K-Means cluster",
            str(row["Cluster_Name"]),
            f"Cluster {row['Cluster']}",
        ),
        (
            "Recency",
            f"{row['Recency']:,.0f} days",
            f"R score {row['R_Score']}",
        ),
        (
            "Total spend",
            fmt_money(row["Monetary"]),
            (
                f"{row['Frequency']:,.0f} purchases · "
                f"F {row['F_Score']} · M {row['M_Score']}"
            ),
        ),
    ])

    if st.toggle(
        "Show recent invoices for this customer",
        key="seg_history",
    ):
        tx = load_or_stop(load_transactions, "Transaction data")
        customer_tx = tx[tx["CustomerID"] == chosen]

        if customer_tx.empty:
            st.info("No transactions found for this customer.")
            return

        invoices = (
            customer_tx.groupby("InvoiceNo")
            .agg(
                Date=("InvoiceDate", "max"),
                Units=("Quantity", "sum"),
                Revenue=("Revenue", "sum"),
            )
            .sort_values("Date", ascending=False)
            .head(10)
            .reset_index()
        )

        st.dataframe(
            invoices,
            hide_index=True,
            width="stretch",
            column_config={
                "InvoiceNo": st.column_config.TextColumn("Invoice"),
                "Date": st.column_config.DatetimeColumn(
                    "Date", format="DD MMM YYYY"
                ),
                "Units": st.column_config.NumberColumn(
                    "Net units", format="%d"
                ),
                "Revenue": st.column_config.NumberColumn(
                    "Net revenue", format=f"{CURRENCY}%.2f"
                ),
            },
        )


def explorer_tab(seg: pd.DataFrame) -> None:
    with card("explorer"):
        section(
            "Customer explorer",
            "Search customers and filter by RFM segment or K-Means cluster.",
        )

        col_id, col_segment, col_cluster = st.columns(3)

        search = col_id.text_input(
            "Customer ID",
            placeholder="e.g. 12347",
            key="seg_search",
        )
        selected_segments = col_segment.multiselect(
            "RFM segment",
            sorted(seg["Segment"].unique()),
            placeholder="All segments",
            key="seg_filter",
        )
        selected_clusters = col_cluster.multiselect(
            "K-Means cluster",
            sorted(seg["Cluster_Name"].unique()),
            placeholder="All clusters",
            key="cluster_filter",
        )

        view = seg.copy()

        if search.strip():
            view = view[
                view["CustomerID"].astype(str).str.contains(
                    search.strip(), regex=False
                )
            ]
        if selected_segments:
            view = view[view["Segment"].isin(selected_segments)]
        if selected_clusters:
            view = view[view["Cluster_Name"].isin(selected_clusters)]

        if view.empty:
            st.info(
                "No customers match these filters. "
                "Clear a filter or check the Customer ID."
            )
            return

        st.caption(
            f"Showing {len(view):,} of {len(seg):,} customers. "
            "Click a column header to sort."
        )

        columns = [
            "CustomerID", "Segment", "Cluster_Name", "Recency",
            "Frequency", "Monetary", "R_Score", "F_Score",
            "M_Score", "RFM_Score",
        ]
        table = view.sort_values("Monetary", ascending=False)[columns]

        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            height=380,
            column_config={
                "CustomerID": st.column_config.NumberColumn(
                    "Customer ID", format="%d"
                ),
                "Segment": st.column_config.TextColumn("RFM segment"),
                "Cluster_Name": st.column_config.TextColumn(
                    "K-Means cluster"
                ),
                "Recency": st.column_config.NumberColumn(
                    "Recency (days)", format="%d"
                ),
                "Frequency": st.column_config.NumberColumn(
                    "Frequency", format="%d"
                ),
                "Monetary": st.column_config.NumberColumn(
                    "Spend", format=f"{CURRENCY}%.2f"
                ),
                "R_Score": st.column_config.NumberColumn("R", format="%d"),
                "F_Score": st.column_config.NumberColumn("F", format="%d"),
                "M_Score": st.column_config.NumberColumn("M", format="%d"),
                "RFM_Score": st.column_config.TextColumn("RFM score"),
            },
        )

        st.download_button(
            "Download filtered customers (CSV)",
            table.to_csv(index=False).encode("utf-8"),
            file_name="retailpulse_customers.csv",
            mime="text/csv",
            icon=":material/download:",
        )

        st.divider()
        customer_detail(view, search)


def page_segments() -> None:
    header(
        "Customer Segmentation",
        "Compare rule-based RFM segments with unsupervised K-Means "
        "clusters, then explore individual customer profiles.",
    )

    seg = load_or_stop(load_segments, "Customer segments")

    if seg.empty:
        st.info("The customer segments file contains no rows.")
        return

    kpi_row([
        (
            "Customers",
            fmt_int(seg["CustomerID"].nunique()),
            (
                f"{seg['Segment'].nunique()} RFM segments · "
                f"{seg['Cluster_Name'].nunique()} clusters"
            ),
        ),
        (
            "Total customer spend",
            fmt_money(seg["Monetary"].sum()),
            "Sum of customer Monetary values",
        ),
        (
            "Average frequency",
            f"{seg['Frequency'].mean():,.1f}",
            "Purchases per customer",
        ),
        (
            "Median recency",
            f"{seg['Recency'].median():,.0f} days",
            "Days since last purchase",
        ),
    ])

    note(
        "<b>These are two different methods.</b> "
        "RFM segments are <b>rule-based</b>: R, F and M scores are "
        "mapped to named groups. K-Means clusters are "
        "<b>unsupervised</b>: the algorithm groups customers by "
        "similarity. The methods can disagree; the overlap table "
        "shows how customers are distributed across both."
    )

    rfm_tab, kmeans_tab, explorer = st.tabs([
        "RFM segments · rule-based",
        "K-Means clusters · unsupervised",
        "Customer explorer",
    ])

    with rfm_tab:
        group_tab(seg, "Segment", "RFM segment", "rfm")

    with kmeans_tab:
        group_tab(seg, "Cluster_Name", "cluster", "km")

        st.write("")
        with card("overlap"):
            section(
                "Where the two methods overlap",
                "Rows are RFM segments; columns are K-Means clusters.",
            )
            overlap_heatmap(seg)

    with explorer:
        explorer_tab(seg)


# ============================================================================
# 7. Page: Churn Prediction
# ============================================================================

def result_panel(
    predicted: int,
    probability,
    frequency: float,
    monetary: float,
    seg,
) -> None:
    """Display the model prediction and relevant context."""
    label_description = f"Recency >= {CHURN_RECENCY_DAYS} days"
    tone = AMBER if predicted == 1 else ACCENT

    section(
        "Prediction result",
        "Output from the saved Logistic Regression model.",
    )

    parts = [
        f'<span class="rp-pill" style="color:{tone};'
        f'border-color:{tone}">PREDICTED CLASS {predicted}</span>'
    ]

    if probability is not None:
        probability_percent = probability * 100

        parts.extend([
            f'<div class="rp-big">{probability:.1%}</div>',
            (
                '<div class="rp-sub">'
                f'Model probability of class 1 ({label_description})'
                '</div>'
            ),
            (
                '<div class="rp-bar">'
                f'<div class="rp-bar-fill" '
                f'style="width:{probability_percent:.1f}%;'
                f'background:{tone}"></div>'
                '<div class="rp-bar-mark" style="left:50%"></div>'
                '</div>'
            ),
            (
                '<div class="rp-bar-scale">'
                '<span>0%</span>'
                '<span>50% decision threshold</span>'
                '<span>100%</span>'
                '</div>'
            ),
        ])

    st.markdown("".join(parts), unsafe_allow_html=True)

    if predicted == 1:
        meaning = (
            f"This purchase profile was classified as "
            f"<b>class 1</b>, meaning Recency >= "
            f"{CHURN_RECENCY_DAYS} days in the training labels."
        )
    else:
        meaning = (
            f"This purchase profile was classified as "
            f"<b>class 0</b>, meaning Recency < "
            f"{CHURN_RECENCY_DAYS} days in the training labels."
        )

    note(
        meaning
        + " This is a statistical association in the training dataset, "
        "not a confirmed future outcome for a real customer."
    )

    if seg is not None and not seg.empty:
        frequency_percentile = (
            seg["Frequency"] <= frequency
        ).mean() * 100
        monetary_percentile = (
            seg["Monetary"] <= monetary
        ).mean() * 100

        st.caption(
            f"Context: Frequency is at approximately the "
            f"{frequency_percentile:.0f}th percentile, and spend is "
            f"at approximately the {monetary_percentile:.0f}th "
            f"percentile among {len(seg):,} customers."
        )


def page_churn() -> None:
    header(
        "Churn Prediction",
        f"Score a purchase profile using the project's saved model. "
        f"The target is a historical inactivity label "
        f"(Recency >= {CHURN_RECENCY_DAYS} days), "
        "not verified future churn.",
    )

    # Segments are optional. They provide default input values
    # and context for comparing a profile with existing customers.
    try:
        seg = load_segments()
    except Exception:
        seg = None

    if seg is not None and not seg.empty:
        freq_default = max(1, int(seg["Frequency"].median()))
        monetary_default = max(
            1.0, round(float(seg["Monetary"].median()), 2)
        )
    else:
        freq_default = 5
        monetary_default = 500.0

    left, right = st.columns([2, 3], gap="large")

    with left, card("churn_inputs"):
        section(
            "Customer profile",
            "Enter values from the customer's purchase history.",
        )

        with st.form("churn_form", border=False):
            frequency = st.number_input(
                "Frequency (number of purchases)",
                min_value=1,
                max_value=100_000,
                value=freq_default,
                step=1,
                help="Purchase frequency, as defined in your RFM table.",
            )

            monetary = st.number_input(
                f"Monetary value (total spend, {CURRENCY})",
                min_value=0.01,
                max_value=10_000_000.0,
                value=monetary_default,
                step=10.0,
                format="%.2f",
                help="Total recorded spend for the customer.",
            )

            submitted = st.form_submit_button(
                "Run prediction",
                type="primary",
                width="stretch",
            )

    with right, card("churn_result"):
        if not submitted:
            st.markdown(
                '<div class="rp-empty">'
                '<b>No prediction yet</b><br>'
                "Enter Frequency and Monetary values, then select "
                "<b>Run prediction</b>."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            try:
                # Use the shared function from src/predict_churn.py.
                # Its probability is returned as a percentage from 0 to 100.
                predicted, probability_percent = predict_customer_churn({
                    "Frequency": frequency,
                    "Monetary": monetary,
                })

            except (ValueError, FileNotFoundError) as error:
                st.error(f"Could not make prediction: {error}")

            except Exception as error:
                st.error(f"Unexpected prediction error: {error}")

            else:
                if seg is not None and not seg.empty:
                    beyond_range = (
                        frequency > seg["Frequency"].max()
                        or monetary > seg["Monetary"].max()
                    )

                    if beyond_range:
                        st.warning(
                            "At least one input exceeds the largest "
                            "value in your customer data. Treat this "
                            "prediction with extra caution."
                        )

                # result_panel expects a fraction between 0 and 1,
                # so convert the returned percentage.
                probability = (
                    float(probability_percent) / 100
                    if probability_percent is not None
                    else None
                )

                result_panel(
                    int(predicted),
                    probability,
                    float(frequency),
                    float(monetary),
                    seg,
                )

    st.write("")

    with card("churn_limits"):
        section(
            "How to read this result",
            "Important limitations of the current model.",
        )

        label_share = ""

        if (
            seg is not None
            and not seg.empty
            and "Recency" in seg.columns
        ):
            share = (
                seg["Recency"] >= CHURN_RECENCY_DAYS
            ).mean()
            label_share = (
                f" In your customer data, {share:.0%} carry this label."
            )

        st.markdown(
            f"""
- **Project-defined inactivity label:** Class 1 means
  `Recency >= {CHURN_RECENCY_DAYS}` days.{label_share}
- **Model inputs:** The model uses Frequency and Monetary.
  Recency defines the label but is not an input to this model.
- **Probability:** This is the model's estimated probability of
  class 1, not a verified forecast of future churn.
- **50% reference line:** This is a reference threshold, not
  necessarily the best threshold for a business decision.
- **Evaluation:** Refer to the `churn_prediction` notebook for
  the model's training and test metrics.
- **Limitations:** The data comes from one retailer and a limited
  historical period, so results may differ for other businesses.
"""
        )


# ============================================================================
# 8. Application shell
# ============================================================================

def main() -> None:
    st.set_page_config(
        page_title="RetailPulse",
        page_icon=":material/monitoring:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CSS, unsafe_allow_html=True)

    pages = [
        st.Page(
            page_overview,
            title="Business Overview",
            icon=":material/dashboard:",
            default=True,
        ),
        st.Page(
            page_segments,
            title="Customer Segmentation",
            icon=":material/groups:",
            url_path="segments",
        ),
        st.Page(
            page_churn,
            title="Churn Prediction",
            icon=":material/query_stats:",
            url_path="churn",
        ),
    ]

    navigation = st.navigation(pages)

    with st.sidebar:
        st.markdown(
            f'<div class="rp-logo">{LOGO_SVG}</div>',
            unsafe_allow_html=True,
        )

        st.caption("Revenue & Customer Intelligence")
        st.divider()

        if st.button(
            "Reload data",
            icon=":material/refresh:",
            width="stretch",
            help=(
                "Clear cached data after updating files or "
                "rerunning notebooks."
            ),
        ):
            st.cache_data.clear()
            st.toast("Data cache cleared. Resources will reload from disk.")

        st.caption(
            "Dataset: Online Retail. "
            "Amounts are in pounds sterling. "
            "Results depend on the processed data and saved models."
        )

    navigation.run()


if __name__ == "__main__":
    main()
