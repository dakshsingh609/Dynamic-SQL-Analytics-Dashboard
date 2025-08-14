import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
from datetime import datetime
from io import BytesIO
import itertools
import pandas.api.types as ptypes
import re

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
    AGGRID_AVAILABLE = True
except ImportError:
    AGGRID_AVAILABLE = False

# --- Streamlit page config ---
st.set_page_config(
    page_title="Dynamic SQL Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://www.streamlit.io/docs',
        'Report a bug': "https://github.com/streamlit/streamlit/issues",
        'About': "# Dynamic SQL analytics dashboard for any PostgreSQL database."
    }
)

st.title("🚀 Dynamic SQL Analytics Dashboard")
st.markdown("---")

# --- Session state initialization ---
for key, default in {
    "db_connected": False,
    "db_host": "localhost",
    "db_name": "",
    "db_user": "postgres",
    "db_password": "",
    "db_port": "5432",
    "table_names": [],
    "relationships": [],
    "current_filtered_df": pd.DataFrame(),
    "current_kpis": {},
    "current_table_name_for_display": "N/A",
    "current_schema_for_filters": {},
    "active_filters_single": {},
    "active_filters_cross": {},
    "drilldown_filters": {},
    "page_size_single": 50,
    "page_number_single": 1,
    "page_size_cross": 50,
    "page_number_cross": 1,
    "total_rows_single": 0,
    "total_rows_cross": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Database helper functions ---
def get_connection(conn_details):
    return psycopg2.connect(**conn_details)

@st.cache_data(ttl=3600)
def get_table_names(conn_details):
    try:
        with get_connection(conn_details) as conn:
            with conn.cursor() as curs:
                curs.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema='public' AND table_type='BASE TABLE';
                """)
                return [r[0] for r in curs.fetchall()]
    except Exception as e:
        st.error(f"Error loading table names: {e}")
        return []

@st.cache_data(ttl=3600)
def get_table_schema(conn_details, table_name):
    try:
        with get_connection(conn_details) as conn:
            with conn.cursor() as curs:
                curs.execute("""
                    SELECT column_name, data_type FROM information_schema.columns
                    WHERE table_schema='public' AND table_name=%s
                    ORDER BY ordinal_position;
                """, (table_name,))
                return {r[0]: r[1] for r in curs.fetchall()}
    except Exception as e:
        st.error(f"Error loading schema for {table_name}: {e}")
        return {}

@st.cache_data(ttl=3600)
def get_table_row_count(conn_details, table_name):
    try:
        with get_connection(conn_details) as conn:
            with conn.cursor() as curs:
                curs.execute(f"SELECT COUNT(*) FROM \"{table_name}\"")
                return curs.fetchone()[0]
    except Exception as e:
        st.error(f"Error getting row count for {table_name}: {e}")
        return 0

@st.cache_data(ttl=3600)
def get_foreign_key_relationships(conn_details):
    try:
        with get_connection(conn_details) as conn:
            with conn.cursor() as curs:
                curs.execute("""
                  SELECT
                    tc.table_name AS foreign_table,
                    kcu.column_name AS foreign_column,
                    ccu.table_name AS primary_table,
                    ccu.column_name AS primary_column
                  FROM information_schema.table_constraints AS tc
                  JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
                  JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema
                  WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public';
                """)
                return [{
                    "foreign_table": r[0],
                    "foreign_column": r[1],
                    "primary_table": r[2],
                    "primary_column": r[3]
                } for r in curs.fetchall()]
    except Exception as e:
        st.error(f"Error loading foreign key relationships: {e}")
        return []

def detect_all_relationships_extended(conn_details, tables):
    foreign_keys = get_foreign_key_relationships(conn_details)
    schemas = {t: get_table_schema(conn_details, t) for t in tables}
    inferred_rels = []
    for t1, t2 in itertools.combinations(tables, 2):
        s1 = schemas[t1]
        s2 = schemas[t2]
        common_cols = set(s1.keys()).intersection(s2.keys())
        for c in common_cols:
            t1type = str(s1[c]).lower() if s1[c] else ''
            t2type = str(s2[c]).lower() if s2[c] else ''
            if (
                ("int" in t1type and "int" in t2type)
                or ("char" in t1type and "char" in t2type)
                or ("text" in t1type and "text" in t2type)
                or ("numeric" in t1type and "numeric" in t2type)
                or ("boolean" in t1type and "boolean" in t2type)
            ):
                fk_exists = any(
                    (fk['foreign_table'], fk['foreign_column'], fk['primary_table'], fk['primary_column']) == (t1,c,t2,c)
                    or (fk['foreign_table'], fk['foreign_column'], fk['primary_table'], fk['primary_column']) == (t2,c,t1,c)
                    for fk in foreign_keys
                )
                if not fk_exists:
                    inferred_rels.append({
                        "foreign_table": t1, "foreign_column": c,
                        "primary_table": t2, "primary_column": c
                    })
    return foreign_keys + inferred_rels

def format_sql_value(value, dtype):
    if value is None:
        return "NULL"
    if dtype is None:
        try:
            float(value)
            return str(value)
        except:
            return f"'{value}'"
    dtype = str(dtype).lower()
    if any(x in dtype for x in ['int', 'numeric', 'float', 'double precision', 'real']):
        return str(value)
    elif any(x in dtype for x in ['date', 'timestamp']):
        return f"'{value}'"
    else:
        return f"'{value}'"

def where_clause_from_filters(table_name, filters, schema_map=None):
    clauses = []
    prefix = table_name + "_"
    for key, val in filters.items():
        if not key.startswith(prefix):
            continue
        col = key[len(prefix):]
        dtype = None
        if schema_map and key in schema_map:
            dtype = schema_map[key]
        if isinstance(val, dict):
            if val.get('type') == 'range':
                minv, maxv = val.get('min'), val.get('max')
                if minv is not None and maxv is not None:
                    clauses.append(f'"{table_name}"."{col}" BETWEEN {format_sql_value(minv, dtype)} AND {format_sql_value(maxv, dtype)}')
                elif minv is not None:
                    clauses.append(f'"{table_name}"."{col}" >= {format_sql_value(minv, dtype)}')
                elif maxv is not None:
                    clauses.append(f'"{table_name}"."{col}" <= {format_sql_value(maxv, dtype)}')
            elif val.get('type') == 'list':
                vals = val.get('values', [])
                if vals:
                    vals_str = ", ".join(format_sql_value(v, dtype) for v in vals)
                    clauses.append(f'"{table_name}"."{col}" IN ({vals_str})')
            elif val.get('type') == 'eq':
                v = val.get('value')
                if v is not None:
                    clauses.append(f'"{table_name}"."{col}" = {format_sql_value(v, dtype)}')
    return " AND ".join(clauses)

def build_join_clauses(tables, relationships):
    base_table = tables[0]
    remaining = set(tables[1:])
    joins = []
    in_query = {base_table}
    while remaining:
        progressed = False
        for rel in relationships:
            if rel['primary_table'] in in_query and rel['foreign_table'] in remaining:
                left_table, left_col = rel['primary_table'], rel['primary_column']
                right_table, right_col = rel['foreign_table'], rel['foreign_column']
            elif rel['foreign_table'] in in_query and rel['primary_table'] in remaining:
                left_table, left_col = rel['foreign_table'], rel['foreign_column']
                right_table, right_col = rel['primary_table'], rel['primary_column']
            else:
                continue
            join_sql = f'LEFT JOIN "{right_table}" ON "{left_table}"."{left_col}" = "{right_table}"."{right_col}"'
            joins.append(join_sql)
            in_query.add(right_table)
            remaining.remove(right_table)
            progressed = True
            break
        if not progressed:
            break
    return " ".join(joins)

def get_select_and_schema(tables, conn_details):
    select_cols = []
    full_schema_map = {}
    for table in tables:
        schema = get_table_schema(conn_details, table)
        for c, dt in schema.items():
            alias = f"{table}_{c}"
            select_cols.append(f'"{table}"."{c}" AS "{alias}"')
            full_schema_map[alias] = dt
    return ", ".join(select_cols), full_schema_map

@st.cache_data(ttl=3600)
def load_data_from_query(conn_details, query):
    try:
        with get_connection(conn_details) as conn:
            df = pd.read_sql(query, conn)
            for c in df.columns:
                if 'date' in c.lower() or 'time' in c.lower():
                    df[c] = pd.to_datetime(df[c], errors='coerce')
            return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_total_row_count_with_filters(conn_details, base_query):
    try:
        with get_connection(conn_details) as conn:
            count_query = f"SELECT COUNT(*) FROM ({base_query}) AS subquery"
            with conn.cursor() as curs:
                curs.execute(count_query)
                return curs.fetchone()[0]
    except Exception as e:
        st.error(f"Error getting row count: {e}")
        return 0

def load_paged_filtered_data(conn_details, query_base, page_size, page_number):
    offset = (page_number - 1) * page_size
    final_query = f"{query_base} LIMIT {page_size} OFFSET {offset};"
    return load_data_from_query(conn_details, final_query)

def apply_filters_ui(df_for_options, full_schema_map, key_prefix):
    tmp_filters_key = f"tmp_filters_{key_prefix}"
    if tmp_filters_key not in st.session_state:
        st.session_state[tmp_filters_key] = {}
    with st.expander("Apply Filters", expanded=True):
        cols = st.columns(3)
        filter_count = 0
        filters = {}
        for col_alias in sorted(full_schema_map.keys()):
            table_name, col_name = col_alias.split('_', 1)
            col_data = df_for_options.get(col_alias)
            if col_data is None:
                continue
            unique_vals = col_data.nunique(dropna=True)
            if unique_vals > 50:
                filter_count += 1
                continue
            with cols[filter_count % 3]:
                label = f"Filter on {col_alias.replace('_', ' ').title()}"
                tooltip = f"Filter data on {col_alias}"
                if ptypes.is_object_dtype(col_data) or ptypes.is_bool_dtype(col_data):
                    options = sorted(col_data.dropna().unique())
                    default_vals = st.session_state[tmp_filters_key].get(col_alias, {"type": "list", "values": []}).get("values")
                    selected = st.multiselect(label, options, default=default_vals, key=f"{key_prefix}multi{col_alias}", help=tooltip)
                    if selected:
                        filters[col_alias] = {"type": "list", "values": selected}
                elif ptypes.is_numeric_dtype(col_data) and not ptypes.is_datetime64_any_dtype(col_data):
                    try:
                        minv_raw = col_data.min()
                        maxv_raw = col_data.max()
                        if pd.isna(minv_raw) or pd.isna(maxv_raw):
                            st.write(f"{label}: No valid numeric values")
                        else:
                            minv, maxv = float(minv_raw), float(maxv_raw)
                            tf = st.session_state[tmp_filters_key].get(col_alias, {"type": "range", "min": minv, "max": maxv})
                            min_val = tf.get("min", minv)
                            max_val = tf.get("max", maxv)
                            selected = st.slider(label, minv, maxv, (min_val, max_val), key=f"{key_prefix}slider{col_alias}", help=tooltip)
                            if selected[0] != minv or selected[1] != maxv:
                                filters[col_alias] = {"type": "range", "min": selected[0], "max": selected[1]}
                    except Exception:
                        st.write(f"{label}: Unavailable")
                elif ptypes.is_datetime64_any_dtype(col_data):
                    non_null = col_data.dropna()
                    if non_null.empty:
                        st.write(f"{label}: No valid dates")
                    else:
                        min_date_raw, max_date_raw = non_null.min().date(), non_null.max().date()
                        tf = st.session_state[tmp_filters_key].get(col_alias, {"type": "range", "min": min_date_raw, "max": max_date_raw})
                        date_range_value = (tf.get("min", min_date_raw), tf.get("max", max_date_raw))
                        daterange = st.date_input(label, value=date_range_value, min_value=min_date_raw, max_value=max_date_raw, key=f"{key_prefix}date{col_alias}", help=tooltip)
                        if len(daterange) == 2 and (daterange[0] != min_date_raw or daterange[1] != max_date_raw):
                            filters[col_alias] = {"type": "range", "min": daterange[0], "max": daterange[1]}
            filter_count += 1
    st.session_state[tmp_filters_key] = filters
    return filters

def is_numeric_type(dt):
    if dt is None:
        return False
    dt = str(dt).lower()
    return any(x in dt for x in ['int', 'numeric', 'float', 'double', 'real'])

def is_datetime_type(dt):
    if dt is None:
        return False
    dt = str(dt).lower()
    return any(x in dt for x in ['date', 'timestamp'])

def is_categorical_type(dt):
    if dt is None:
        return False
    dt = str(dt).lower()
    return any(x in dt for x in ['char', 'text', 'varchar', 'bool'])

# ...existing code...

def generate_kpis(conn_details, base_query, schema_map):
    kpis = {}
    numeric_cols = [c for c, dt in schema_map.items() if is_numeric_type(dt)]
    # Filter out ID-like columns
    non_id_cols = [c for c in numeric_cols if not re.search(r'(?i)\b(id|id|key)\b', c.split('_', 1)[-1])]
    
    # Heuristics for meaningful columns
    meaningful_cols = [
        c for c in non_id_cols
        if re.search(r'(?i)\b(amount|fee|price|cost|rate|total|quantity|capacity|score|value)\b', c.split('_', 1)[-1])
    ] or non_id_cols  # Fallback to all non-ID numeric columns if none match
    
    # Automatically select aggregates based on column type
    # For monetary/amount-like: SUM and AVG
    # For quantity/capacity-like: SUM and MAX
    # Limit to top 2 columns for simplicity
    selected_cols = meaningful_cols[:2]
    aggregates = {}
    for col in selected_cols:
        col_lower = col.lower()
        if any(term in col_lower for term in ['amount', 'fee', 'price', 'cost', 'total', 'value']):
            aggregates[col] = ['SUM', 'AVG']
        elif any(term in col_lower for term in ['quantity', 'capacity', 'score']):
            aggregates[col] = ['SUM', 'MAX']
        else:
            aggregates[col] = ['AVG', 'MAX']

    try:
        with get_connection(conn_details) as conn:
            with conn.cursor() as curs:
                # Always include COUNT
                curs.execute(f"SELECT COUNT(*) FROM ({base_query}) AS sub")
                count_val = curs.fetchone()[0]
                kpis["Total Rows"] = f"{count_val:,}"

                # Compute automated KPIs
                for col, aggs in aggregates.items():
                    for agg in aggs:
                        curs.execute(f"SELECT {agg}(\"{col}\") FROM ({base_query}) AS sub")
                        val = curs.fetchone()[0]
                        if val is not None:
                            kpis[f"{agg} of {col.split('_',1)[-1].title()}"] = f"{val:,.2f}"
    except Exception as e:
        st.error(f"Error generating KPIs: {e}")

    return kpis

def generate_smart_charts(conn_details, base_query, schema_map, total_rows, key_prefix=""):
    st.markdown("#### Suggested Visualizations")
    numeric_cols = [c for c in schema_map if is_numeric_type(schema_map[c])]
    categorical_cols = [c for c in schema_map if is_categorical_type(schema_map[c]) and c in schema_map]
    datetime_cols = [c for c in schema_map if is_datetime_type(schema_map[c])]

    chart_types = []
    if categorical_cols and numeric_cols:
        chart_types.append("Bar Chart (Category vs Value)")
    if categorical_cols:
        chart_types.append("Bar Chart (Category vs Count)") 
        chart_types.append("Pie Chart (Category Distribution)") 
        chart_types.append("Pie Chart (Category Count Distribution)")
        chart_types.append("Stacked Bar Chart (% Share)")
        chart_types.append("Donut Chart (% Distribution)")
    if datetime_cols and numeric_cols:
        chart_types.append("Line Chart (Time Trend)")
    if len(numeric_cols) >= 2:
        chart_types.append("Scatter Plot (Value vs Value)")
    if numeric_cols:
        chart_types.append("Histogram (Value Distribution)")

    if not chart_types:
        st.warning("No valid chart types for current data.")
        return

    chart_type = st.selectbox("Choose Chart Type:", chart_types, key=f"{key_prefix}_chart_type")

    if chart_type == "Bar Chart (Category vs Value)":
        cat_col = st.selectbox("Category Column:", categorical_cols, key=f"{key_prefix}_bar_cat")
        num_col = st.selectbox("Numeric Column:", numeric_cols, key=f"{key_prefix}_bar_num")
        if cat_col and num_col:
            agg_query = f'SELECT "{cat_col}", SUM("{num_col}") as value FROM ({base_query}) AS sub GROUP BY "{cat_col}" ORDER BY value DESC LIMIT 50'
            grouped = load_data_from_query(conn_details, agg_query)
            fig = px.bar(grouped, x=cat_col, y='value', title=f"Sum of {num_col.split('_',1)[-1].title()} by {cat_col.split('_',1)[-1].title()}", text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Bar Chart (Category vs Count)":
        cat_col = st.selectbox("Category Column:", categorical_cols, key=f"{key_prefix}_bar_cat_count")
        if cat_col:
            agg_query = f'SELECT "{cat_col}", COUNT(*) as count FROM ({base_query}) AS sub GROUP BY "{cat_col}" ORDER BY count DESC LIMIT 50'
            count_df = load_data_from_query(conn_details, agg_query)
            fig = px.bar(count_df, x=cat_col, y='count', title=f"Count of each {cat_col.split('_',1)[-1].title()}", text='count')
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Pie Chart (Category Distribution)":
        cat_col = st.selectbox("Category Column:", categorical_cols, key=f"{key_prefix}_pie_cat")
        if cat_col:
            agg_query = f'SELECT "{cat_col}", COUNT(*) as count FROM ({base_query}) AS sub GROUP BY "{cat_col}" ORDER BY count DESC LIMIT 50'
            df_agg = load_data_from_query(conn_details, agg_query)
            fig = px.pie(df_agg, names=cat_col, values='count', title=f"Distribution of {cat_col.split('_',1)[-1].title()}", hole=0.3)
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Pie Chart (Category Count Distribution)":
        cat_col = st.selectbox("Category Column:", categorical_cols, key=f"{key_prefix}_pie_cat_count")
        if cat_col:
            agg_query = f'SELECT "{cat_col}", COUNT(*) as count FROM ({base_query}) AS sub GROUP BY "{cat_col}" ORDER BY count DESC LIMIT 50'
            df_agg = load_data_from_query(conn_details, agg_query)
            fig = px.pie(df_agg, names=cat_col, values='count', hole=0.5,
                         title=f"Count Distribution of {cat_col.split('_',1)[-1].title()}")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Line Chart (Time Trend)":
        dt_col = st.selectbox("Date/Time Column:", datetime_cols, key=f"{key_prefix}_line_dt")
        num_col = st.selectbox("Numeric Column:", numeric_cols, key=f"{key_prefix}_line_num")
        if dt_col and num_col:
            agg = st.radio("Aggregation Level:", ["Daily", "Monthly", "Yearly"], horizontal=True, key=f"{key_prefix}_line_agg")
            if agg == "Daily":
                period_expr = f"DATE(\"{dt_col}\")"
            elif agg == "Monthly":
                period_expr = f"DATE_TRUNC('month', \"{dt_col}\")"
            else:
                period_expr = f"DATE_TRUNC('year', \"{dt_col}\")"
            agg_query = f'SELECT {period_expr} as period, SUM("{num_col}") as value FROM ({base_query}) AS sub WHERE "{dt_col}" IS NOT NULL GROUP BY period ORDER BY period'
            grouped = load_data_from_query(conn_details, agg_query)
            grouped['period'] = pd.to_datetime(grouped['period'])
            fig = px.line(grouped, x='period', y='value', title=f"{num_col.split('_',1)[-1].title()} Trend Over Time ({agg})")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter Plot (Value vs Value)":
        x_col = st.selectbox("X Axis:", numeric_cols, key=f"{key_prefix}_scatter_x")
        y_opts = [c for c in numeric_cols if c != x_col]
        y_col = st.selectbox("Y Axis:", y_opts, key=f"{key_prefix}_scatter_y")
        if x_col and y_col:
            if total_rows > 10000:
                st.warning("Data too large for full scatter plot. Sampling 10,000 rows.")
                sample_query = f'SELECT "{x_col}", "{y_col}" FROM ({base_query}) AS sub TABLESAMPLE SYSTEM (1) LIMIT 10000'
            else:
                sample_query = f'SELECT "{x_col}", "{y_col}" FROM ({base_query}) AS sub'
            df_sample = load_data_from_query(conn_details, sample_query)
            fig = px.scatter(df_sample, x=x_col, y=y_col, title=f"{x_col.split('_',1)[-1].title()} vs {y_col.split('_',1)[-1].title()}")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Histogram (Value Distribution)":
        num_col = st.selectbox("Select Numeric Column:", numeric_cols, key=f"{key_prefix}_hist_num")
        if num_col:
            if total_rows > 10000:
                st.warning("Data too large for full histogram. Sampling 10,000 rows.")
                sample_query = f'SELECT "{num_col}" FROM ({base_query}) AS sub TABLESAMPLE SYSTEM (1) LIMIT 10000'
            else:
                sample_query = f'SELECT "{num_col}" FROM ({base_query}) AS sub'
            df_sample = load_data_from_query(conn_details, sample_query)
            fig = px.histogram(df_sample, x=num_col, title=f"Distribution of {num_col.split('_',1)[-1].title()}")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Stacked Bar Chart (% Share)":
        cat_col = st.selectbox("Category Column:", categorical_cols, key=f"{key_prefix}_stack_cat")
        sub_opts = [c for c in categorical_cols if c != cat_col]
        sub_col = st.selectbox("Subcategory (e.g. status/client):", sub_opts, key=f"{key_prefix}_stack_sub")
        if cat_col and sub_col:
            agg_query = f'SELECT "{cat_col}", "{sub_col}", COUNT(*) as count FROM ({base_query}) AS sub GROUP BY "{cat_col}", "{sub_col}"'
            count_df = load_data_from_query(conn_details, agg_query)
            total_per_cat = count_df.groupby(cat_col)['count'].transform('sum')
            count_df['percent'] = (count_df['count'] / total_per_cat * 100).round(2)
            fig = px.bar(count_df, x=cat_col, y='percent', color=sub_col,
                         title=f"% Share of {sub_col.split('_',1)[-1].title()} within each {cat_col.split('_',1)[-1].title()}", text='percent')
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Donut Chart (% Distribution)":
        cat_col = st.selectbox("Category Column:", categorical_cols, key=f"{key_prefix}_donut_cat")
        if cat_col:
            agg_query = f'SELECT "{cat_col}", COUNT(*) as count FROM ({base_query}) AS sub GROUP BY "{cat_col}" ORDER BY count DESC LIMIT 50'
            donut_df = load_data_from_query(conn_details, agg_query)
            fig = px.pie(donut_df, names=cat_col, values='count', hole=0.5,
                         title=f"{cat_col.split('_',1)[-1].title()} Distribution (%)")
            st.plotly_chart(fig, use_container_width=True)

# ...existing code...

# --- DB connection dialog ---
# ...existing code...

# --- DB connection dialog (moved to sidebar) ---
def db_connection_dialog():
    with st.sidebar.expander("⚙ Database Connection Settings", expanded=True):
        host = st.text_input("DB Host", value=st.session_state.db_host)
        name = st.text_input("DB Name", value=st.session_state.db_name)
        user = st.text_input("DB User", value=st.session_state.db_user)
        password = st.text_input("DB Password", value=st.session_state.db_password, type="password")
        port = st.text_input("DB Port", value=st.session_state.db_port)

        if st.button("Preview Connection Details"):
            st.info(f"""
                Current DB Connection: 
                Host: {host} 
                Database: {name} 
                User: {user} 
                Port: {port}
            """)

        if st.button("Save Connection Details"):
            st.session_state.db_host = host
            st.session_state.db_name = name
            st.session_state.db_user = user
            st.session_state.db_password = password
            st.session_state.db_port = port
            st.success("Connection details saved!")

        connect = st.button("Connect to Database")

        return connect, {
            "host": host,
            "dbname": name,
            "user": user,
            "password": password,
            "port": port
        }

# --- Main UI ---
connect_clicked, conn_details = db_connection_dialog()

# ...existing code...

if connect_clicked:
    with st.spinner("Connecting..."):
        tables = get_table_names(conn_details)
        if tables:
            st.session_state.db_connected = True
            st.session_state.table_names = tables
            st.session_state.relationships = detect_all_relationships_extended(conn_details, tables)
            st.session_state.active_filters_single.clear()
            st.session_state.active_filters_cross.clear()
            st.session_state.drilldown_filters.clear()
            st.session_state.current_filtered_df = pd.DataFrame()
            st.session_state.current_kpis = {}
            st.session_state.current_table_name_for_display = "N/A"
            st.session_state.current_schema_for_filters = {}
            st.success("Successfully connected!")
        else:
            st.error("Failed to connect or no tables found. Check your credentials.")

st.markdown("---")

# ...existing code...

if st.session_state.db_connected:
    # --- Move Database Table Row Counts to sidebar expander ---
    with st.sidebar.expander("📋 Database Table Row Counts", expanded=False):
        row_counts = {}
        total_rows_db = 0
        for table in st.session_state.table_names:
            count = get_table_row_count(conn_details, table)
            row_counts[table] = count
            total_rows_db += count
        st.write("Table Row Counts:")
        for table, count in row_counts.items():
            st.write(f"{table}: {count:,} rows")
        if total_rows_db >= 1000000:
            st.success(f"The database holds at least 1 million rows (total: {total_rows_db:,}).")
        else:
            st.info(f"The database holds {total_rows_db:,} rows in total.")

    tab_single, tab_cross, tab_filtered = st.tabs([
        "📊 Single Table Analysis", "🤝 Cross-Table Analysis", "📄 Filtered Data & Exports"
    ])

    # ...existing code...

    # --- Tab: Single Table Analysis ---
    with tab_single:
        st.header("Single Table Analysis")
        selected_table = st.selectbox("Select a table", st.session_state.table_names, key="single_table_sel")
        if selected_table:
            schema_map = get_table_schema(conn_details, selected_table)
            select_cols = [f'"{c}" AS "{selected_table}_{c}"' for c in schema_map]
            sample_query = f'SELECT {", ".join(select_cols)} FROM "{selected_table}" LIMIT 1000;'
            df_sample = load_data_from_query(conn_details, sample_query)

            single_schema_map_aliased = {f'{selected_table}_{c}': dt for c, dt in schema_map.items()}
            previous_filters = st.session_state.active_filters_single.get(selected_table, {})
            tmp_filters_key = f"tmp_filters_single_{selected_table}"
            if tmp_filters_key not in st.session_state:
                st.session_state[tmp_filters_key] = previous_filters

            active_filters = apply_filters_ui(df_sample, single_schema_map_aliased, key_prefix=f"single_{selected_table}")
            st.session_state.active_filters_single[selected_table] = active_filters

            where_clause = where_clause_from_filters(selected_table, active_filters, single_schema_map_aliased)
            where_sql = f"WHERE {where_clause}" if where_clause else ""

            base_query = f'SELECT {", ".join(select_cols)} FROM "{selected_table}" {where_sql}'

            with st.expander("Pagination", expanded=True):
                col_page_size, col_page_num = st.columns([1, 4])
                page_size_new = col_page_size.selectbox("Rows per page", [25, 50, 100, 500], index=[25,50,100,500].index(st.session_state.page_size_single), key="page_size_single_sel")

                if page_size_new != st.session_state.page_size_single:
                    st.session_state.page_number_single = 1

                total_rows = get_total_row_count_with_filters(conn_details, base_query)
                st.session_state.total_rows_single = total_rows
                total_pages = (total_rows + page_size_new - 1) // page_size_new

                page_number = col_page_num.number_input("Page number", min_value=1, max_value=max(1, total_pages), value=st.session_state.page_number_single, key="page_number_single_sel")

                st.session_state.page_size_single = page_size_new
                st.session_state.page_number_single = page_number

            df_filtered_paged = load_paged_filtered_data(conn_details, base_query, st.session_state.page_size_single, st.session_state.page_number_single)
            kpis = generate_kpis(conn_details, base_query, single_schema_map_aliased)

            st.subheader("Data Preview")
            st.info(f"Showing page {st.session_state.page_number_single} of {total_pages}. Total filtered rows: {total_rows:,}")
            if AGGRID_AVAILABLE:
                gb = GridOptionsBuilder.from_dataframe(df_filtered_paged)
                gb.configure_side_bar()
                opts = gb.build()
                AgGrid(df_filtered_paged, gridOptions=opts, key=f"single_table_grid_{selected_table}")
            else:
                st.dataframe(df_filtered_paged, use_container_width=True)

            st.subheader("KPIs (Computed on Full Filtered Data)")
            if kpis:
                cols = st.columns(len(kpis))
                for i, (n, v) in enumerate(kpis.items()):
                    with cols[i]:
                        st.metric(label=n, value=v)
            else:
                st.info("No KPIs available. No suitable numeric columns found for aggregation.")

            st.subheader("Visualizations (Computed on Full Filtered Data Where Possible)")
            generate_smart_charts(conn_details, base_query, single_schema_map_aliased, total_rows, key_prefix=f"single_{selected_table}")

            st.session_state.current_filtered_df = df_filtered_paged
            st.session_state.current_kpis = kpis
            st.session_state.current_table_name_for_display = selected_table
            st.session_state.current_schema_for_filters = single_schema_map_aliased
        else:
            st.info("Please select a table.")

    # --- Tab: Cross-Table Analysis ---
    with tab_cross:
        st.header("Cross-Table Analysis (Auto Joined)")
        if len(st.session_state.table_names) < 2:
            st.warning("At least two tables required for cross-table analysis.")
        else:
            rels = st.session_state.relationships
            if not rels:
                st.warning("No relationships detected; cross-table analysis unavailable.")
            else:
                all_tables = st.session_state.table_names
                all_select_cols, full_schema_map = get_select_and_schema(all_tables, conn_details)
                joins = build_join_clauses(all_tables, rels)

                sample_query_cross = f'SELECT {all_select_cols} FROM "{all_tables[0]}" {joins} LIMIT 1000;'
                df_sample_cross = load_data_from_query(conn_details, sample_query_cross)

                tmp_filters_key_cross = "tmp_filters_cross"
                previous_cross_filters = st.session_state.active_filters_cross
                if tmp_filters_key_cross not in st.session_state:
                    st.session_state[tmp_filters_key_cross] = previous_cross_filters

                active_filters = apply_filters_ui(df_sample_cross, full_schema_map, key_prefix="cross")
                st.session_state.active_filters_cross = active_filters

                where_clauses_list = []
                for table in all_tables:
                    clause = where_clause_from_filters(table, active_filters, full_schema_map)
                    if clause:
                        where_clauses_list.append(clause)
                where_sql = f"WHERE {' AND '.join(where_clauses_list)}" if where_clauses_list else ""

                base_query_cross = f'SELECT {all_select_cols} FROM "{all_tables[0]}" {joins} {where_sql}'

                with st.expander("Pagination", expanded=True):
                    col_page_size, col_page_num = st.columns([1, 4])
                    page_size_new = col_page_size.selectbox("Rows per page", [25, 50, 100, 500], index=[25,50,100,500].index(st.session_state.page_size_cross), key="page_size_cross_sel")

                    if page_size_new != st.session_state.page_size_cross:
                        st.session_state.page_number_cross = 1

                    total_rows = get_total_row_count_with_filters(conn_details, base_query_cross)
                    st.session_state.total_rows_cross = total_rows
                    total_pages = (total_rows + page_size_new - 1) // page_size_new

                    page_number = col_page_num.number_input("Page number", min_value=1, max_value=max(1, total_pages), value=st.session_state.page_number_cross, key="page_number_cross_sel")

                    st.session_state.page_size_cross = page_size_new
                    st.session_state.page_number_cross = page_number

                df_joined_paged = load_paged_filtered_data(conn_details, base_query_cross, st.session_state.page_size_cross, st.session_state.page_number_cross)
                kpis = generate_kpis(conn_details, base_query_cross, full_schema_map)

                st.subheader("Joined Data Preview")
                st.info(f"Showing page {st.session_state.page_number_cross} of {total_pages}. Total filtered rows: {total_rows:,}")
                if AGGRID_AVAILABLE:
                    gb = GridOptionsBuilder.from_dataframe(df_joined_paged)
                    gb.configure_side_bar()
                    opts = gb.build()
                    AgGrid(df_joined_paged, gridOptions=opts, key="cross_table_grid")
                else:
                    st.dataframe(df_joined_paged, use_container_width=True)

                st.subheader("KPIs (Computed on Full Filtered Data)")
                if kpis:
                    cols = st.columns(len(kpis))
                    for i, (n, v) in enumerate(kpis.items()):
                        with cols[i]:
                            st.metric(label=n, value=v)
                else:
                    st.info("No KPIs available. No suitable numeric columns found for aggregation.")

                st.subheader("Visualizations (Computed on Full Filtered Data Where Possible)")
                generate_smart_charts(conn_details, base_query_cross, full_schema_map, total_rows, "cross_table")

                st.session_state.current_filtered_df = df_joined_paged
                st.session_state.current_kpis = kpis
                st.session_state.current_table_name_for_display = "Cross-Table Combined"
                st.session_state.current_schema_for_filters = full_schema_map

    # --- Tab: Filtered Data & Exports ---
    with tab_filtered:
        st.header("Filtered Data & Exports")
        if st.session_state.current_filtered_df.empty:
            st.warning("No filtered data available. Run an analysis first.")
        else:
            st.markdown(f"Data Source: {st.session_state.current_table_name_for_display}")

            st.subheader("KPIs")
            if st.session_state.current_kpis:
                kpi_cols = st.columns(len(st.session_state.current_kpis))
                for i, (n, v) in enumerate(st.session_state.current_kpis.items()):
                    with kpi_cols[i]:
                        st.metric(label=n, value=v)
            else:
                st.info("No KPIs available.")

            st.markdown("---")
            st.subheader("Filtered Data Preview")
            if AGGRID_AVAILABLE:
                gb = GridOptionsBuilder.from_dataframe(st.session_state.current_filtered_df)
                gb.configure_side_bar()
                opts = gb.build()
                AgGrid(
                    st.session_state.current_filtered_df,
                    gridOptions=opts,
                    key="current_filtered_data_grid",
                )
            else:
                st.dataframe(st.session_state.current_filtered_df, use_container_width=True)

            st.markdown("---")
            st.subheader("Export Data")
            export_col1, export_col2 = st.columns(2)
            with export_col1:
                csv_buffer = BytesIO()
                st.session_state.current_filtered_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download Current View as CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f'filtered_data_{st.session_state.current_table_name_for_display}.csv',
                    mime='text/csv'
                )
            with export_col2:
                excel_buffer = BytesIO()
                st.session_state.current_filtered_df.to_excel(excel_buffer, index=False)
                st.download_button(
                    label="Download Current View as Excel",
                    data=excel_buffer.getvalue(),
                    file_name=f'filtered_data_{st.session_state.current_table_name_for_display}.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

else:
    st.info("Enter database connection details above to get started.")