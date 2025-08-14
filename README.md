# Dynamic SQL Analytics Dashboard
A Streamlit-based interactive analytics dashboard that allows you to connect to any PostgreSQL database, explore tables, filter data dynamically, auto-detect relationships, compute KPIs, and generate smart visualizations — all without writing SQL manually.

Features

Easy DB Connection — Connect to any PostgreSQL database via UI (no code changes needed).

Table and Schema Exploration — Automatically fetches table names, row counts, and column types.

Relationship Detection — Auto-detects foreign key and inferred relationships between tables.

Single Table Analysis — Apply advanced filters, view paginated data, and generate KPIs.

Cross-Table Analysis — Auto-joins multiple tables based on relationships for combined analysis.

Smart Visualizations — Automatically suggests and generates:

Bar Charts

Pie and Donut Charts

Line Charts (time trends)

Scatter Plots

Histograms

Stacked Bar Charts

KPIs — Auto-generates relevant KPIs based on column type and name heuristics.

Data Export — Export filtered views to CSV or Excel.

Optimized Performance — Uses caching (st.cache_data) for table metadata and repeated queries.

Interactive Data Tables — Supports st_aggrid if installed, for advanced table features.
