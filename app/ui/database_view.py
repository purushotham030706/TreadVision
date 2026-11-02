"""
TreadVision: Module 3 — Tyre Tread Database & Intelligence Catalog
"""
from pathlib import Path
import streamlit as st
from app.database.connection import get_db_session
from app.database.models import Tyre, Vehicle, Fitment
from app.database.importer import import_tyres_from_csv
from app.ui.components import render_header


def render_database():
    """Render the Tyre Database & Automotive Intelligence Catalog."""
    render_header(
        "Module 3: Tyre Tread Database",
        "Explore indexed tyre models, high-resolution tread pattern geometries, and verified vehicle fitments."
    )

    # Prominent Forensic Matcher Banner Card
    f_c1, f_c2 = st.columns([3, 1])
    with f_c1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%); border: 1px solid #4338ca; border-radius: 10px; padding: 18px 20px;">
            <div style="font-family: 'Chakra Petch', sans-serif; font-size: 1.15rem; font-weight: 700; color: #ffffff;">
                🔬 Forensic Impression Matcher
            </div>
            <div style="color: #cbd5e1; font-size: 0.88rem; margin-top: 4px;">
                Identify unknown crime-scene impressions left in mud, soil, sand, or dirt against reference patterns.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with f_c2:
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        if st.button("🚀 Match Impression", use_container_width=True, type="primary"):
            st.session_state["nav_selection"] = "forensic"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    tab_catalog, tab_vehicles, tab_import = st.tabs(["Tyre Catalog", "Vehicle Fitments", "Import Records"])

    session = get_db_session()
    try:
        # TAB 1: Tyre Catalog
        with tab_catalog:
            # Filters
            col_search, col_mfg, col_cat = st.columns([2, 1, 1])
            with col_search:
                search_query = st.text_input("🔍 Search tyre model or pattern name:", "")
            with col_mfg:
                mfgs = sorted(list({t.manufacturer for t in session.query(Tyre.manufacturer).all()}))
                selected_mfg = st.selectbox("Manufacturer:", ["All"] + mfgs)
            with col_cat:
                cats = sorted(list({t.category for t in session.query(Tyre.category).all()}))
                selected_cat = st.selectbox("Category:", ["All"] + cats)

            # Query Filter
            query = session.query(Tyre)
            if search_query:
                query = query.filter(Tyre.model.ilike(f"%{search_query}%") | Tyre.pattern_name.ilike(f"%{search_query}%"))
            if selected_mfg != "All":
                query = query.filter(Tyre.manufacturer == selected_mfg)
            if selected_cat != "All":
                query = query.filter(Tyre.category == selected_cat)

            tyres = query.all()
            st.markdown(f"**Showing {len(tyres)} tyre models:**")

            # Tyre Cards Grid
            for t in tyres:
                with st.expander(f"{t.manufacturer} — {t.model} ({t.category})", expanded=False):
                    card_col_img, card_col_info = st.columns([1, 2], gap="medium")

                    with card_col_img:
                        if t.tread_pattern and Path(t.tread_pattern.image_path).exists():
                            st.image(t.tread_pattern.image_path, caption=t.pattern_name, use_container_width=True)
                        else:
                            st.markdown("<div style='background:#1e293b;height:180px;border-radius:8px;display:flex;align-items:center;justify-content:center;color:#64748b;'>No Pattern Image</div>", unsafe_allow_html=True)

                    with card_col_info:
                        st.markdown(f"### {t.manufacturer} {t.model}")
                        st.markdown(f"**Pattern Name:** `{t.pattern_name}`")
                        st.markdown(f"**Category:** `{t.category}` | **Construction:** `{t.construction}`")
                        if t.tread_pattern:
                            st.markdown(f"**Pattern Type:** `{t.tread_pattern.pattern_type}` | **Groove Count:** `{t.tread_pattern.groove_count}`")
                            st.markdown(f"*{t.tread_pattern.description}*")
                        st.markdown(f"*{t.description}*")

                        # Available Sizes
                        sizes_list = [s.formatted_size for s in t.sizes]
                        st.markdown(f"**Available OEM Sizes:** `{', '.join(sizes_list)}`")

                        # Compatible Vehicles
                        vehicles = [f"{f.vehicle.manufacturer} {f.vehicle.model}" for f in t.fitments if f.vehicle]
                        if vehicles:
                            st.markdown(f"**Compatible Vehicles:** {', '.join(set(vehicles))}")

        # TAB 2: Vehicle Fitments
        with tab_vehicles:
            st.subheader("Verified Vehicle Fitment Index")
            vehicles = session.query(Vehicle).all()
            for v in vehicles:
                v_fits = [f for f in v.fitments]
                with st.expander(f"{v.manufacturer} {v.model} {v.generation or ''} ({len(v_fits)} Associated Tyres)", expanded=False):
                    st.markdown(f"**Production Period:** {v.year_from or 'N/A'} - {v.year_to or 'Present'}")
                    if v_fits:
                        fit_rows = []
                        for f in v_fits:
                            fit_rows.append({
                                "Tyre": f"{f.tyre.manufacturer} {f.tyre.model}",
                                "Size": f.size,
                                "Fitment Type": f.fitment_type,
                                "Source Spec": f.source
                            })
                        st.table(fit_rows)
                    else:
                        st.write("No direct tyre fitments mapped yet.")

        # TAB 3: Import Records
        with tab_import:
            st.subheader("Import Additional Tyre Records via CSV")
            st.markdown("""
            Extend the database with verified tyre catalogs. Upload a CSV with standard column headers:
            `manufacturer, model, pattern_name, category, construction, description, image_path, pattern_type, sizes`
            """)
            uploaded_csv = st.file_uploader("Upload CSV file:", type=["csv"])
            if uploaded_csv is not None:
                # Save temp and run importer
                temp_csv = Path("data") / "temp_import.csv"
                with open(temp_csv, "wb") as f:
                    f.write(uploaded_csv.getbuffer())

                import_res = import_tyres_from_csv(str(temp_csv))
                if import_res["success"]:
                    st.success(import_res["message"])
                    if import_res["errors"]:
                        with st.expander("Import Warnings"):
                            for err in import_res["errors"]:
                                st.warning(err)
                else:
                    st.error(import_res["message"])

    finally:
        session.close()
