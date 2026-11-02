"""
TreadVision: Module 3 — Tyre Tread Database & Technical Reference Catalogue
"""
from pathlib import Path
import streamlit as st
import pandas as pd
from app.database.connection import get_db_session
from app.database.models import Tyre, Vehicle, Fitment, TreadPattern
from app.database.importer import import_tyres_from_csv
from app.ui.components import render_header


def render_database():
    """Render the Tyre Database & Technical Reference Catalogue."""
    render_header(
        "TREAD DATABASE",
        "Reference library of tyre tread patterns, specifications and vehicle applications."
    )

    # Forensic Matcher Shortcut Panel (Restrained Engineering Callout)
    f_c1, f_c2 = st.columns([3, 1], gap="medium")
    with f_c1:
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #D8D5CE; border-left: 3px solid #8F1D2C; border-radius: 4px; padding: 14px 18px;">
            <div style="font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; color: #8F1D2C; font-weight: 700; margin-bottom: 2px;">
                FORENSIC CAPABILITY
            </div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #171717; margin-bottom: 2px;">
                Forensic Tread Impression Matcher
            </div>
            <div style="font-size: 0.84rem; color: #696762;">
                Query an unknown crime-scene tyre impression against reference patterns to retrieve candidate tyre models and vehicle applications.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with f_c2:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        if st.button("ANALYZE IMPRESSION", key="btn_db_forensic", use_container_width=True, type="primary"):
            st.session_state["nav_selection"] = "forensic"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    tab_catalog, tab_vehicles, tab_import = st.tabs(["REFERENCE PATTERNS", "VEHICLE APPLICATIONS", "IMPORT RECORDS"])

    session = get_db_session()
    try:
        # TAB 1: Technical Catalogue
        with tab_catalog:
            # Search & Multi-Filters
            st.markdown("""
            <div style="font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; color: #171717; font-weight: 700; margin-bottom: 6px;">
                SEARCH REFERENCE CATALOGUE
            </div>
            """, unsafe_allow_html=True)

            col_search, col_mfg, col_cat, col_view = st.columns([2, 1, 1, 1], gap="small")
            with col_search:
                search_query = st.text_input("Search model or pattern:", "", placeholder="e.g. Michelin Primacy 4", label_visibility="collapsed")
            with col_mfg:
                mfgs = sorted(list({t.manufacturer for t in session.query(Tyre.manufacturer).all()}))
                selected_mfg = st.selectbox("Manufacturer:", ["All Manufacturers"] + mfgs, label_visibility="collapsed")
            with col_cat:
                cats = sorted(list({t.category for t in session.query(Tyre.category).all()}))
                selected_cat = st.selectbox("Category:", ["All Categories"] + cats, label_visibility="collapsed")
            with col_view:
                view_mode = st.selectbox("Display Mode:", ["Catalogue Table", "Technical Cards"], label_visibility="collapsed")

            # Database Query Filter
            query = session.query(Tyre)
            if search_query:
                query = query.filter(Tyre.model.ilike(f"%{search_query}%") | Tyre.pattern_name.ilike(f"%{search_query}%") | Tyre.manufacturer.ilike(f"%{search_query}%"))
            if selected_mfg != "All Manufacturers":
                query = query.filter(Tyre.manufacturer == selected_mfg)
            if selected_cat != "All Categories":
                query = query.filter(Tyre.category == selected_cat)

            tyres = query.all()

            st.markdown(f"<div style='font-size: 0.78rem; color: #696762; margin-top: 8px; margin-bottom: 16px;'>INDEXED RECORDS: <span class='tv-mono' style='color:#171717; font-weight:600;'>{len(tyres)}</span> TYRE MODELS</div>", unsafe_allow_html=True)

            if view_mode == "Catalogue Table":
                # Structured Catalogue List
                table_rows = []
                for t in tyres:
                    vehicles = [f"{f.vehicle.manufacturer} {f.vehicle.model}" for f in t.fitments if f.vehicle]
                    app_count = len(set(vehicles))
                    grooves = t.tread_pattern.groove_count if t.tread_pattern else 4
                    p_type = t.tread_pattern.pattern_type if t.tread_pattern else "Asymmetric"
                    table_rows.append({
                        "MANUFACTURER": t.manufacturer.upper(),
                        "MODEL": t.model,
                        "PATTERN": t.pattern_name,
                        "CATEGORY": t.category,
                        "GEOMETRY": p_type,
                        "GROOVES": grooves,
                        "APPLICATIONS": app_count
                    })
                df_cat = pd.DataFrame(table_rows)
                st.dataframe(df_cat, use_container_width=True, hide_index=True)

            # Technical Specification Cards / Expanded View
            st.markdown("<div style='margin-top: 20px; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #171717; margin-bottom: 10px;'>TECHNICAL SPECIFICATIONS & PROFILES</div>", unsafe_allow_html=True)

            for t in tyres:
                with st.expander(f"{t.manufacturer.upper()} — {t.model} ({t.category})", expanded=False):
                    spec_col_img, spec_col_details = st.columns([1.2, 2.5], gap="large")

                    with spec_col_img:
                        if t.tread_pattern and Path(t.tread_pattern.image_path).exists():
                            st.image(t.tread_pattern.image_path, caption=f"Pattern: {t.pattern_name}", use_container_width=True)
                        else:
                            st.markdown("<div style='background:#ECEAE5; height:200px; display:flex; align-items:center; justify-content:center; color:#696762; font-size:0.8rem; border:1px solid #D8D5CE;'>Reference image unavailable</div>", unsafe_allow_html=True)

                    with spec_col_details:
                        st.markdown(f"""
                        <div style="font-size: 0.74rem; letter-spacing: 0.1em; text-transform: uppercase; color: #8F1D2C; font-weight: 700;">
                            {t.category.upper()} &nbsp;|&nbsp; {t.construction.upper()}
                        </div>
                        <h2 style="margin: 2px 0 10px 0; color: #171717;">{t.manufacturer} {t.model}</h2>
                        <div style="font-size: 0.88rem; color: #696762; margin-bottom: 14px; line-height: 1.5;">
                            {t.description}
                        </div>
                        """, unsafe_allow_html=True)

                        # Specification Grid
                        p_type = t.tread_pattern.pattern_type if t.tread_pattern else "Asymmetric"
                        grooves = t.tread_pattern.groove_count if t.tread_pattern else 4
                        symmetry = t.tread_pattern.symmetry if t.tread_pattern else "Asymmetric"

                        st.markdown(f"""
                        <div style="background: #FFFFFF; border: 1px solid #D8D5CE; border-radius: 4px; padding: 14px; margin-bottom: 12px;">
                            <div style="font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: #696762; font-weight: 600; margin-bottom: 6px;">
                                SPECIFICATIONS
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; font-size: 0.82rem;">
                                <div><strong>Pattern:</strong> {t.pattern_name}</div>
                                <div><strong>Geometry:</strong> {p_type}</div>
                                <div><strong>Symmetry:</strong> {symmetry}</div>
                                <div><strong>Drainage Grooves:</strong> {grooves}</div>
                                <div><strong>Construction:</strong> {t.construction}</div>
                                <div><strong>Category:</strong> {t.category}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Available Sizes
                        sizes_list = [s.formatted_size for s in t.sizes]
                        st.markdown(f"**Available OEM Sizes:** `{', '.join(sizes_list) if sizes_list else 'Standard Sizes'}`")

                        # Mapped Vehicle Applications
                        vehicles = [f"{f.vehicle.manufacturer} {f.vehicle.model}" for f in t.fitments if f.vehicle]
                        if vehicles:
                            st.markdown(f"**Common Vehicle Applications:** {', '.join(sorted(list(set(vehicles))))}")

        # TAB 2: Vehicle Fitments Index
        with tab_vehicles:
            st.markdown("""
            <div style="font-size: 0.76rem; letter-spacing: 0.08em; text-transform: uppercase; color: #171717; font-weight: 700; margin-bottom: 8px;">
                VEHICLE FITMENT INDEX
            </div>
            """, unsafe_allow_html=True)

            vehicles = session.query(Vehicle).all()
            for v in vehicles:
                v_fits = [f for f in v.fitments]
                with st.expander(f"{v.manufacturer.upper()} {v.model.upper()} {v.generation or ''} — ({len(v_fits)} Fitment Records)", expanded=False):
                    st.markdown(f"**Production Era:** {v.year_from or 'N/A'} - {v.year_to or 'Present'}")
                    if v_fits:
                        fit_rows = []
                        for f in v_fits:
                            fit_rows.append({
                                "TYRE MANUFACTURER": f.tyre.manufacturer.upper(),
                                "MODEL": f.tyre.model,
                                "SIZE": f.size,
                                "FITMENT TYPE": f.fitment_type,
                                "SOURCE": f.source
                            })
                        df_fits = pd.DataFrame(fit_rows)
                        st.dataframe(df_fits, use_container_width=True, hide_index=True)
                    else:
                        st.markdown("<div style='font-size:0.82rem; color:#696762;'>No direct tyre fitments mapped in current database slice.</div>", unsafe_allow_html=True)

        # TAB 3: CSV Batch Importer
        with tab_import:
            st.markdown("""
            <div style="font-size: 0.76rem; letter-spacing: 0.08em; text-transform: uppercase; color: #171717; font-weight: 700; margin-bottom: 8px;">
                IMPORT TYRE RECORDS (CSV)
            </div>
            <div style="font-size: 0.85rem; color: #696762; margin-bottom: 12px; line-height: 1.5;">
                Extend the database with verified tyre catalogs. The CSV must contain standard schema columns:<br>
                <code class="tv-mono">manufacturer, model, pattern_name, category, construction, description, image_path, pattern_type, sizes</code>
            </div>
            """, unsafe_allow_html=True)

            uploaded_csv = st.file_uploader("Select CSV catalogue file:", type=["csv"])
            if uploaded_csv is not None:
                temp_csv = Path("data") / "temp_import.csv"
                with open(temp_csv, "wb") as f:
                    f.write(uploaded_csv.getbuffer())

                import_res = import_tyres_from_csv(str(temp_csv))
                if import_res["success"]:
                    st.success(import_res["message"])
                    if import_res["errors"]:
                        with st.expander("Import Warnings", expanded=True):
                            for err in import_res["errors"]:
                                st.warning(err)
                else:
                    st.error(import_res["message"])

    finally:
        session.close()
