import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

from streamlit_option_menu import option_menu
from streamlit_pandas_profiling import st_profile_report
from scipy.interpolate import griddata
from tqdm import tqdm
from dotenv import load_dotenv
from pymongo import MongoClient
from ydata_profiling import ProfileReport
from bson import ObjectId

# App Settings
st.set_page_config(
    page_title="Visualisasi Data Termodinamika",
    layout="wide"
)

st.markdown("""
    <style>
        div.block-container {padding-top:1rem;}
    </style>
""", unsafe_allow_html=True)

# MongoDB Connections
def store_to_mongo(df, collection, mode="append"):
    """
    Stores a dataframe to a MongoDB collection.

    Parameters:
    df (pd.DataFrame): The dataframe to be stored.
    collection (pymongo.collection.Collection): The MongoDB collection.
    mode (str): The mode of operation - "replace" to replace existing data, "append" to add to existing data.
    """
    # Convert dataframe to list of dictionaries with progress bar
    data_dict = [row for row in tqdm(df.to_dict(orient="records"), desc="Converting dataframe")]

    if mode == "replace":
        # Delete all existing documents in the collection
        collection.delete_many({})
        print("Existing data replaced.")
    
    # Insert all documents at once
    collection.insert_many(data_dict)
    print(f"{len(data_dict)} documents inserted.")

def load_from_mongo(collection):
    """
    Loads data from a MongoDB collection into a pandas dataframe.

    Parameters:
    collection (pymongo.collection.Collection): The MongoDB collection.

    Returns:
    pd.DataFrame: The loaded dataframe.
    """
    cursor = collection.find()
    df = pd.DataFrame(list(cursor))
    
    return df

load_dotenv(".env")
client = MongoClient(
    os.getenv("MONGO_CONNECTION_STRING"),
    serverSelectionTimeoutMS=300000
)
db = client[os.getenv("MONGO_DATABASE_NAME")]
collection_baja = db[os.getenv("MONGO_COLLECTION_NAME_BAJA")]
collection_pltu = db[os.getenv("MONGO_COLLECTION_NAME_PLTU")]

# Title
st.markdown("<h1 style='text-align: center;'>Visualisasi Data Termodinamika</h1>", unsafe_allow_html=True)
st.write("")
st.write("")
st.write("")

# Sidebar
with st.sidebar:
    data_source = st.selectbox(
        label="Data Source",
        options=["Baja Tahan Karat", "Pembangkit Listrik Tenaga Uap (PLTU)"]
    )

    st.divider()

    pages = option_menu(
        "Main Menu",
        ["Visualisasi Otomatis", "Visualisasi 2D", "Visualisasi 3D", "Korelasi Pearson", "Data"],
        menu_icon="cast",
        icons=["1-circle-fill", "2-circle-fill", "3-circle-fill", "4-circle-fill", "5-circle-fill"],
        default_index=0
    )

# Data Source
if data_source == "Baja Tahan Karat":
    collection = collection_baja
else:
    collection = collection_pltu

df_id = load_from_mongo(collection)
df = df_id.drop("_id", axis=1).copy()

# Cache
@st.cache_data
def gen_profile_report(data, *report_args, **report_kwargs):
    return data.profile_report(*report_args, **report_kwargs)

# Pages
def visualisasi_otomatis():
    st.info("Click the red button to generate the profile report. Please be aware that this process may take a while, especially with a big dataset.", icon="ℹ️")
    
    col1, col2 = st.columns([6, 1])
    with col2:
        generate = st.button(
            label="Generate",
            type="primary",
            use_container_width=True
        )

    if generate:
        pr = gen_profile_report(df, minimal=True)
        st_profile_report(pr)

def visualisasi_2d():
    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis = st.multiselect(
            label="X Axis",
            options=df.columns,
            default=None,
            max_selections=2
        )

    with col2:
        y_axis = st.multiselect(
            label="Y Axis",
            options=df.columns,
            default=None,
            max_selections=2
        )

    with col3:
        plot_options = ["All", "Box Plot", "Violin Plot", "Scatter Plot", "Line Plot"]
        selected_plots = st.multiselect(
            label="Plots",
            options=plot_options,
            default=None,
        )

    if "All" in selected_plots:
        selected_plots = plot_options[1:]
    else:
        selected_plots = [plot for plot in selected_plots if plot != "All"]

    if x_axis and y_axis:
        total_vars = len(x_axis) + len(y_axis)
        if total_vars > 3:
            st.error("The total number of selected variables for X and Y axes cannot exceed 3. Please adjust your selections.", icon="🚨")
        else:
            if len(x_axis) == 2 and len(y_axis) == 2:
                st.error("You cannot select 2 variables for both X and Y axes simultaneously. Please adjust your selections.", icon="🚨")
            elif any(col in y_axis for col in x_axis):
                st.error("X axis and Y axis cannot have the same values. Please select different columns.", icon="🚨")
            else:
                plot_containers = []
                if len(selected_plots) == 1:
                    plot_containers.append(st.container())
                elif len(selected_plots) == 2:
                    col1, col2 = st.columns(2)
                    plot_containers.extend([col1, col2])
                elif len(selected_plots) == 3:
                    col1, col2 = st.columns(2)
                    plot_containers.extend([col1, col2])
                    plot_containers.append(st.container())
                elif len(selected_plots) == 4:
                    col1, col2 = st.columns(2)
                    plot_containers.extend([col1, col2])
                    col3, col4 = st.columns(2)
                    plot_containers.extend([col3, col4])

                def plot_figure(plot_func, plot_type):
                    fig = go.Figure()
                    colors = px.colors.qualitative.Plotly
                    color_idx = 0
                    for xi in x_axis:
                        for yi in y_axis:
                            if plot_type == "line":
                                fig.add_trace(go.Scatter(
                                    x=df[xi],
                                    y=df[yi],
                                    mode="lines",
                                    line=dict(color=colors[color_idx]),
                                    name=f"{xi} vs {yi}"
                                ))
                            elif plot_type == "box":
                                fig.add_trace(go.Box(
                                    y=df[yi],
                                    x=df[xi],
                                    marker_color=colors[color_idx],
                                    name=f"{xi} vs {yi}"
                                ))
                            elif plot_type == "violin":
                                fig.add_trace(go.Violin(
                                    y=df[yi],
                                    x=df[xi],
                                    marker_color=colors[color_idx],
                                    name=f"{xi} vs {yi}"
                                ))
                            elif plot_type == "scatter":
                                fig.add_trace(go.Scatter(
                                    x=df[xi],
                                    y=df[yi],
                                    mode="markers",
                                    marker=dict(color=colors[color_idx]),
                                    name=f"{xi} vs {yi}"
                                ))
                            color_idx = (color_idx + 1) % len(colors)
                    return fig

                for i, plot in enumerate(selected_plots):
                    with plot_containers[i]:
                        if plot == "Box Plot":
                            fig_box = plot_figure(None, "box")
                            fig_box.update_layout(
                                margin=dict(l=0, r=0, t=0, b=0),
                                xaxis_showgrid=False,
                                yaxis_showgrid=False,
                                legend_title_text="Variables"
                            )
                            with st.container(border=True):
                                st.subheader("Box Plot")
                                st.plotly_chart(fig_box, use_container_width=True)

                        elif plot == "Violin Plot":
                            fig_violin = plot_figure(None, "violin")
                            fig_violin.update_layout(
                                margin=dict(l=0, r=0, t=0, b=0),
                                xaxis_showgrid=False,
                                yaxis_showgrid=False,
                                legend_title_text="Variables"
                            )
                            with st.container(border=True):
                                st.subheader("Violin Plot")
                                st.plotly_chart(fig_violin, use_container_width=True)

                        elif plot == "Scatter Plot":
                            fig_scatter = plot_figure(None, "scatter")
                            fig_scatter.update_layout(
                                margin=dict(l=0, r=0, t=0, b=0),
                                xaxis_showgrid=False,
                                yaxis_showgrid=False,
                                legend_title_text="Variables"
                            )
                            with st.container(border=True):
                                st.subheader("Scatter Plot")
                                st.plotly_chart(fig_scatter, use_container_width=True)

                        elif plot == "Line Plot":
                            fig_line = plot_figure(None, "line")
                            fig_line.update_layout(
                                margin=dict(l=0, r=0, t=0, b=0),
                                xaxis_showgrid=False,
                                yaxis_showgrid=False,
                                legend_title_text="Variables"
                            )
                            with st.container(border=True):
                                st.subheader("Line Plot")
                                st.plotly_chart(fig_line, use_container_width=True)

def visualisasi_3d():
    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis = st.selectbox(
            label="X Axis",
            options=df.columns,
            index=None
        )

    with col2:
        y_axis = st.selectbox(
            label="Y Axis",
            options=df.columns,
            index=None
        )

    with col3:
        z_axis = st.selectbox(
            label="Z Axis",
            options=df.columns,
            index=None
        )

    if (x_axis is not None) and (y_axis is not None) and (z_axis is not None):
        if (x_axis == y_axis) or (x_axis == z_axis) or (y_axis == z_axis):
            st.error("X axis, Y axis, and Z axis cannot have the same values. Please select different columns.", icon="🚨")
        else:
            # Contour Plot
            fig_contour = go.Figure(data=go.Contour(z=df[z_axis], x=df[x_axis], y=df[y_axis]))
            fig_contour.update_traces(contours_coloring="heatmap")
            fig_contour.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis_showgrid=False,
                yaxis_showgrid=False
            )
            with st.container(border=True):
                st.subheader("Contour Plot")
                st.plotly_chart(fig_contour, use_container_width=True)

            # 3D Surface Plot
            x = np.array(df[x_axis])
            y = np.array(df[y_axis])
            z = np.array(df[z_axis])

            xi = np.linspace(x.min(), x.max(), 100)
            yi = np.linspace(y.min(), y.max(), 100)

            X, Y = np.meshgrid(xi, yi)
            Z = griddata((x, y), z, (X, Y), method="cubic")

            fig_surface = go.Figure(data=[go.Surface(x=xi, y=yi, z=Z)])
            fig_surface.update_layout(
                margin=dict(l=0, r=0, b=0, t=40)
            )
            with st.container(border=True):
                st.subheader("3D Surface Plot")
                st.plotly_chart(fig_surface, use_container_width=True)

def korelasi_pearson():
    corr_matrix = df.corr(method="pearson")

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
    ))
    fig_heatmap.update_layout(
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    with st.container(border=True):
        st.subheader("Korelasi Pearson")
        st.plotly_chart(fig_heatmap)

def clear_cache():
    load_from_mongo.clear()

def data():
    df_id = load_from_mongo(collection)
    st.write(df_id.shape)
    st.dataframe(df_id, use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["Create", "Update", "Delete"])

    with tab1:
        st.subheader("Create Data")
        num_cols = len(df.columns)
        num_rows = (num_cols // 5) + (1 if num_cols % 5 != 0 else 0)
        
        manual_data = []
        for row in range(num_rows):
            cols = st.columns(5)
            for col in range(5):
                col_idx = row * 5 + col
                if col_idx < num_cols:
                    col_name = df.columns[col_idx]
                    value = cols[col].number_input(f"{col_name}", format="%.6f")
                    manual_data.append(value)

        st.write("")

        if st.button("Create", type="primary"):
            if len(manual_data) == num_cols:
                new_data = {df.columns[i]: [manual_data[i]] for i in range(num_cols)}
                new_df = pd.DataFrame(new_data)
                store_to_mongo(new_df, collection, mode="append")
                st.success("Row created successfully!", icon="✅")

                df_id = load_from_mongo(collection)
                st.dataframe(df_id, use_container_width=True).data = df_id
                st.rerun()
            else:
                st.error("Please fill in all inputs.", icon="🚨")

        st.divider()

        st.subheader("Upload Data")
        if 'data_uploaded' not in st.session_state:
            st.session_state.data_uploaded = False

        upload_file = st.file_uploader("Upload Excel File", type=["xlsx"], label_visibility="collapsed")
        if upload_file and not st.session_state.data_uploaded:
            uploaded_df = pd.read_excel(upload_file)
            if uploaded_df.columns.to_list() == df.columns.to_list():
                store_to_mongo(uploaded_df, collection, mode="append")
                st.session_state.data_uploaded = True
                st.success("Data uploaded successfully!", icon="✅")
                st.rerun()
            else:
                st.error("The columns of the uploaded file do not match the existing data.", icon="🚨")

    # Clear the upload flag after rerun
    if st.session_state.data_uploaded:
        st.session_state.data_uploaded = False

    with tab2:
        st.subheader("Update Data")

        row_id = st.selectbox("Select ID", options=df_id["_id"].astype(str).tolist(), index=None)

        if row_id:
            selected_row = df_id[df_id["_id"] == ObjectId(row_id)].iloc[0]

            num_cols = len(df.columns)
            num_rows = (num_cols // 5) + (1 if num_cols % 5 != 0 else 0)

            update_data = {}
            for row in range(num_rows):
                cols = st.columns(5)
                for col in range(5):
                    col_idx = row * 5 + col
                    if col_idx < num_cols:
                        col_name = df.columns[col_idx]
                        value = cols[col].number_input(
                            label=col_name, 
                            value=None, 
                            format="%.6f",
                            key=f"update_{col_name}"
                        )
                        update_data[col_name] = value

            st.write("")

            if st.button("Update", type="primary"):
                update_data = {k: v for k, v in update_data.items() if v is not None}

                if update_data:
                    collection.update_one({"_id": ObjectId(row_id)}, {"$set": update_data})
                    st.success("Row updated successfully!", icon="✅")

                    df_id = load_from_mongo(collection)
                    st.dataframe(df_id, use_container_width=True).data = df_id
                    st.rerun()
                else:
                    st.error("No changes detected. Please modify at least one value.", icon="🚨")

    with tab3:
        st.subheader("Delete Data")

        row_ids_to_delete = st.multiselect("Select ID(s)", options=df_id["_id"].astype(str).tolist())

        if row_ids_to_delete:
            if st.button("Delete", type="primary"):
                object_ids = [ObjectId(row_id) for row_id in row_ids_to_delete]
                collection.delete_many({"_id": {"$in": object_ids}})

                if len(row_ids_to_delete) == 1:
                    st.success("Row deleted successfully!", icon="✅")
                else:
                    st.success("Rows deleted successfully!", icon="✅")

                df_id = load_from_mongo(collection)
                st.dataframe(df_id, use_container_width=True).data = df_id
                st.rerun()

# Main Content
if pages == "Visualisasi Otomatis":
    visualisasi_otomatis()
elif pages == "Visualisasi 2D":
    visualisasi_2d()
elif pages == "Visualisasi 3D":
    visualisasi_3d()
elif pages == "Korelasi Pearson":
    korelasi_pearson()
else:
    data()