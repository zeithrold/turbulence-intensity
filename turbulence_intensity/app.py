import streamlit as st
from zipfile import ZipFile
from tempfile import TemporaryDirectory
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import data
import numpy as np
from scipy.interpolate import griddata

st.title("Turbulence Intensity Calculator")

file_upload = st.file_uploader(
    "上传一个包含风速数据的zip文件，风速数据应当为csv格式", type="zip"
)

preview_tab, parameter_tab = st.tabs(["预览数据", "参数设置"])


def show_mean_by_time_3d_surface_plot(
    data: pd.DataFrame, relative_coord_column_tags: list[str], title: str
):
    x = data[relative_coord_column_tags[0]].values
    y = data[relative_coord_column_tags[1]].values
    z = data["Euclidean Velocity [mm/s]"].values

    # Create a regular grid to interpolate the data
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    # Interpolate the data
    zi_grid = griddata((x, y), z, (xi_grid, yi_grid), method="cubic")

    # 创建3D Surface Plot
    fig = go.Figure(
        data=[
            go.Surface(
                x=xi,
                y=yi,
                z=zi_grid,
                colorscale="Viridis",
                colorbar=dict(title="Euclidean Velocity [mm/s]"),
            )
        ]
    )

    # 设置 Label
    fig.update_layout(
        scene=dict(
            xaxis_title=relative_coord_column_tags[0],
            yaxis_title=relative_coord_column_tags[1],
            zaxis_title="Euclidean Velocity [mm/s]",
        ),
        title=title,
    )

    return fig


def show_itegral_scale_plot(data: pd.DataFrame, velocity_column_tag_list: list[str]):
    # 现在有如下数据：
    # "Coord Tag"、"Diff Multiplier" 与 velocity_column_tag_list提供的列名
    # 想要以"Diff Multiplier"为x轴，以velocity_column_tag_list提供的列名与"Coord Tag"的组合为y轴
    # 画出折线图
    # Create an empty figure
    fig = go.Figure()

    # For each velocity column, add a trace for each coordinate tag
    for velocity_col in velocity_column_tag_list:
        # Get unique coordinate tags
        coord_tags = data["Coord Tag"].unique()

        for coord_tag in coord_tags:
            # Filter data for this coordinate tag
            filtered_data = data[data["Coord Tag"] == coord_tag]

            # Add trace for this velocity column and coordinate tag
            fig.add_trace(
                go.Scatter(
                    x=filtered_data["Diff Multiplier"],
                    y=filtered_data[velocity_col],
                    mode="lines+markers",
                    name=f"{velocity_col} - {coord_tag}",
                )
            )

    # Update layout
    fig.update_layout(
        title="Integral Scale Analysis",
        xaxis_title="Difference Multiplier",
        yaxis_title="Correlation",
        hovermode="x unified",
        legend_title="Velocity Components",
    )

    return fig
    pass


def show_mean_velocity_by_distance_plot(data: pd.DataFrame):
    # Create figure for mean velocity by distance
    fig = go.Figure()

    # Add min to max velocity range as a filled area
    fig.add_trace(
        go.Scatter(
            x=data["Mean Distance"],
            y=data["Max Velocity"],
            mode="lines",
            name="Max Velocity",
            line=dict(width=0.5, color="rgba(0,100,80,0.5)"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data["Mean Distance"],
            y=data["Min Velocity"],
            mode="lines",
            name="Min Velocity",
            line=dict(width=0.5, color="rgba(0,100,80,0.5)"),
            fill="tonexty",
            fillcolor="rgba(0,176,246,0.2)",
        )
    )

    # Add mean velocity line
    fig.add_trace(
        go.Scatter(
            x=data["Mean Distance"],
            y=data["Mean Velocity"],
            mode="lines+markers",
            name="Mean Velocity",
            line=dict(color="rgb(0,100,80)", width=2),
        )
    )

    # Add standard deviation as error bars
    fig.add_trace(
        go.Scatter(
            x=data["Mean Distance"],
            y=data["Mean Velocity"] + data["Std Velocity"],
            mode="lines",
            line=dict(width=0, color="rgba(0,100,80,0.2)"),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data["Mean Distance"],
            y=data["Mean Velocity"] - data["Std Velocity"],
            mode="lines",
            line=dict(width=0, color="rgba(0,100,80,0.2)"),
            fill="tonexty",
            fillcolor="rgba(0,176,246,0.2)",
            showlegend=False,
        )
    )

    # Update layout
    fig.update_layout(
        title="Velocity Distribution by Distance",
        xaxis_title="Mean Distance",
        yaxis_title="Velocity (m/s)",
        hovermode="x",
    )

    return fig


if file_upload:
    st.info(
        "请务必在预览数据后 **明确指定计算数据的四至范围，避免风扇区域的采样混乱导致数据分析有误！**",
        icon="🚨",
    )

    with ZipFile(file_upload) as zip_file:
        file_names = zip_file.namelist()
        csv_files = [
            file_name for file_name in file_names if file_name.endswith(".csv")
        ]
        if not csv_files:
            st.error("zip文件中没有csv文件")

        preview_dataframe = pd.read_csv(zip_file.open(csv_files[0]))

        columns = preview_dataframe.columns

        dataframe_dict = {key: pd.read_csv(zip_file.open(key)) for key in csv_files}

    with preview_tab:
        coord_column_tags = st.multiselect(
            "选择x和y坐标的列名",
            columns,
            max_selections=2,
            key="coord_column_tags_preview",
        )
        preview_button = st.button("预览数据")
        if preview_button:
            assert len(coord_column_tags) == 2, "请选择两个列名"
            fig = px.scatter(
                preview_dataframe,
                x=coord_column_tags[0],
                y=coord_column_tags[1],
            )
            st.plotly_chart(fig)

    with parameter_tab:
        max_diff_multiplier = st.number_input("最大差分乘数", value=10, step=1)

        total_multiplier = st.number_input("总乘数（用于转换单位）", value=1000, step=1)

        st.info("下面的坐标参数均是在适用于转换单位的情况下的坐标")

        left_up_x_tab, left_up_y_tab = st.columns(2)

        left_up_x = left_up_x_tab.number_input("左上角 `x` 坐标", value=-25)
        left_up_y = left_up_y_tab.number_input("左上角 `y` 坐标", value=25)

        right_down_x_tab, right_down_y_tab = st.columns(2)

        right_down_x = right_down_x_tab.number_input("右下角 `x` 坐标", value=25)
        right_down_y = right_down_y_tab.number_input("右下角 `y` 坐标", value=-25)

        centre_x_tab, centre_y_tab = st.columns(2)

        centre_x = centre_x_tab.number_input("中心 `x` 坐标", value=0)
        centre_y = centre_y_tab.number_input("中心 `y` 坐标", value=0)

        mean_velocity_group_bins = st.number_input("平均速度分组数", value=10, step=1)

        coord_column_tags = st.multiselect(
            "选择 `x` 和 `y` 坐标的列名",
            columns,
            max_selections=2,
            key="coord_column_tags_parameter",
        )

        velocity_column_tags = st.multiselect("选择速度列名", columns, max_selections=2)

        calculate_button = st.button("计算", type="primary")

        if calculate_button:
            dataframe_dict = {
                Path(key).name.replace(".csv", ""): data.limit_area(
                    value * total_multiplier,
                    coordinate_column_tag_list=coord_column_tags,
                    limit=[[left_up_x, left_up_y], [right_down_x, right_down_y]],
                )
                for key, value in dataframe_dict.items()
            }

            merged_dataframe = data.merge_dataframes(
                dataframe_dict, coord_column_tags, "Timestamp"
            )

            relative_coord_column_tags = [x + " [relative]" for x in coord_column_tags]

            relative_coord_tagged_array = data.relative_coordinate(
                merged_dataframe, coord_column_tags
            )

            merged_dataframe[relative_coord_column_tags] = relative_coord_tagged_array

            merged_dataframe = data.tag_uuid(
                merged_dataframe, "UUID", coord_column_tags
            )

            st.write(merged_dataframe)

            mean_velocity_by_time_result = data.mean_velocity_by_time(
                merged_dataframe,
                "UUID",
                coord_column_tags,
                relative_coord_column_tags,
                velocity_column_tags,
                "Euclidean Velocity [mm/s]",
            )

            fig_mean_by_time = show_mean_by_time_3d_surface_plot(
                mean_velocity_by_time_result,
                relative_coord_column_tags,
                "Mean Velocity by Time (Simple Average)",
            )
            st.plotly_chart(fig_mean_by_time, key="mean_by_time")

            mean_turbulent_intensity_result = data.mean_turbulent_intensity(
                merged_dataframe,
                "UUID",
                velocity_column_tags,
                relative_coord_column_tags,
                "Euclidean Velocity [mm/s]",
            )

            fig_turbulent_intensity = show_mean_by_time_3d_surface_plot(
                mean_turbulent_intensity_result,
                relative_coord_column_tags,
                "Turbulent Intensity",
            )

            st.plotly_chart(fig_turbulent_intensity, key="turbulent_intensity")

            mean_velocity_by_distance_result = data.mean_velocity_by_distance(
                mean_velocity_by_time_result,
                coord_column_tags,
                np.array([centre_x, centre_y]),
                "Euclidean Velocity [mm/s]",
                mean_velocity_group_bins,
            )

            fig_mean_velocity_by_distance = show_mean_velocity_by_distance_plot(
                mean_velocity_by_distance_result
            )

            st.plotly_chart(fig_mean_velocity_by_distance)

            integral_scale_result = data.integral_scale(
                merged_dataframe,
                relative_coord_column_tags,
                velocity_column_tags,
                "Timestamp",
                max_diff_multiplier=max_diff_multiplier,
            )

            fig_integral_scale = show_itegral_scale_plot(
                integral_scale_result, velocity_column_tags
            )

            st.plotly_chart(fig_integral_scale)

            merged_dataframe_csv = merged_dataframe.to_csv(index=False).encode("utf-8")
            mean_velocity_by_time_csv = mean_velocity_by_time_result.to_csv(
                index=False
            ).encode("utf-8")
            mean_velocity_by_distance_csv = mean_velocity_by_distance_result.to_csv(
                index=False
            ).encode("utf-8")
            integral_scale_csv = integral_scale_result.to_csv(index=False).encode(
                "utf-8"
            )
            mean_turbulent_intensity_result_csv = (
                mean_turbulent_intensity_result.to_csv(index=False).encode("utf-8")
            )

            with TemporaryDirectory() as temp_dir:
                with ZipFile(Path(temp_dir) / "result.zip", "w") as zip_file:
                    zip_file.writestr("merged_dataframe.csv", merged_dataframe_csv)
                    zip_file.writestr(
                        "mean_velocity_by_time.csv", mean_velocity_by_time_csv
                    )
                    zip_file.writestr(
                        "mean_velocity_by_distance.csv", mean_velocity_by_distance_csv
                    )
                    zip_file.writestr("integral_scale.csv", integral_scale_csv)
                    zip_file.writestr(
                        "mean_turbulent_intensity_result.csv",
                        mean_turbulent_intensity_result_csv,
                    )

                file_path = Path(temp_dir) / "result.zip"

                st.download_button(
                    "下载结果",
                    data=file_path.read_bytes(),
                    file_name="result.zip",
                    key="download_button",
                )
