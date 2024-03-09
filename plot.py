import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# from data import get_velocity_means_by_radius_group


def plot_heatmap(df: pd.DataFrame, column: str, title: str):
    """
    绘制热力图
    
    Args:
        df: 输入数据，需要使用['x [relative]', 'y[relative]']列
        column: 需要绘制的列名
    
    Returns:
        绘制结果
    """
    opdf = df.copy()
    groups = opdf["Group"].unique()
    images = []
    for group in groups:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        group_df = opdf[opdf["Group"] == group]
        # 计算时域平均
        group_df = (
            group_df.groupby(["y [relative]", "x [relative]"]).mean().reset_index()
        )
        # 绘制热力图
        ax.scatter(
            group_df["x [relative]"],
            group_df["y [relative]"],
            c=group_df[column],
            cmap="viridis",
        )
        ax.set_title(f"{title} (Group {group})")
        # 设置热力图参考轴
        ax.set_xlabel("x [relative]")
        ax.set_ylabel("y [relative]")
        cbar = fig.colorbar(ax.collections[0], ax=ax, orientation="vertical")
        cbar.set_label(column)
        fig.canvas.draw()
        # 设置颜色条
        images.append(
            Image.frombytes(
                "RGBA", fig.canvas.get_width_height(), bytes(fig.canvas.buffer_rgba())
            )
        )
    return images

def plot_velocity_heatmap(df: pd.DataFrame):
    """
    绘制时域平均速度热力图，使用数据中的 ['x [relative]', 'y[relative]', 'Tick', 'Group', 'Velocity |V| [m/s]'] 列
    
    Args:
        df: 输入数据，使用['x [relative]', 'y[relative]', 'Tick', 'Group', 'Velocity |V| [m/s]']列
    
    
    Returns:
        绘制结果
    """
    return plot_heatmap(df, "Velocity |V| [m/s]", "Velocity Heatmap")


def plot_velocity_rms_heatmap(df: pd.DataFrame):
    """
    绘制时域均方平均速度热力图，使用数据中的 ['x [relative]', 'y[relative]', 'Tick', 'Group', 'RMS Velocity |V| [m/s]'] 列
    
    Args:
        df: 输入数据，使用['x [relative]', 'y[relative]', 'Tick', 'Group', 'Velocity |V| [m/s]']列
    
    
    Returns:
        绘制结果
    """
    return plot_heatmap(df, "RMS Velocity |V| [m/s]", "RMS Velocity Heatmap")


def plot_velocity_avg(df: pd.DataFrame):
    """"
    绘制湍流场平均速度随工况变化.

    Args:
        df: 输入数据，使用['Group', 'Avg Velocity |V| [m/s]']列
    """
    opdf = df.copy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制线状图，点画更大一点
    ax.plot(opdf["Group"], opdf["Avg Velocity |V| [m/s]"], marker="o")
    ax.set_title("Velocity Means by Group")
    ax.set_xlabel("Group")
    ax.set_ylabel("Avg Velocity |V| [m/s]")
    return fig

    
def plot_velocity_rms(df: pd.DataFrame):
    opdf = df.copy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制线状图，点画更大一点
    ax.plot(opdf["Group"], opdf["RMS Velocity |V| [m/s]"], marker="o")
    ax.set_title("Velocity Means by Group")
    ax.set_xlabel("Group")
    ax.set_ylabel("RMS Velocity |V| [m/s]")
    return fig

def plot_velocity_rms_means_by_radius_group(df: pd.DataFrame):
    """
    绘制在不同中心距离半径组下时域速度平均值的平均值
    @param df 经过get_velocity_means_by_radius_group处理后的数据，
        使用数据中的
        ['Radius Group', 'Group', 'Velocity |V| [m/s]'] 列
    @return 绘制结果
    """
    opdf = df.copy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    opdf["Radius Group [left]"] = opdf["Radius Group"].apply(
        lambda x: x.left
    ).astype(float)
    opdf = opdf.sort_values("Radius Group [left]")
    groups = opdf["Group"].unique()
    for group in groups:
        group_df = opdf[opdf["Group"] == group]
        ax.plot(
            group_df["Radius Group [left]"],
            group_df["RMS Velocity |V| [m/s]"],
            marker="o",
            label=f"Group {group}",
        )
    ax.set_title("RMS Velocity Means by Radius Group")
    ax.set_xlabel("Radius Group")
    ax.set_ylabel("RMS Velocity |V| [m/s]")
    ax.legend()
    return fig

def plot_correlation_coefficient(df: pd.DataFrame):
    """
    绘制相关系数图
    @param df 经过get_correlation_coefficient处理后的数据，
        使用数据中的
        [Group, k, x_u, y_u, x_v, y_v] 列
    @return 绘制结果
    """
    opdf = df.copy()
    groups = opdf["Group"].unique()
    images = []
    for group in groups:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        group_df = opdf[opdf["Group"] == group]
        ax.plot(group_df["k"], group_df["x_u"], label="x_u")
        ax.plot(group_df["k"], group_df["y_u"], label="y_u")
        ax.plot(group_df["k"], group_df["x_v"], label="x_v")
        ax.plot(group_df["k"], group_df["y_v"], label="y_v")
        ax.set_title(f"Correlation Coefficient (Group {group})")
        ax.set_xlabel("k")
        ax.set_ylabel("Correlation Coefficient")
        ax.legend()
        fig.canvas.draw()
        images.append(
            Image.frombytes(
                "RGBA", fig.canvas.get_width_height(), bytes(fig.canvas.buffer_rgba())
            )
        )
    return images