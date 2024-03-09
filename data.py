# 本文件当中所有DataFrame输入均为全图全时刻数据
import pandas as pd
from tqdm import tqdm


def transform_relative_coordinates(df: pd.DataFrame):
    """
    转换为相对坐标，替代原有浮点数

    Args:
        df: 原输入数据

    Returns:
        转换后数据，新增'x [relative]', 'y [relative]'两列，分别代表各轴相对坐标
        格式为:
        ```
        {
            "x_min": float, # x坐标最小值
            "x_diff": float, # x坐标间距
            "y_min": float, # y坐标最小值
            "y_diff": float, # y坐标间距
        }
        ```
    """
    opdf = df.copy()
    # 目标操作Column: 'x [mm]', 'y [mm]'
    # XY坐标均为等间距采样，先求出间距
    axies = ["x [mm]", "y [mm]"]
    transform_params: dict[str, float] = {}
    for axis in axies:
        sorted_df = opdf.sort_values(axis)
        diff_df = sorted_df.diff().dropna()  # 求差分
        diff_df = diff_df[diff_df[axis] != 0.0]  # 排除0值
        diff = diff_df[axis].mean()
        relative_axis = axis.replace("[mm]", "[relative]")
        transform_params[f"{axis[0]}_min"] = opdf[axis].min()
        transform_params[f"{axis[0]}_diff"] = diff
        # 转换为相对坐标
        opdf = opdf.assign(**{relative_axis: ((opdf[axis] - opdf[axis].min()) / diff)})
        # 四舍五入转换为整数
        opdf[relative_axis] = opdf[relative_axis].round().astype(int)
    return opdf, transform_params


def get_velocity_avg_by_group(df: pd.DataFrame):
    """计算平均速度随工况变化

    Args:
        df (pd.DataFrame): 原始输入数据，使用`['x [relative]', 'y [relative]', 'Group', 'Velocity u [m/s]', 'Velocity v [m/s]']`列
    """
    # 分别计算'Velocity u [m/s]', 'Velocity v [m/s]'随时间平均
    opdf = df.copy()
    means = (
        opdf.groupby(["Group", "x [relative]", "y [relative]"])
        .mean()[["Velocity u [m/s]", "Velocity v [m/s]"]]
        .reset_index()
    )
    means["Avg Velocity |V| [m/s]"] = (
        means["Velocity u [m/s]"] ** 2 + means["Velocity v [m/s]"] ** 2
    ) ** 0.5
    result = means.groupby("Group").mean()["Avg Velocity |V| [m/s]"].reset_index()
    return result


def get_absolute_coordinates(transform_params: dict[str, float], x: int, y: int):
    """
    获取相对坐标下的绝对坐标

    Args:
        transform_params: 转换参数
        x: 相对x坐标
        y: 相对y坐标

    Returns:
        转换后数据，以元组(tuple[float, float])形式返回
    """
    x_min = transform_params["x_min"]
    x_diff = transform_params["x_diff"]
    y_min = transform_params["y_min"]
    y_diff = transform_params["y_diff"]
    return x_min + x * x_diff, y_min + y * y_diff


def get_distance(df: pd.DataFrame, transform_params: dict[str, float], x: int, y: int):
    """
    获取相对坐标下的距离，使用欧氏距离，获取数据中`['x [mm]', 'y[mm]']`列

    Args:
        df: 原输入数据
        transform_params: 坐标转换参数
        x: 相对x坐标
        y: 相对y坐标

    Returns:
        转换后数据，新增'Distance'列，代表各点到目标点的距离
    """
    opdf = df.copy()
    abs_x, abs_y = get_absolute_coordinates(transform_params, x, y)
    opdf["Distance"] = (
        (opdf["x [mm]"] - abs_x) ** 2 + (opdf["y [mm]"] - abs_y) ** 2
    ) ** 0.5
    return opdf


def get_velocity_means_by_time(df: pd.DataFrame):
    """
    获取时域平均速度

    Args:
        df: 原输入数据，请确认已经转换为相对坐标，即已经使用`transform_relative_coordinates`处理过
            将使用
            ```
            [
                'x [relative]',
                'y [relative]',
                'Tick',
                'Group',
                'Velocity |V| [m/s]'
            ]
            ```
            列

    Returns:
        时域平均速度数
    """
    opdf = df.copy()
    mean_df = (
        opdf.groupby(["y [relative]", "x [relative]", "Group"]).mean().reset_index()
    )
    return mean_df


def transform_relative_velocity(df: pd.DataFrame):
    """
    获取相对速度

    Args:
        df: 原输入数据，请确认已经转换为相对坐标，即已经使用`transform_relative_coordinates`处理过，将使用`['x [relative]', 'y [relative]', 'Tick', 'Group', 'Velocity |V| [m/s]']`列

    Returns:
        相对速度数据，新增
        ```
        [
            'Velocity |V| [relative, m/s]',
            'Velocity u [relative, m/s]',
            'Velocity v [relative, m/s]'
        ]
        ```
        列，代表各点相对速度
    """
    opdf = df.copy()
    # 获取时域平均速度
    mean_df = get_velocity_means_by_time(opdf)
    # 仅获取相对速度
    mean_df = mean_df[
        [
            "x [relative]",
            "y [relative]",
            "Group",
            "Velocity |V| [m/s]",
            "Velocity u [m/s]",
            "Velocity v [m/s]",
        ]
    ]
    # 计算相对速度
    opdf = pd.merge(
        opdf,
        mean_df,
        on=["x [relative]", "y [relative]", "Group"],
        suffixes=("", "_mean"),
    )
    target_axis = ["Velocity |V| [m/s]", "Velocity u [m/s]", "Velocity v [m/s]"]
    for axis in target_axis:
        opdf[axis.replace("[m/s]", "[relative, m/s]")] = (
            opdf[axis] - opdf[f"{axis}_mean"]
        )
    # 然后就可以删掉多余的列了
    opdf = opdf.drop(columns=[f"{axis}_mean" for axis in target_axis])
    return opdf


def get_rms_velocity(df: pd.DataFrame):
    """
    计算时域均方速度

    Args:
        df: 原输入数据，请确认已经转换为相对坐标与相对速度，即已经使用`transform_relative_coordinates、`transform_relative_velocity`处理过，将使用`['x [relative]', 'y [relative]', 'Group', 'Velocity u [relative, m/s]', 'Velocity v [relative, m/s]']`列

    Returns:
        时域均方速度数据
        转换后的数据新增
        ```
        [
            'RMS Velocity u [m/s]',
            'RMS Velocity v [m/s]',
            'RMS Velocity |V| [m/s]'
        ]
        ```
        列，代表各点时域均方速度
    """
    opdf = df.copy()
    target_axis = ["Velocity u [relative, m/s]", "Velocity v [relative, m/s]"]
    for axis in target_axis:
        opdf[f'RMS {axis.replace("[relative, m/s]", "[m/s]")}'] = opdf.groupby(
            ["x [relative]", "y [relative]", "Group"]
        )[axis].transform(lambda x: (x**2).mean() ** 0.5)
    opdf[f"RMS Velocity |V| [m/s]"] = (
        (opdf[f"RMS Velocity u [m/s]"] ** 2 + opdf[f"RMS Velocity v [m/s]"] ** 2) / 2
    ) ** 0.5
    opdf = opdf.reset_index(drop=True)
    return opdf


def get_correlation_coefficient(df: pd.DataFrame, k_range: tuple[int, int]):
    """
    计算相关系数
    分别计算x,y坐标x,y速度的相关系数，计算公式（假设是x坐标方向的x方向速度）为
    $$
    R_{uu}(\\Delta x) = \\frac{<u_f(x, y, t)u_f(x + \\Delta x, y, t)>}{<u_fx(x, y, t)>}
    $$
    相当于\\Delta x之间的两个点乘积的时域平均除全空间的时域平均


    Args:
        df: 原输入数据，请确认已经转换为相对坐标与相对速度，即已经使用`transform_relative_coordinates`、`transform_relative_velocity`处理过，将使用数据中的`['x [relative]', 'y [relative]', 'Group', 'Velocity u [relative, m/s]', 'Velocity v [relative, m/s]']`列
        k_range: 计算相关系数中，坐标单位间隔的范围

    Returns:
        相关系数数据，是一个包含了Group, k, x_u, y_u, x_v, y_v的DataFrame
    """
    opdf = df.copy()
    group_mean = (
        (opdf**2)
        .groupby("Group")
        .mean()[["Velocity u [relative, m/s]", "Velocity v [relative, m/s]"]]
        .reset_index()
    )
    result = pd.DataFrame(columns=["Group", "k", "x_u", "y_u", "x_v", "y_v"])
    for k in tqdm(range(*k_range)):
        # 使用DataFrame.groupby().shift()来实现滑动窗口
        # 由于相对坐标是整数递增，所以可以直接使用shift
        # 对Group、y、Tick进行分组，即可以对x轴的方向求xy方向的相关系数
        # 然后对y [relative]和Tick求均值
        df_list = []
        for axis in ["y", "x"]:
            reversed_axis = "x" if axis == "y" else "y"
            target_column = f"{axis} [relative]"
            r = (
                opdf.groupby(["Group", target_column, "Tick"])[
                    ["Velocity u [relative, m/s]", "Velocity v [relative, m/s]"]
                ]
                .apply(lambda x: x.shift(-k) * x)
                .reset_index()
                .drop(["level_3", target_column, "Tick"], axis=1)
                .dropna()
                .groupby("Group")
                .mean()
                .reset_index()
            )
            r[f"{reversed_axis}_u"] = (
                r["Velocity u [relative, m/s]"]
                / group_mean["Velocity u [relative, m/s]"]
            )
            r[f"{reversed_axis}_v"] = (
                r["Velocity v [relative, m/s]"]
                / group_mean["Velocity v [relative, m/s]"]
            )
            r = r[[f"{reversed_axis}_u", f"{reversed_axis}_v", "Group"]]
            df_list.append(r)
        # 将两个方向的相关系数合并
        k_result = pd.merge(df_list[0], df_list[1], on="Group")
        k_result["k"] = k
        result = pd.concat([result, k_result])
    return result

def get_velocity_rms_by_group(df: pd.DataFrame):
    """将时域均方速度按工况分组取平均

    Args:
        df (pd.DataFrame): 原始输入数据，使用`['Group', 'RMS Velocity |V| [m/s]']`列
    """
    opdf = df.copy()
    return opdf.groupby('Group').mean()["RMS Velocity |V| [m/s]"].reset_index()

def get_velocity_rms_by_radius(df: pd.DataFrame, convert_params: dict, x: int, y: int):
    """将时域均方速度按半径分组取平均

    Args:
        df (pd.DataFrame): 原始输入数据，使用`['x [relative]', 'y [relative]', 'RMS Velocity |V| [m/s]']`列
        convert_params: 转换参数
        x: 相对x坐标
        y: 相对y坐标
        min_error: 最小误差
    """
    opdf = df.copy()
    abs_x, abs_y = get_absolute_coordinates(convert_params, x, y)
    opdf["Radius"] = (
        (opdf["x [mm]"] - abs_x) ** 2 + (opdf["y [mm]"] - abs_y) ** 2
    ) ** 0.5
    # 以半径区间分组
    opdf["Radius Group"] = pd.cut(opdf["Radius"], bins=10)
    return opdf.groupby(["Group", "Radius Group"]).mean()["RMS Velocity |V| [m/s]"].reset_index()
