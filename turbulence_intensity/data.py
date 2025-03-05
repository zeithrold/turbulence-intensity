import pandas as pd
import numpy as np
import uuid
import streamlit as st


def tag_uuid(
    data: pd.DataFrame,
    uuid_column_tag: str,
    relative_coordinate_column_tag_list: list[str],
):
    """按照相对坐标点添加UUID列。

    Args:
        data (pd.DataFrame): 数据框
        uuid_column_tag (str): UUID列名
        relative_coordinate_column_tag_list (list[str]): 相对坐标列名，如`['x', 'y']`，该列名必须维度大于等于2.

    Returns:
        pd.DataFrame: 添加UUID列后的数据框
    """

    def _applier(row: pd.Series):
        return uuid.uuid5(
            uuid.NAMESPACE_DNS,
            str(row[relative_coordinate_column_tag_list].to_numpy()),
        ).hex

    data[uuid_column_tag] = data.apply(_applier, axis=1)

    return data


def columns(df: pd.DataFrame):
    """获取数据框的列名。

    Args:
        df (pd.DataFrame): 数据框

    Returns:
        list[str]: 列名列表
    """
    return df.columns.tolist()


def velocity_mean(data: np.ndarray):
    """计算二范数平均速度，计算公式为 $N(a, b) = \\sqrt{\\frac{a_1^2 + a_2^2 + ...}{|A|}}$
    其中 |A| 为向量的维度，即 a 的长度。

    Args:
        data (np.ndarray): 二维数组，每行代表一个速度向量，若非二维数组则会抛出异常

    Returns:
        np.ndarray: 一维数组，每个元素代表对应速度向量的二范数
    """

    assert len(data.shape) == 2, "输入数据必须是二维数组"
    # 速度向量的维度的平方根，以保证速度向量的维度与公式定义的平均速度一致。
    additional_weight = np.sqrt(data.shape[1])
    return np.linalg.norm(data, 2, axis=1) / additional_weight


def distance(data: np.ndarray, center: np.ndarray):
    """计算数据点到中心点的欧氏距离

    Args:
        data (np.ndarray): 输入数据，要求是二维数组，其中每行代表一个数据点
        center (np.ndarray): 中心点，要求是一维数组

    Returns:
        np.ndarray: 一维数组，每个元素代表对应数据点到中心点的欧氏距离
    """
    assert len(data.shape) == 2, "输入数据必须是二维数组"
    assert len(center.shape) == 1, "中心点必须是一维数组"
    assert data.shape[1] == center.shape[0], "数据维度与中心点维度不匹配"

    return np.linalg.norm(data - center, 2, axis=1)


def distance_filterer(
    target_row: pd.Series, coordinate_column_tag: list[str], precision: float
):
    def _filterer(row: pd.Series):
        return (
            distance(
                row[coordinate_column_tag].to_numpy(),
                target_row[coordinate_column_tag].to_numpy(),
            )
            < precision
        )

    return _filterer


def mean_velocity_by_time(
    data: pd.DataFrame,
    uuid_column_tag: str,
    coordinate_column_tag_list: list[str],
    relative_coordinate_column_tag_list: list[str],
    velocity_column_tag_list: list[str],
    euclidean_mean_velocity_column_tag: str,
):
    """计算统一坐标点的时域欧式平均速度。具体而言，该程序首先对每个方向的速度进行二范数平均，然后对所有方向的平均速度进行均方平均。

    Args:
        data (pd.DataFrame): 数据框
        uuid_column_tag (str): UUID列名
        coordinate_column_tag_list (list[str]): 坐标列名，如`['x', 'y']`，该列名必须维度大于等于2.
        relative_coordinate_column_tag_list (list[str]): 相对坐标列名，如`['x', 'y']`，该列名必须维度大于等于2.
        velocity_column_tag_list (list[str]): 速度列名，如`['Vx', 'Vy', 'Vz']`
        euclidean_mean_velocity_tag (str): 欧式平均速度列名

    Returns:
        pd.DataFrame: 各方向的时域平均速度数据框，保留uuid_column_tag、euclidean_mean_velocity_tag、与各velocity_column_tag的数据列名，而数据内容是各方向的时域平均速度。
    """

    def _applier(group: pd.DataFrame):
        filtered_group = group[velocity_column_tag_list]
        velocity_data = filtered_group[velocity_column_tag_list].to_numpy()
        mean_sub_velocity_by_time = np.linalg.norm(velocity_data, 2, axis=0)
        assert velocity_data.shape[1] == len(mean_sub_velocity_by_time), (
            "数据长度不匹配"
        )
        result_dict = {
            key: value
            for key, value in zip(velocity_column_tag_list, mean_sub_velocity_by_time)
        }
        for key in coordinate_column_tag_list:
            result_dict[key] = group[key].iloc[0]
        for key in relative_coordinate_column_tag_list:
            result_dict[key] = group[key].iloc[0]
        return pd.Series(result_dict)

    result_dataframe = data.groupby(uuid_column_tag).apply(_applier)

    DIMENSION_WEIGHT = np.sqrt(len(velocity_column_tag_list))

    result_dataframe[euclidean_mean_velocity_column_tag] = (
        np.linalg.norm(result_dataframe[velocity_column_tag_list].to_numpy(), 2, axis=1)
        / DIMENSION_WEIGHT
    )

    result_dataframe[uuid_column_tag] = result_dataframe.index

    result_dataframe.reset_index(drop=True, inplace=True)

    return result_dataframe[
        [
            uuid_column_tag,
            euclidean_mean_velocity_column_tag,
            *coordinate_column_tag_list,
            *relative_coordinate_column_tag_list,
            *velocity_column_tag_list,
        ]
    ]


def mean_turbulent_intensity(
    data: pd.DataFrame,
    uuid_column_tag: str,
    velocity_column_tag_list: list[str],
    relative_coordinate_column_tag_list: list[str],
    euclidean_mean_velocity_column_tag: str,
    coordinate_column_tag_list: list[str],
):
    def _applier(group: pd.DataFrame):
        data_length = len(group)

        mean_data = group[velocity_column_tag_list].mean().to_numpy()
        total_data = group[velocity_column_tag_list].to_numpy()

        turbulent_mean_data = np.linalg.norm(
            total_data - mean_data, 2, axis=0
        ) / np.sqrt(data_length)

        euclidean_mean = np.linalg.norm(turbulent_mean_data, 2) / np.sqrt(
            len(turbulent_mean_data)
        )
        result_dict = {
            key: value
            for key, value in zip(velocity_column_tag_list, turbulent_mean_data)
        }
        result_dict[euclidean_mean_velocity_column_tag] = euclidean_mean
        for key in coordinate_column_tag_list:
            result_dict[key] = group[key].iloc[0]
        for key in relative_coordinate_column_tag_list:
            result_dict[key] = group[key].iloc[0]

        return pd.Series(result_dict)

    return data.groupby(uuid_column_tag).apply(_applier)


def mean_velocity_by_distance(
    data: pd.DataFrame,
    coordinate_column_tag_list: list[str],
    center_coordinate: np.ndarray,
    euclidean_mean_velocity_tag: str,
    bins: int = 10,
):
    """对每个点的时域欧式平均速度进行分组，分组的依据是到中心点的欧式距离。

    Args:
        data (pd.DataFrame): 数据框
        coordinate_column_tag_list (list[str]): 坐标列名，如`['x', 'y']，该列名必须维度大于等于2.
        center_coordinate (np.ndarray): 中心坐标，中心坐标的维度与顺序必须与`coordinate_column_tag_list`一致。
        euclidean_mean_velocity_tag (str): 欧式平均速度列名
        bins (int, optional): 分组数，默认为10

    Returns:
        pd.DataFrame: 新的数据框，包括`Min Velocity`, `Max Velocity`, `Mean Velocity`, `Std Velocity`列。
        分别代表每个分组的最小、最大、平均、标准差速度。
    """
    working_data = data.copy()

    working_data["Distance"] = np.linalg.norm(
        working_data[coordinate_column_tag_list].to_numpy() - center_coordinate,
        2,
        axis=1,
    )

    working_data["Distance Group"] = pd.cut(working_data["Distance"], bins=bins)

    group = working_data.groupby("Distance Group")

    euclidean_max_velocity = group[euclidean_mean_velocity_tag].max()
    euclidean_min_velocity = group[euclidean_mean_velocity_tag].min()

    euclidean_mean_velocity = group[euclidean_mean_velocity_tag].mean()
    euclidean_std_velocity = group[euclidean_mean_velocity_tag].std()

    return pd.DataFrame(
        {
            "Mean Distance": group["Distance"].mean(),
            "Min Velocity": euclidean_min_velocity,
            "Max Velocity": euclidean_max_velocity,
            "Mean Velocity": euclidean_mean_velocity,
            "Std Velocity": euclidean_std_velocity,
        }
    ).reset_index(drop=True)


def integral_scale(
    data: pd.DataFrame,
    relative_coordinate_column_tag_list: list[str],
    velocity_column_tag_list: list[str],
    timestamp_column_tag: str,
    max_diff_multiplier: int = 10,
):
    """计算积分尺度。

    Args:
        data (pd.DataFrame): 数据框
        relative_coordinate_column_tag_list (list[str]): 相对坐标列名，如`['x', 'y']`，该列名必须维度等于2.
        velocity_column_tag_list (list[str]): 速度列名，如`['Vx', 'Vy']`，该列名必须维度等于2，且与`relative_coordinate_column_tag_list`一一对应。
        timestamp_tag (str): 时间戳列名
        max_diff_multiplier (int, optional): 计算积分尺度过程中的最大差值倍数，默认为10，但该值若大于实际相对坐标的最大差值，则取实际相对坐标的最大差值。
    """
    assert len(relative_coordinate_column_tag_list) == len(velocity_column_tag_list), (
        "坐标列与速度列维度不匹配"
    )
    assert len(relative_coordinate_column_tag_list) == 2, "坐标列维度必须等于2"
    working_dataframe = data.copy()

    total_velocity_mean = (
        (working_dataframe[velocity_column_tag_list] ** 2).mean().to_numpy()
    )

    dataframe_list = []

    for i in range(1, max_diff_multiplier + 1):
        for coord_tag in relative_coordinate_column_tag_list:
            assert coord_tag in working_dataframe.columns, f"坐标列{coord_tag}不存在"

            copied_relative_coordinate_column_tag_list = (
                relative_coordinate_column_tag_list.copy()
            )
            copied_relative_coordinate_column_tag_list.remove(coord_tag)
            reversed_coord_tag = copied_relative_coordinate_column_tag_list[0]

            group = working_dataframe.groupby([timestamp_column_tag, coord_tag])

            result_array = (
                group[velocity_column_tag_list]
                .apply(lambda x: x.shift(-i) * x)
                .dropna()
                # .groupby(coord_tag)
                .mean()
                .to_numpy()
                / total_velocity_mean
            )

            result_dict = {
                key: value for key, value in zip(velocity_column_tag_list, result_array)
            }
            result_dict["Diff Multiplier"] = i
            result_dict["Coord Tag"] = reversed_coord_tag
            dataframe_list.append(result_dict)

    return pd.DataFrame(dataframe_list)


def relative_coordinate(
    data: pd.DataFrame, coordinate_column_tag_list: list[str], min_diff: float = 0.1
) -> np.ndarray:
    """获取相对坐标。分别在x、y方向上减去最小值，并按照每个轴的差值进行整数量化。

    **请务必注意！输入的数据必须经过矩形限制区域。部分工况下风扇区域采样点呈随机态，会严重妨碍相对坐标的计算。**

    Args:
        data (pd.DataFrame): 数据框
        coordinate_column_tag_list (list[str]): 坐标列名，如`['x', 'y']，该列名必须维度大于等于2.
        min_diff (float, optional): 最小差值，默认为0.1

    Returns:
        np.ndarray: 相对坐标参数，一共 (N, M) 维度，其中 M 为坐标列数，N 为数据长度。
    """

    array_list = []

    for tag in coordinate_column_tag_list:
        assert tag in data.columns, f"坐标列{tag}不存在"
        coordinate_min = data[tag].min()
        coordinate_unique_array = data[tag].unique()

        coordinate_unique_array.sort()

        coordinate_diff_array = np.array(
            [x for x in np.diff(coordinate_unique_array) if x > min_diff]
        )

        # 最大值抑制
        coordinate_diff_array.sort()

        coordinate_diff_array = coordinate_diff_array[:-3]

        mean_diff = abs(np.mean(coordinate_diff_array))
        std_diff = np.std(coordinate_diff_array)
        if std_diff / mean_diff > 0.1:
            st.warning(
                f"坐标差值的标准差过大{std_diff / mean_diff}，可能存在采样点不均匀的情况，请检查数据"
            )
        array_list.append(
            np.array((data[tag] - coordinate_min) / mean_diff, dtype=np.int32)
        )

    return np.array(array_list).T


def limit_area(
    data: pd.DataFrame, coordinate_column_tag_list: list[str], limit: list[list[float]]
):
    """以矩形限制数据区域。

    Args:
        data (pd.DataFrame): 数据框
        coordinate_column_tag_list (list[str]): 坐标列名，如`['x', 'y']，该列名必须维度大于等于2.
        limit (list[list[float]]): 限制区域，如`[[0, 0], [1, 1]]`，代表左下角坐标和右上角坐标，该坐标必须与`coordinate_column_tag_list`一致。

    Returns:
        pd.DataFrame: 除去限制区域外的数据框
    """
    x_min, y_max = limit[0]
    x_max, y_min = limit[1]
    return data[
        (data[coordinate_column_tag_list[0]] >= x_min)
        & (data[coordinate_column_tag_list[0]] <= x_max)
        & (data[coordinate_column_tag_list[1]] >= y_min)
        & (data[coordinate_column_tag_list[1]] <= y_max)
    ]


def merge_dataframes(
    dataframe_dict: dict[int, pd.DataFrame],
    coordinate_column_tag_list: list[str],
    timestamp_column_tag: str,
):
    """将多个数据框以坐标为基准按照时间轴合并。由于测量设备测定点可能不准确，使用`precision`参数来判定是否为同一点。
    若两点距离小于`precision`，则认为是同一点，当存在多个点时，随机抽取并发出警告。

    **请务必将数据框进行限制区域处理，否则可能会导致合并错误。**
    Args:
        dataframe_dict (dict[int, pd.DataFrame]): 数据框字典，键为时间戳，值为数据框
        coordinate_column_tag_list (list[str]): 坐标列名，如`['x', 'y']，该列名必须维度大于等于2.
        timestamp_column_tag (str): 时间戳列名

    Returns:
        pd.DataFrame: 合并后的数据框，新增`timestamp_tag`列，代表时间戳。
    """

    assert len(coordinate_column_tag_list) >= 2, "坐标列维度必须大于等于2"

    dataframe_list = []

    for key, value in dataframe_dict.items():
        value[timestamp_column_tag] = key
        dataframe_list.append(value)

    return pd.concat(dataframe_list).reset_index(drop=True)
