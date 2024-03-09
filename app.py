import gradio as gr
import plot
import data
import file
from tempfile import TemporaryDirectory, NamedTemporaryFile
from zipfile import ZipFile
from loguru import logger


def preview_turbulence_field(f: str):
    df = file.get_dataframe(f)
    opdf, _ = data.transform_relative_coordinates(df)
    fig = plot.plot_velocity_heatmap(opdf)
    return fig


def analyze(
    f: str,
    center_x: int,
    center_y: int,
    preserve_counts: int,
    max_delta_k: int,
    progress=gr.Progress(track_tqdm=True),
):
    """
    需要返回的数据：
    - 遄流场平均速度
    - 湍流脉动速度均方根
    - 湍流强度变化
    - 脉动速度空间相关
    - 数据文件
    """
    df = file.get_dataframe(f)
    logger.debug("开始分析")
    opdf, transform_params = data.transform_relative_coordinates(df)
    if preserve_counts:
        opdf = opdf[opdf["Tick"] <= opdf["Tick"].min() + preserve_counts]
    if center_x == -1 or center_y == -1:
        raise gr.Error("请先确认中心点相对坐标后再进行分析")
    logger.debug("正在计算中心点距离")
    opdf = data.get_distance(opdf, transform_params, center_x, center_y)
    logger.debug("正在计算相对速度")
    opdf = data.transform_relative_velocity(opdf)
    logger.debug("正在计算均方根速度")
    opdf = data.get_rms_velocity(opdf)
    logger.debug("正在计算平均速度")
    avg_by_group_df = data.get_velocity_avg_by_group(opdf)
    avg_plot = plot.plot_velocity_avg(avg_by_group_df)
    rms_heatmap_plot = plot.plot_velocity_rms_heatmap(opdf)
    logger.debug("正在计算均方根变化")
    rms_by_group_df = data.get_velocity_rms_by_group(opdf)
    rms_by_group_plot = plot.plot_velocity_rms(rms_by_group_df)
    rms_by_radius_df = data.get_velocity_rms_by_radius(
        opdf, transform_params, center_x, center_y
    )
    rms_by_radius_plot = plot.plot_velocity_rms_means_by_radius_group(rms_by_radius_df)
    logger.debug("正在计算相关系数")
    coefficient_df = data.get_correlation_coefficient(opdf, (1, max_delta_k + 1))
    coefficient_plot = plot.plot_correlation_coefficient(coefficient_df)
    logger.debug("正在保存数据")
    temp_dir = TemporaryDirectory(prefix="turbulence_analysis_")
    avg_by_group_df.to_csv(temp_dir.name + "/avg_by_group.csv", index=False)
    rms_by_group_df.to_csv(temp_dir.name + "/rms_by_group.csv", index=False)
    rms_by_radius_df.to_csv(temp_dir.name + "/rms_by_radius.csv", index=False)
    coefficient_df.to_csv(temp_dir.name + "/correlation_coefficient.csv", index=False)
    temp_zip = NamedTemporaryFile(suffix=".zip", delete=False)
    with ZipFile(temp_zip.name, "w") as z:
        z.write(temp_dir.name + "/avg_by_group.csv", "avg_by_group.csv")
        z.write(temp_dir.name + "/rms_by_group.csv", "rms_by_group.csv")
        z.write(temp_dir.name + "/rms_by_radius.csv", "rms_by_radius.csv")
        z.write(
            temp_dir.name + "/correlation_coefficient.csv",
            "correlation_coefficient.csv",
        )
    temp_zip.close()
    return (
        avg_plot,
        rms_heatmap_plot,
        rms_by_group_plot,
        rms_by_radius_plot,
        coefficient_plot,
        temp_zip.name,
    )


with gr.Blocks() as demo:
    title = gr.Markdown("# 湍流分析相关计算套件")
    with gr.Column():
        input_title = gr.Markdown("## 输入")
        with gr.Tab("文件上传"):
            input_file = gr.File(
                label="上传文件", file_count="single", file_types=[".zip"]
            )
            with gr.Accordion(label="文件上传说明(重要)"):
                gr.Markdown(
                    """
                            **请严格按照文件要求上传，否则容易出现解析失败的问题**
                            - 具体的湍流数据文件格式为csv，均为瞬时场，时间信息命名在文件名中。例如，文件名为`40001.csv`，代表Tick为40001的瞬时场。
                            - 压缩包的第一级目录为工况组（如转速），例如`400/40001.csv`，代表转速为400的第40001刻瞬时场。
                            - 当使用压缩工具打开压缩包时，若只有一个文件夹，请仔细确认该文件夹是否为工况组，若不是，请将文件夹内的文件移到压缩包的根目录。
                            - 除了第一级目录与文件名，其他目录结构不做要求，也就是说，中间可以有多级目录，但是不会对解析造成影响。
                            - 上传完成后，**请务必确认在参数调整选项卡中的参数**，先点击`预览遄流场`，根据预览遄流场的相对坐标确定中心点，然后再点击`开始分析`。
                            """
                )
            with gr.Row():
                preview_button = gr.Button(value="预览遄流场", variant="secondary")
                analyze_button = gr.Button(value="开始分析", variant="primary")
        with gr.Tab("参数调整"):
            preserve_counts = gr.Number(
                label="保留刻数",
                minimum=0,
                value=0,
                info="保留的刻数，若为0，则保留全部刻数，该选项用于舍去后期数据不良的情况",
            )
            max_delta_k = gr.Number(
                label="最大相关系数计算距离",
                minimum=1,
                value=10,
                info="最大相关系数计算距离，用于计算脉动速度空间相关",
            )
            with gr.Group():
                center_x = gr.Number(
                    label="中心点x相对坐标",
                    minimum=-1,
                    value=-1,
                    info="中心点x坐标，相对坐标，若想要预览遄流场，请输入-1",
                )
                center_y = gr.Number(
                    label="中心点y相对坐标",
                    minimum=-1,
                    value=-1,
                    info="中心点y坐标，相对坐标，若想要预览遄流场，请输入-1",
                )

    with gr.Column():
        output_title = gr.Markdown("## 输出")
        with gr.Tab("遄流场预览"):
            preview_plot = gr.Gallery(preview=True)
        with gr.Tab("遄流场平均速度"):
            avg_plot = gr.Plot()
        with gr.Tab("湍流脉动速度均方根"):
            rms_plot = gr.Gallery(preview=True)
        with gr.Tab("湍流强度变化"):
            turbulance_by_group = gr.Plot()
            turbulance_by_radius = gr.Plot()
        with gr.Tab("脉动速度空间相关"):
            turbulance_correlation = gr.Gallery()
        with gr.Tab("数据文件"):
            output_file = gr.File(label="下载结果")

    preview_button.click(
        preview_turbulence_field, inputs=input_file, outputs=preview_plot
    ).then(lambda: gr.Info("请务必在参数中确认中心点相对坐标后再点击开始分析！"))
    analyze_button.click(
        analyze,
        inputs=[input_file, center_x, center_y, preserve_counts, max_delta_k],
        outputs=[
            avg_plot,
            rms_plot,
            turbulance_by_group,
            turbulance_by_radius,
            turbulance_correlation,
            output_file,
        ],
    )

demo.launch()
