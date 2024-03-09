from zipfile import ZipFile, ZipInfo
from pathlib import PurePath
import pandas as pd

sheet_suffixes = (".xls", ".csv", "xlsx")


def get_sheet_groups(filelist: list[ZipInfo]):
    filename_group = [PurePath(f.filename).parts[0] for f in filelist]
    return list(set(filename_group))


def get_dataframe(zipfile: ZipFile | str, tick_limit: None | int = None):
    if isinstance(zipfile, str):
        zipfile = ZipFile(zipfile)
    filelist = zipfile.filelist
    filtered_filelist = [f for f in filelist if f.filename.endswith(sheet_suffixes)]
    sheet_groups = get_sheet_groups(filtered_filelist)
    dataframe_groups = []
    for group in sheet_groups:
        target_filelist = [f for f in filtered_filelist if f.filename.startswith(group)]
        dataframe_list = []
        for f in target_filelist:
            filename = f.filename
            tick = PurePath(filename).stem
            df = (
                pd.read_excel(zipfile.open(f.filename))
                if f.filename.endswith(".xls")
                else pd.read_csv(zipfile.open(f.filename), sep=";")
            )
            df["Tick"] = tick
            df["Tick"] = df["Tick"].astype("int")
            dataframe_list.append(df)
        composed_dataframe = pd.concat(dataframe_list)
        composed_dataframe["Group"] = group
        dataframe_groups.append(composed_dataframe)
    result = pd.concat(dataframe_groups)
    if tick_limit:
        result = result[(result["Tick"]) <= (result["Tick"].min() + tick_limit)]
    result["Group"] = result["Group"].astype("int")
    return result
