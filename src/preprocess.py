import re
import pandas as pd


def replace_letters(series: pd.Series) -> pd.Series:
    series = series.str.replace("a", "а")
    series = series.str.replace("h", "н")
    series = series.str.replace("k", "к")
    series = series.str.replace("b", "в")
    series = series.str.replace("c", "с")
    series = series.str.replace("o", "о")
    series = series.str.replace("p", "р")
    series = series.str.replace("t", "т")
    series = series.str.replace("x", "х")
    series = series.str.replace("y", "у")
    series = series.str.replace("e", "е")
    series = series.str.replace("m", "м")
    return series


def replace_nan(series: pd.Series) -> pd.Series:
    series = series.str.replace("nan", " ")
    series = series.str.replace("null", " ")
    return series


def preprocess_text(series: pd.Series) -> pd.Series:
    series = series.str.lower()
    series = replace_nan(series)
    # series = replace_letters(series)
    series = series.apply(lambda x: re.sub(r"[ ]+", " ", x))
    return series