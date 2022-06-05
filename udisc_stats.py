import pandas as pd
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt

seg_cols = [
    "PlayerName",
    "CourseName",
    "LayoutNameAdj",
]

def print_segments(df):
    print(f"Players: {list(df.PlayerName.unique())}")
    print(f"Courses: {list(df.CourseName.unique())}")
    print(f"Layouts: {list(df.LayoutNameAdj.unique())}")

def preprocess(df):
    df = df.copy()
    df.Date = pd.to_datetime(df.Date)

    min_pavers_date = df[df.LayoutName == "Paver Tees"].Date.min()
    df["LayoutNameAdj"] = df.LayoutName
    df.loc[
        (df.CourseName == "Bryan Park") &
        (df.LayoutName == "Main") &
        (df.Date <= min_pavers_date),
        "LayoutNameAdj"
    ] = "Paver Tees"
    df.loc[
        (df.CourseName == "Bryan Park") &
        (df.LayoutName == "Main") &
        (df.Date > min_pavers_date),
        "LayoutNameAdj"
    ] = "Yellows Tees"

    df['Year'] = pd.DatetimeIndex(df['Date']).year
    
    return df

def get_year_stats(df):
    year_df = df.groupby(seg_cols + ["Year"]).mean().reset_index()
    return year_df

def moving_avg(df, val_col, seg_cols, date_col, period, new_col=None):
    df = df.sort_values(seg_cols + [date_col]).reset_index(drop=True)
    
    date_df = df.set_index(date_col)
    
    if not new_col:
        new_col = val_col
        
    df[new_col] = date_df.groupby(
        seg_cols
    )[val_col].rolling(period, min_periods=1).mean().reset_index(drop=True)
    
    return df

def get_score_avg(df, periods=[5, 10, 20]):
    periods = [5, 10, 20]

    ma_df = df.rename(columns={"+/-": "Score"})

    for p in periods:
        ma_df = moving_avg(
            df=ma_df,
            val_col="Score",
            seg_cols=seg_cols,
            date_col="Date",
            period=p,
            new_col=f"{p} Round Avg"
        )
        
    score_df = ma_df.melt(id_vars=seg_cols + ["Date"], value_vars=["Score"] + [f"{p} Round Avg" for p in periods])
    
    return score_df

def get_score_counts(df, period=10):
    hole_cols = [x for x in df.columns if x.startswith("Hole")]

    melt_df = df.melt(id_vars=seg_cols + ["Date"], value_vars=hole_cols)
    melt_df.rename(columns={"variable": "Hole", "value": "Score"}, inplace=True)
    melt_df = melt_df[melt_df["Score"] != 0.0].reset_index(drop=True)

    par_df = melt_df[melt_df.PlayerName == "Par"].reset_index(drop=True)
    par_df.drop("PlayerName", axis=1, inplace=True)
    par_df.rename(columns={"Score": "Par"}, inplace=True)

    players_df = melt_df[melt_df.PlayerName != "Par"].reset_index(drop=True)

    players_par_df = players_df.merge(par_df, on=["CourseName", "LayoutNameAdj", "Date", "Hole"])

    players_par_df.dropna(subset=["Score"], inplace=True)
    players_par_df["Diff"] = players_par_df.Score - players_par_df.Par


    score_map = {
        -2.0: "Eagle-",
        -1.0: "Birdie",
        0.0: "Par",
        1.0: "Bogie",
        2.0: "Double Bogie",
        3.0: "Triple Bogie+",
    }

    players_par_df["ScoreName"] = players_par_df["Diff"].map(lambda x: score_map[x] if x in score_map else None)

    players_par_df.loc[players_par_df["Diff"] < -2, "ScoreName"] = "Eagle-"
    players_par_df.loc[players_par_df["Diff"] > 3, "ScoreName"] = "Triple Bogie+"

    assert len(players_par_df[players_par_df.Diff.isna()]) == 0

    group_df = players_par_df[
        seg_cols + ["Date", "ScoreName", "Score"]
    ].groupby(seg_cols + ["Date", "ScoreName"]).count().reset_index()
    group_df.rename(columns={"Score": "Frequency"}, inplace=True)

    idx_df = group_df[seg_cols + ["Date"]].groupby(seg_cols + ["Date"]).count().reset_index()
    idx_df.reset_index(inplace=True)
    idx_df.rename(columns={"index": "TmpMergeCol"}, inplace=True)

    score_name_df = pd.DataFrame(
        list(product(idx_df.TmpMergeCol.values.tolist(), list(score_map.values()))),
        columns=["TmpMergeCol", "ScoreName"]
    )

    idx_score_name_df = idx_df.merge(score_name_df, on=["TmpMergeCol"])
    idx_score_name_df.drop("TmpMergeCol", axis=1, inplace=True)

    merge_df = idx_score_name_df.merge(group_df, on=seg_cols + ["Date", "ScoreName"], how="outer")
    merge_df.Frequency.fillna(0.0, inplace=True)

    ma_df = moving_avg(
        df=merge_df,
        val_col="Frequency",
        seg_cols=seg_cols + ["ScoreName"],
        date_col="Date",
        period=period,
    )
    
    return ma_df

def get_player_stats(df, player, course, layout):
    year_df = get_year_stats(df)
    print(f"Yearly Stats for {player} at {course} from the {layout}")
    display(year_df[
        (year_df.PlayerName == player) &
        (year_df.CourseName == course) &
        (year_df.LayoutNameAdj == layout)
    ])

    score_df = get_score_avg(df)
    fig, ax = plt.subplots(figsize=(15, 8))
    viz_df = score_df[
        (score_df.PlayerName == player) &
        (score_df.CourseName == course) &
        (score_df.LayoutNameAdj == layout)
    ]
    sns.lineplot(data=viz_df, x="Date", y="value", hue="variable", ax=ax).set(
        title=f"Score Relative to Par for {player} at {course} from the {layout}"
    )

    score_counts_df = get_score_counts(df)
    fig, ax = plt.subplots(figsize=(15, 8))
    viz_df = score_counts_df[
        (score_counts_df.PlayerName == player) &
        (score_counts_df.CourseName == course) &
        (score_counts_df.LayoutNameAdj == layout)
    ]
    sns.lineplot(x="Date", y="Frequency", hue="ScoreName", data=viz_df).set(
        title=f"10 Round Avg of # of Each Score Achieved by {player} at {course} on the {layout}"
    )