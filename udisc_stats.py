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

def preprocess(df, drop_partial=True):
    df = df.copy()
    df.Date = pd.to_datetime(df.Date)

    min_pavers_date = df[df.LayoutName == "Paver Tees"].Date.min()
    print(min_pavers_date)
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
    
    if drop_partial:
        hole_cols = [x for x in df.columns if x.startswith("Hole")]
        # Missing Scores represented as 0.0
        for col in hole_cols:
            df = df[df[col] != 0.0]
    
    return df.drop_duplicates().reset_index(drop=True)

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

def get_score_counts(df, period=10, holes=None):
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
    
    if holes is not None:
        players_par_df = players_par_df[players_par_df.Hole.isin(holes)]

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

def get_month_df(df):
    month_df = df.copy()
    month_df["Month"] = (month_df['Date'].dt.floor("D") + pd.offsets.MonthBegin(-1))
    month_df.rename(columns={"+/-": "Score", "Total": "Num Rounds"}, inplace=True)
    month_agg_df = month_df.groupby(seg_cols + ["Month"]).agg({"Score": "mean", "Num Rounds": "count"}).reset_index()
    
    return month_agg_df

def plot_month_df(viz_df, title):
    fig, ax = plt.subplots(figsize=(15, 8))

    viz_df["MonthStr"] = viz_df["Month"].dt.strftime("%Y-%m")

    pastel_orange=(1.0, 0.7058823529411765, 0.5098039215686274)
    sns.barplot(x="MonthStr", y="Num Rounds", color=pastel_orange, data=viz_df, ax=ax)
    ax.grid(visible=True, axis="y", linestyle="--")
    ax2 = ax.twinx()
    sns.lineplot(x="MonthStr", y="Score", marker='o', data=viz_df, ax=ax2)
    ax.set_title(title)

    for x,y in zip(viz_df.MonthStr, viz_df.Score):

        label = "{:.2f}".format(y)

        ax2.annotate(label,
                     (x,y),
                     textcoords="offset points",
                     xytext=(0,10),
                     ha='center')

        
def get_goal(df):
    monkey_df = df[
        (df.PlayerName == "Monkey") &
        (df.CourseName == "Bryan Park") &
        (df.LayoutName == "Yellows Tees")
    ]

    score = monkey_df["+/-"].sum()

    print("Score:", score)

def get_player_stats(df, player, course, layout, holes=None, min_date=None):
    if min_date is None:
        min_date = df.Date.min()
    if isinstance(min_date, str):
        min_date = pd.Timestamp(min_date)

    year_df = get_year_stats(df)
    print(f"Yearly Stats for {player} at {course} from the {layout}")
    display(year_df[
        (year_df.PlayerName == player) &
        (year_df.CourseName == course) &
        (year_df.LayoutNameAdj == layout)
    ].dropna(axis=1, how='all'))

    score_df = get_score_avg(df)
    fig, ax = plt.subplots(figsize=(15, 8))
    viz_df = score_df[
        (score_df.PlayerName == player) &
        (score_df.CourseName == course) &
        (score_df.LayoutNameAdj == layout) &
        (score_df.Date >= min_date)
    ]
    sns.lineplot(data=viz_df, x="Date", y="value", hue="variable", ax=ax).set(
        title=f"Score Relative to Par for {player} at {course} from the {layout}"
    )

    score_counts_df = get_score_counts(df, holes=holes)
    fig, ax = plt.subplots(figsize=(15, 8))
    viz_df = score_counts_df[
        (score_counts_df.PlayerName == player) &
        (score_counts_df.CourseName == course) &
        (score_counts_df.LayoutNameAdj == layout) &
        (score_counts_df.Date >= min_date)
    ]
    sns.lineplot(x="Date", y="Frequency", hue="ScoreName", data=viz_df).set(
        title=f"10 Round Avg of # of Each Score Achieved by {player} at {course} on the {layout}"
    )
    
    month_agg_df = get_month_df(df)
    
    viz_df = month_agg_df[
        (month_agg_df.PlayerName == player) &
        (month_agg_df.CourseName == course) &
        (month_agg_df.LayoutNameAdj == layout) &
        (month_agg_df.Month >= min_date)
    ]
    
    title = f"Avg Score & Number of Rounds Played by {player} at {course} on the {layout}"
    plot_month_df(viz_df, title)