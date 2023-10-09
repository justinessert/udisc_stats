from math import ceil
import os
from typing import List, Optional
import pandas as pd
from itertools import product
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
import calmap
import plotly.graph_objects as go

seg_cols = [
    "PlayerName",
    "CourseName",
    "LayoutNameAdj",
]

SCORE_MAP = {
    -2.0: "Eagle-",
    -1.0: "Birdie",
    0.0: "Par",
    1.0: "Bogie",
    2.0: "Double Bogie",
    3.0: "Triple Bogie+",
}

def _save_plot(fig: plt.Figure, plot_dir: Optional[str] = None, filepath: Optional[str] = None):
    if plot_dir and filepath:
        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(os.path.join(plot_dir, filepath))

def get_year_stats(df: pd.DataFrame) -> pd.DataFrame:
    year_df = df.groupby(seg_cols + ["Year"]).mean().reset_index()
    return year_df

def moving_avg(
    df: pd.DataFrame,
    val_col: str,
    seg_cols: List[str],
    date_col: str,
    period: int,
    new_col: str=None,
) -> pd.DataFrame:
    df = df.sort_values(seg_cols + [date_col]).reset_index(drop=True)
    
    date_df = df.set_index(date_col)
    
    if not new_col:
        new_col = val_col
        
    df[new_col] = date_df.groupby(
        seg_cols
    )[val_col].rolling(period, min_periods=1).mean().reset_index(drop=True)
    
    return df

def get_score_avg(
    round_score_df: pd.DataFrame,
    periods: List[int] = None,
) -> pd.DataFrame:
    periods = periods or [5, 10, 20]
    ma_df = round_score_df.copy(deep=True)

    for p in periods:
        ma_df = moving_avg(
            df=ma_df,
            val_col="Diff",
            seg_cols=seg_cols,
            date_col="Date",
            period=p,
            new_col=f"{p} Round Avg"
        )
        
    score_df = ma_df.melt(id_vars=seg_cols + ["Date"], value_vars=["Diff"] + [f"{p} Round Avg" for p in periods])
    
    return score_df

def get_score_df(
    df: pd.DataFrame,
    holes: Optional[List[str]]=None,
) -> pd.DataFrame:
    hole_cols = [x for x in df.columns if x.startswith("Hole")]

    melt_df = df.melt(id_vars=seg_cols + ["Date", "Year"], value_vars=hole_cols)
    melt_df.rename(columns={"variable": "Hole", "value": "Score"}, inplace=True)
    melt_df.Hole = melt_df.Hole.map(lambda x: f"Hole{int(x.split('Hole')[-1]):02}")
    melt_df = melt_df[melt_df["Score"] != 0.0].reset_index(drop=True)

    par_df = melt_df[melt_df.PlayerName == "Par"].reset_index(drop=True)
    par_df.drop("PlayerName", axis=1, inplace=True)
    par_df.rename(columns={"Score": "Par"}, inplace=True)

    players_df = melt_df[melt_df.PlayerName != "Par"].reset_index(drop=True)

    players_par_df = players_df.merge(par_df, on=["CourseName", "LayoutNameAdj", "Date", "Year", "Hole"])

    players_par_df.dropna(subset=["Score"], inplace=True)
    players_par_df["Diff"] = players_par_df.Score - players_par_df.Par

    players_par_df["ScoreName"] = players_par_df["Diff"].map(lambda x: SCORE_MAP[x] if x in SCORE_MAP else None)

    players_par_df.loc[players_par_df["Diff"] < -2, "ScoreName"] = "Eagle-"
    players_par_df.loc[players_par_df["Diff"] > 3, "ScoreName"] = "Triple Bogie+"

    assert len(players_par_df[players_par_df.Diff.isna()]) == 0
    
    if holes is not None:
        players_par_df = players_par_df[players_par_df.Hole.isin(holes)]
    
    players_par_df = players_par_df.sort_values(['Date', 'PlayerName', 'CourseName', 'LayoutNameAdj', "Hole"]).reset_index(drop=True)
    
    return players_par_df

def get_round_score_df(
    score_df: pd.DataFrame,
):
    round_score_df = score_df.groupby(seg_cols + ["Date", "Year"]).sum().reset_index()
    round_score_df["Diff"] = round_score_df["Score"] - round_score_df["Par"]

    sort_cols = ['Date', 'PlayerName', 'CourseName', 'LayoutNameAdj']
    round_score_df = round_score_df.sort_values(sort_cols).reset_index(drop=True)

    return round_score_df

def get_cumulative_score_df(
    df: pd.DataFrame,
) -> pd.DataFrame:
    score_df = get_score_df(df)

    seg_cols = ["PlayerName", "CourseName", "LayoutNameAdj", "Date"]
    seg_hole_cols = seg_cols + ["Hole"]
    score_df = score_df.sort_values(seg_hole_cols).reset_index(drop=True)

    score_df[
        "CumulativeDiff"
    ] = score_df.groupby(seg_hole_cols).sum().groupby(seg_cols).cumsum().reset_index()["Diff"].astype(float)

    return score_df

def get_score_counts(
    score_df: pd.DataFrame,
    period: Optional[int]=10,
) -> pd.DataFrame:

    group_df = score_df[
        seg_cols + ["Date", "Year", "ScoreName", "Score"]
    ].groupby(seg_cols + ["Date", "Year", "ScoreName"]).count().reset_index()
    group_df.rename(columns={"Score": "Frequency"}, inplace=True)

    idx_df = group_df[seg_cols + ["Date", "Year"]].groupby(seg_cols + ["Date", "Year"]).count().reset_index()
    idx_df.reset_index(inplace=True)
    idx_df.rename(columns={"index": "TmpMergeCol"}, inplace=True)

    score_name_df = pd.DataFrame(
        list(product(idx_df.TmpMergeCol.values.tolist(), list(SCORE_MAP.values()))),
        columns=["TmpMergeCol", "ScoreName"]
    )

    idx_score_name_df = idx_df.merge(score_name_df, on=["TmpMergeCol"])
    idx_score_name_df.drop("TmpMergeCol", axis=1, inplace=True)

    merge_df = idx_score_name_df.merge(group_df, on=seg_cols + ["Date", "Year", "ScoreName"], how="outer")
    merge_df.Frequency.fillna(0.0, inplace=True)
    
    if not period:
        return merge_df

    ma_df = moving_avg(
        df=merge_df,
        val_col="Frequency",
        seg_cols=seg_cols + ["ScoreName"],
        date_col="Date",
        period=period,
    )
    
    return ma_df

def get_hist_df(player_score_df):
    seg_cols = ["Hole", "Score", "ScoreName"]
    hist_df = player_score_df[seg_cols + ["Diff"]].groupby(seg_cols).count().reset_index()
    hist_df.rename(columns={"Diff": "Count"}, inplace=True)

    holes = list(hist_df.Hole.unique())
    score_names = list(hist_df.ScoreName.unique())

    all_scores = pd.DataFrame(list(product(holes, score_names)), columns=["Hole", "ScoreName"])

    hist_df = hist_df.merge(all_scores, on=["Hole", "ScoreName"], how="outer")
    hist_df["Order"] = hist_df["ScoreName"].map(lambda x: score_names.index(x))
    hist_df = hist_df.sort_values(["Hole", "Order"]).reset_index(drop=True)
    hist_df.Count.fillna(0, inplace=True)

    return hist_df


def get_month_df(round_score_df: pd.DataFrame) -> pd.DataFrame:
    month_df = round_score_df.copy()
    month_df["Month"] = (month_df['Date'].dt.floor("D") + pd.offsets.MonthBegin(-1))
    month_df["Num Rounds"] = 1
    month_agg_df = month_df.groupby(seg_cols + ["Month"]).agg({"Diff": "mean", "Num Rounds": "count"}).reset_index()
    
    return month_agg_df

def plot_month_df(viz_df: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(15, 8))

    viz_df["MonthStr"] = viz_df["Month"].dt.strftime("%Y-%m")

    pastel_orange=(1.0, 0.7058823529411765, 0.5098039215686274)
    sns.barplot(x="MonthStr", y="Num Rounds", color=pastel_orange, data=viz_df, ax=ax)
    ax.grid(visible=True, axis="y", linestyle="--")
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=45)
    ax2 = ax.twinx()
    sns.lineplot(x="MonthStr", y="Diff", marker='o', data=viz_df, ax=ax2)
    ax.set_title(title)

    for x,y in zip(viz_df.MonthStr, viz_df.Diff):

        label = "{:.2f}".format(y)

        ax2.annotate(label,
                     (x,y),
                     textcoords="offset points",
                     xytext=(0,10),
                     ha='center')
    
    return fig

        
def get_goal(df: pd.DataFrame):
    monkey_df = df[
        (df.PlayerName == "Monkey") &
        (df.CourseName == "Bryan Park") &
        (df.LayoutName == "Yellows Tees")
    ]

    score = monkey_df["+/-"].sum()

    print("Score:", score)

def plot_calmap(df: pd.DataFrame, player: str):
    count_df = df[["Date", "PlayerName", "Total"]].groupby(["Date", "PlayerName"]).count().reset_index()

    player_count_df = count_df[count_df["PlayerName"] == player]
    count_series = player_count_df.set_index("Date")["Total"]

    years = list(player_count_df["Date"].map(lambda x: x.year).unique())

    fig, _ = calmap.calendarplot(count_series, fillcolor='grey', fig_kws=dict(figsize=(15, 3*len(years))))

    fig.suptitle(f"Number of Rounds Played Per Day By {player}", fontsize=26)

    return fig

def plot_histogram(hist_df: pd.DataFrame) -> plt.Figure:
    score_names = list(hist_df["ScoreName"].unique())
    holes = list(hist_df["Hole"].unique())
    cols = 3
    rows = ceil(len(holes) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15,3*rows), sharex="all", sharey="all")
    for i, hole in enumerate(holes):
        row = i // cols
        col = i % cols
        
        ax=axes[row][col]
        
        viz_df = hist_df[hist_df["Hole"] == hole]
        
        sns.barplot(x="ScoreName", y="Count",
                    data=viz_df, color=sns.color_palette()[0],
                    ax=ax,
        )
        ax.set_title(f"Histogram of Scores for {hole}")
        ax.set_xticklabels(score_names, rotation = 45)
    plt.tight_layout()

    return fig


def _generate_best_score_plot(
    par: int,
    best_overall: int,
    best_ytd: int,
    best_12m: int,
    best_6m: int,
    best_3m: int,
    plot_dir: Optional[str] = None
):

    fig = go.Figure()

    fig.update_layout(
        title={
            "text": "Best Round",
            "xanchor": "center",
            "font": {"size": 48},
            "x": 0.5
        }  
    )

    delta_config = {
        'reference': par, 'relative': False, 'position' : "top", "increasing.color": "red", "decreasing.color": "green"
    }

    fig.add_trace(go.Indicator(
        mode = "delta",
        value = best_overall,
        domain = {'x': [0, 1], 'y': [0.5, 1]},
        title= {"text": "Overall"},
        delta = delta_config))

    fig.add_trace(go.Indicator(
        mode = "delta",
        value = best_ytd,
        delta = delta_config,
        title= {"text": "YTD"},
        domain = {'x': [0, 0.45], 'y': [0.25, 0.4]}))

    fig.add_trace(go.Indicator(
        mode = "delta",
        value = best_12m,
        delta = delta_config,
        title= {"text": "Last 12 Months"},
        domain = {'x': [0.55, 1], 'y': [0.25, 0.4]}))

    fig.add_trace(go.Indicator(
        mode = "delta",
        value = best_6m,
        delta = delta_config,
        title= {"text": "Last 6 Months"},
        domain = {'x': [0, 0.45], 'y': [0, 0.15]}))

    fig.add_trace(go.Indicator(
        mode = "delta",
        value = best_3m,
        delta = delta_config,
        title= {"text": "Last 3 Months"},
        domain = {'x': [0.55, 1.0], 'y': [0, 0.15]}))

    fig.show()
    
    if plot_dir is not None:
        fig.write_image(os.path.join(plot_dir, "best_scores.png"))


def plot_best_scores(
    round_score_df: pd.DataFrame,
    player: str,
    course: str,
    layout: str,
    today: Optional[pd.Timestamp] = None,
    plot_dir: Optional[str] = None,
):
    if today is None:
        today = pd.Timestamp.today()

    segment_round_score_df = round_score_df[
        (round_score_df.PlayerName == player) &
        (round_score_df.CourseName == course) &
        (round_score_df.LayoutNameAdj == layout)
    ]

    segment_round_score_df = segment_round_score_df.sort_values("Score", ascending=True).reset_index(drop=True)

    unique_pars = segment_round_score_df["Par"].unique()
    assert len(unique_pars) == 1

    par = unique_pars[0]

    best_overall = segment_round_score_df["Score"].values[0]

    best_ytd = segment_round_score_df[
        segment_round_score_df.Date >= pd.Timestamp(f"01-01-{pd.Timestamp.today().year}")
    ]["Score"].values[0]

    best_12m = segment_round_score_df[
        segment_round_score_df.Date >= (today - pd.Timedelta(30 * 12, "d"))
    ]["Score"].values[0]

    best_6m = segment_round_score_df[
        segment_round_score_df.Date >= (today - pd.Timedelta(30 * 6, "d"))
    ]["Score"].values[0]

    best_3m = segment_round_score_df[
        segment_round_score_df.Date >= (today - pd.Timedelta(30 * 3, "d"))
    ]["Score"].values[0]

    _generate_best_score_plot(
        par = par,
        best_overall = best_overall,
        best_ytd = best_ytd,
        best_12m = best_12m,
        best_6m = best_6m,
        best_3m = best_3m,
        plot_dir = plot_dir,
    )


def get_player_stats(
    df: pd.DataFrame,
    player: str,
    course: str,
    layout: str,
    holes: Optional[List[str]]=None,
    min_date: Optional[str]=None,
    plot_dir: Optional[str]=None,
):
    if min_date is None:
        min_date = df.Date.min()
    if isinstance(min_date, str):
        min_date = pd.Timestamp(min_date)

    score_df = get_score_df(df, holes)
    round_score_df = get_round_score_df(score_df)
        
    fig = plot_calmap(df, player)
    _save_plot(fig, plot_dir, "round_calendar.png")

    plot_best_scores(
        round_score_df=round_score_df,
        player=player,
        course=course,
        layout=layout,
        plot_dir=plot_dir,
    )

    score_avg_df = get_score_avg(round_score_df)
    fig, ax = plt.subplots(figsize=(15, 8))
    viz_df = score_avg_df[
        (score_avg_df.PlayerName == player) &
        (score_avg_df.CourseName == course) &
        (score_avg_df.LayoutNameAdj == layout) &
        (score_avg_df.Date >= min_date)
    ]
    sns.lineplot(data=viz_df, x="Date", y="value", hue="variable", ax=ax)
    ax.set_title(
        f"Score Relative to Par for {player} at {course} from the {layout}"
    )
    _save_plot(fig, plot_dir, "score_summary.png")

    score_counts_df = get_score_counts(score_df)
    fig, ax = plt.subplots(figsize=(15, 8))
    viz_df = score_counts_df[
        (score_counts_df.PlayerName == player) &
        (score_counts_df.CourseName == course) &
        (score_counts_df.LayoutNameAdj == layout) &
        (score_counts_df.Date >= min_date)
    ]
    sns.lineplot(x="Date", y="Frequency", hue="ScoreName", data=viz_df, ax=ax)
    ax.set_title(
        f"10 Round Avg of # of Each Score Achieved by {player} at {course} on the {layout}", fontsize=24,
    )
    _save_plot(fig, plot_dir, "score_frequency.png")
    
    month_agg_df = get_month_df(round_score_df)
    
    viz_df = month_agg_df[
        (month_agg_df.PlayerName == player) &
        (month_agg_df.CourseName == course) &
        (month_agg_df.LayoutNameAdj == layout) &
        (month_agg_df.Month >= min_date)
    ].reset_index(drop=True)
    
    title = f"Avg Score & Number of Rounds Played by {player} at {course} on the {layout}"
    fig = plot_month_df(viz_df, title)
    _save_plot(fig, plot_dir, "avg_score.png")

    cum_score_df = get_cumulative_score_df(df)
    player_cum_score_df = cum_score_df[
        (cum_score_df["PlayerName"]==player) &
        (cum_score_df["CourseName"]==course) &
        (cum_score_df["LayoutNameAdj"]==layout)
    ]

    fig, ax = plt.subplots(figsize=(15, 15))
    sns.boxplot(data=player_cum_score_df, x="CumulativeDiff", y="Hole", color=sns.color_palette()[0])
    ax.grid(axis="x")
    ax.set_title(f"Overall Score Splits for {player} at {course} from the {layout}", fontsize=24)
    plt.tight_layout()
    _save_plot(fig, plot_dir, "overall_splits.png")

    fig, ax = plt.subplots(figsize=(15, 20))
    sns.boxplot(data=player_cum_score_df, x="CumulativeDiff", y="Hole", hue="Year")
    ax.grid(axis="x")
    ax.set_title(f"Score Splits for {player} at {course} from the {layout} by Year", fontsize=24)
    plt.tight_layout()
    _save_plot(fig, plot_dir, "year_score_splits.png")

    player_score_df = score_df[
        (score_df.PlayerName == player) &
        (score_df.CourseName == course) &
        (score_df.LayoutNameAdj == layout)
    ]
    hist_df = get_hist_df(player_score_df)
    fig = plot_histogram(hist_df)
    fig.suptitle(f"Histogram of Scores per Hole for {player} at {course} from the {layout}", fontsize=24)
    plt.tight_layout()
    _save_plot(fig, plot_dir, "histogram.png")
