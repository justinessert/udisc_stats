from IPython import display


def print_score_pct(player_name, year, df, prev_df, score_name):
    score_pct = df[df["ScoreName"] == score_name]["Frequency"].iloc[0]
    score_prev_pct = prev_df[prev_df["ScoreName"] == score_name]["Frequency"].iloc[0]
    diff_score_pct = score_pct - score_prev_pct
    print(
        f"{player_name}'s {score_name.lower()} percentage in {year} was {score_pct}% ({diff_score_pct:+}%)"
    )


def stats_wrapped(player_name, year, df):
    prev_year = year - 1

    player_df = df[(df["PlayerName"] == player_name) & (df["Year"] == year)]
    player_prev_df = df[(df["PlayerName"] == player_name) & (df["Year"] == prev_year)]

    player_df["month"] = player_df["Date"].dt.month
    display(player_df.groupby(["month"]).count())

    n_rounds = len(player_df)
    n_prev_rounds = len(player_prev_df)
    diff_rounds = int((n_rounds - n_prev_rounds) / n_prev_rounds * 100)

    print(f"{player_name} played {n_rounds} rounds in {year} ({diff_rounds:+}%)")

    score_df = get_score_counts(df, period=None)

    player_score_df = score_df[
        (score_df["PlayerName"] == player_name) & (score_df["Year"] == year)
    ]
    player_prev_score_df = score_df[
        (score_df["PlayerName"] == player_name) & (score_df["Year"] == prev_year)
    ]

    player_score_agg_df = (
        player_score_df.groupby(["PlayerName", "Year", "ScoreName"]).sum().reset_index()
    )
    n_holes = player_score_agg_df["Frequency"].sum()
    player_score_agg_df["Frequency"] = (
        player_score_agg_df["Frequency"] / n_holes * 100
    ).astype(int)

    player_prev_score_agg_df = (
        player_prev_score_df.groupby(["PlayerName", "Year", "ScoreName"])
        .sum()
        .reset_index()
    )
    n_prev_holes = player_prev_score_agg_df["Frequency"].sum()
    player_prev_score_agg_df["Frequency"] = (
        player_prev_score_agg_df["Frequency"] / n_prev_holes * 100
    ).astype(int)

    diff_holes = int((n_holes - n_prev_holes) / n_prev_holes * 100)

    print(f"{player_name} played {n_holes} holes in {year} ({diff_holes:+}%)")
    print("")
    print_score_pct(
        player_name, year, player_score_agg_df, player_prev_score_agg_df, "Birdie"
    )
    print_score_pct(
        player_name, year, player_score_agg_df, player_prev_score_agg_df, "Par"
    )
    print_score_pct(
        player_name, year, player_score_agg_df, player_prev_score_agg_df, "Bogie"
    )

    print("")

    min_score = int(player_df["+/-"].min())
    min_player_df = player_df[player_df["+/-"] == min_score]
    min_score_course = min_player_df["CourseName"].iloc[0]
    min_score_layout = min_player_df["LayoutNameAdj"].iloc[0]
    min_score_date = min_player_df["Date"].iloc[0].strftime("%Y-%m-%d")

    print(
        f"{player_name}'s best round was {min_score:+} on {min_score_date}"
        f" at {min_score_course} playing the {min_score_layout}"
    )
