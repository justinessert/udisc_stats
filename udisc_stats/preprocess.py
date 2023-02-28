import pandas as pd

def print_segments(df):
    print(f"Players: {list(df.PlayerName.unique())}")
    print(f"Courses: {list(df.CourseName.unique())}")
    print(f"Layouts: {list(df.LayoutNameAdj.unique())}")

def preprocess(df, drop_partial=True):
    df = df.copy()
    df.Date = pd.to_datetime(df.Date)

    min_pavers_date = df[df.LayoutName == "Paver Tees"].Date.min()
    print(min_pavers_date)
    max_pavers_date="2022-12-01"
    df["LayoutNameAdj"] = df.LayoutName
    df.loc[
        (df.CourseName == "Bryan Park") &
        (df.LayoutName == "Main") &
        (~df.Date.between(min_pavers_date, max_pavers_date)),
        "LayoutNameAdj"
    ] = "Paver Tees"
    df.loc[
        (df.CourseName == "Bryan Park") &
        (df.LayoutName == "Main") &
        (df.Date.between(min_pavers_date, max_pavers_date)),
        "LayoutNameAdj"
    ] = "Yellows Tees"
    
    df.loc[
        (df.CourseName == "Bryan Park") &
        (df.LayoutName == "Yellow Tees (Shorts)"),
        "LayoutNameAdj"
    ] = "Yellows Tees"

    df['Year'] = pd.DatetimeIndex(df['Date']).year
    
    if drop_partial:
        hole_cols = [x for x in df.columns if x.startswith("Hole")]
        # Missing Scores represented as 0.0
        for col in hole_cols:
            df = df[df[col] != 0.0]
    
    return df.drop_duplicates().reset_index(drop=True)