from itertools import product
import os
from typing import List, Optional
import pandas as pd

from udisc_stats.visualize import get_player_stats


def generate_report(
    df: pd.DataFrame,
    player: str,
    course: str,
    layout: str,
    directory: str,
    holes: Optional[List[str]]=None,
    min_date: Optional[str]=None,
):
    tag = f"{player.replace(' ', '')}_{course.replace(' ', '')}_{layout.replace(' ', '')}"
    plot_dir = os.path.join(directory, "img", tag)
    os.makedirs(plot_dir, exist_ok=True)

    get_player_stats(
        df=df,
        player=player,
        course=course,
        layout=layout,
        holes=holes,
        min_date=min_date,
        plot_dir=plot_dir,
        apply_filter_to_cal = True,
    )

    report_list = [
        f"# Stats for {player} at {course} from the {layout}",
        "",
        "## Best Round",
        "",
        f"![best_scores](img/{tag}/best_scores.png)",
        "",
        "## Calendar of Played Rounds",
        "",
        f"![round_calendar](img/{tag}/round_calendar.png)",
        "",
        "## Score Per Round Metrics",
        "",
        "### Average Scores By Month",
        "",
        f"![avg_score](img/{tag}/avg_score.png)",
        "",
        "### Scores With Windowed Averages",
        "",
        f"![score_summary](img/{tag}/score_summary.png)",
        "",
        "### Number of Birdies/Pars/Bogies/Etc Over Time",
        "",
        f"![score_frequency](img/{tag}/score_frequency.png)",
        "",
        "## Score Per Hole Metrics",
        "",
        "### Scores Per Hole",
        "",
        f"![histogram](img/{tag}/histogram.png)",
        "",
        "### Cumulative Score Splits Per Hole",
        "",
        "#### Overall Cumulative Score Splits Per Hole",
        "",
        f"![overall_splits](img/{tag}/overall_splits.png)",
        "",
        "#### Per Year Cumulative Score Splits Per Hole",
        "",
        f"![year_score_splits](img/{tag}/year_score_splits.png)",
        "",
    ]

    report_text = "\n".join(report_list)

    with open(os.path.join(directory, f"{tag}.md"), 'w') as f:
        f.write(report_text)
    
def generate_reports(
    df: pd.DataFrame,
    players: List[str],
    courses: List[str],
    layouts: dict,
    directory: str,
    holes: Optional[List[str]]=None,
    min_date: Optional[str]=None,
):
    for player, course in product(players, courses):
        for layout in layouts[course]:
            generate_report(
                df=df,
                player=player,
                course=course,
                layout=layout,
                directory=directory,
                holes=holes,
                min_date=min_date,
            )