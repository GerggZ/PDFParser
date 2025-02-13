import pandas as pd

def merge_sentences_by_position(df: pd.DataFrame, y_tol=1, x_tol=3.5):
    """ Merges text boxes that are mistakenly split across rows or columns. """
    if df.empty:
        return df

    df = df.sort_values(by=["page", "y0", "x0"]).reset_index(drop=True)
    merged_rows = []
    merged_text = df.iloc[0].copy()

    for i in range(1, len(df)):
        current_row = df.iloc[i]

        y_dif = abs(current_row["y0"] - merged_text["y0"])
        x_dif = abs(current_row["x0"] - merged_text["x1"])

        if y_dif < y_tol and x_dif < x_tol:
            merged_text["text"] += " " + current_row["text"]
            merged_text["x0"] = min(merged_text["x0"], current_row["x0"])
            merged_text["x1"] = max(merged_text["x1"], current_row["x1"])
        else:
            merged_rows.append(merged_text)
            merged_text = current_row.copy()

    merged_rows.append(merged_text)
    return pd.DataFrame(merged_rows)


def remove_header_footer_text(df: pd.DataFrame, header_height=40, footer_height=800):
    """ Removes unwanted header/footer text based on y-position. """
    return df[(df["y0"] > header_height) & (df["y0"] < footer_height)].reset_index(drop=True)
