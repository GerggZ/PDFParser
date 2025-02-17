import pandas as pd
from pdf_parser.alignment import align_text_boxes
from pdf_parser.pdf_processor import PDFTextProcessor


class TableGenerator:
    """
    Extracts structured tables from a subset of PDF text data.
    """

    DEFAULT_Y_THRESHOLD = 8
    DEFAULT_X_THRESHOLD = 15
    DEFAULT_ALIGN_COLUMNS_X = ["x0", "x1"]
    DEFAULT_ALIGN_COLUMNS_Y = ["y0"]

    @staticmethod
    def align_text_positions(df: pd.DataFrame, x_threshold: float, y_threshold: float,
                             align_columns_x: list, align_columns_y: list) -> pd.DataFrame:
        """Aligns text positions and adds 'x_align' and 'y_align' columns."""
        df = df.copy()  # Avoid modifying the original
        df["x_align"] = align_text_boxes(df, position_columns=align_columns_x,
                                         alignment_threshold=x_threshold, per_page=False)
        df["y_align"] = align_text_boxes(df, position_columns=align_columns_y,
                                         alignment_threshold=y_threshold, per_page=True)
        return df

    @staticmethod
    def sort_text_by_position(df: pd.DataFrame) -> pd.DataFrame:
        """Sorts the text elements by page, y (rows), and x (columns)."""
        return df.sort_values(by=["page", "y_align", "x_align"]).reset_index(drop=True)

    @staticmethod
    def group_rows(df: pd.DataFrame, y_threshold: float):
        """Groups text into row-wise structure based on y-alignment."""
        rows = []
        current_row = []
        last_y = None

        for _, row in df.iterrows():
            if last_y is None or abs(row["y_align"] - last_y) > y_threshold:
                if current_row:
                    rows.append(current_row)
                current_row = []
            current_row.append(row)
            last_y = row["y_align"]

        if current_row:
            rows.append(current_row)

        return rows

    @staticmethod
    def determine_column_positions(df: pd.DataFrame, x_threshold: float):
        """Determines and normalizes column positions."""
        col_positions = sorted(set(row.x_align for row in df.itertuples()))
        normalized_cols = {}
        representative_x = None

        for col_x in col_positions:
            if representative_x is None or abs(col_x - representative_x) > x_threshold:
                representative_x = col_x
            normalized_cols[col_x] = representative_x

        sorted_columns = sorted(set(normalized_cols.values()))
        return normalized_cols, sorted_columns

    @staticmethod
    def build_table(rows, normalized_cols, sorted_columns):
        """Constructs a DataFrame from the extracted table data."""
        col_indices = {x: i for i, x in enumerate(sorted_columns)}
        table_data = []

        for row in rows:
            row_dict = {col_idx: None for col_idx in col_indices.values()}

            for entry in row:
                col_key = normalized_cols[entry["x_align"]]
                col_idx = col_indices[col_key]
                row_dict[col_idx] = (
                    entry["text"] if row_dict[col_idx] is None
                    else row_dict[col_idx] + " " + entry["text"]
                )

            table_data.append([row_dict[i] for i in range(len(col_indices))])

        return pd.DataFrame(table_data)


def generate_table_from_dataframe(
    pdf_data_subset: pd.DataFrame,
    y_threshold: float = TableGenerator.DEFAULT_Y_THRESHOLD,
    x_threshold: float = TableGenerator.DEFAULT_X_THRESHOLD,
    align_columns_x: list = TableGenerator.DEFAULT_ALIGN_COLUMNS_X,
    align_columns_y: list = TableGenerator.DEFAULT_ALIGN_COLUMNS_Y
) -> pd.DataFrame:
    """
    Wrapper function to extract a structured table from a PDF data subset.

    Parameters:
        pdf_data_subset (pd.DataFrame): DataFrame containing table-related rows.
        y_threshold (float): Row alignment threshold.
        x_threshold (float): Column alignment threshold.
        align_columns_x (list): Columns for x-alignment.
        align_columns_y (list): Columns for y-alignment.

    Returns:
        pd.DataFrame: Extracted table as a structured DataFrame.
    """
    if pdf_data_subset.empty:
        raise ValueError("The provided DataFrame subset is empty.")

    # Step-by-step extraction
    df = TableGenerator.align_text_positions(pdf_data_subset, x_threshold, y_threshold, align_columns_x, align_columns_y)
    df = TableGenerator.sort_text_by_position(df)
    rows = TableGenerator.group_rows(df, y_threshold)
    normalized_cols, sorted_columns = TableGenerator.determine_column_positions(df, x_threshold)
    return TableGenerator.build_table(rows, normalized_cols, sorted_columns)


# Example Usage
if __name__ == "__main__":
    pdf_data = PDFTextProcessor("sample.pdf")  # Load PDF
    df_subset = pdf_data[pdf_data["section"] == "Some Section"]  # Filter a section

    extracted_table = generate_table_from_dataframe(df_subset)
    print(extracted_table)
