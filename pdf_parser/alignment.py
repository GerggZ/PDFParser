import pandas as pd

class TextAlignment:
    """
    Provides alignment functionality for text boxes in a DataFrame.
    """

    @staticmethod
    def group_nearby_positions(series, alignment_threshold: float):
        """
        Groups text boxes that are close to each other within the alignment threshold.

        :param series: Pandas Series containing position values.
        :param alignment_threshold: Maximum difference allowed for alignment.
        :return: List of sets where each set contains indices of aligned values.
        """
        groups = []
        visited = set()

        for i, val in enumerate(series):
            if i in visited:
                continue
            group = {i}
            for j, comp_val in enumerate(series):
                if j != i and abs(val - comp_val) <= alignment_threshold:
                    group.add(j)
            visited.update(group)
            groups.append(group)

        return groups

    @staticmethod
    def merge_groups(groups):
        """
        Merges overlapping groups of aligned text boxes into unique sets.

        :param groups: List of sets containing grouped indices.
        :return: Merged groups ensuring unique sets.
        """
        merged = []
        index_map = {}

        for group in groups:
            found = None
            for idx in group:
                if idx in index_map:
                    found = index_map[idx]
                    break

            if found is not None:
                merged[found].update(group)
                for idx in group:
                    index_map[idx] = found
            else:
                merged.append(set(group))
                group_index = len(merged) - 1
                for idx in group:
                    index_map[idx] = group_index

        return [list(g) for g in merged if g]

    @staticmethod
    def compute_alignment_values(groups, df, position_columns):
        """
        Computes an average alignment value for each group of aligned text boxes.

        :param groups: List of sets containing grouped indices.
        :param df: Input DataFrame containing text positions.
        :param position_columns: List of column names to use for averaging.
        :return: Dictionary mapping each index to its computed alignment value.
        """
        group_values = {}
        for group in groups:
            averages = {col: df.loc[group, col].mean() for col in position_columns}
            overall_avg = sum(averages.values()) / len(averages) if averages else 0
            for idx in group:
                group_values[idx] = overall_avg
        return group_values

def align_text_boxes(df: pd.DataFrame, position_columns: list, alignment_threshold: float, per_page: bool = False) -> pd.Series:
    """
    Identifies text boxes that are aligned either horizontally (same line) or vertically (same column).

    :param df: Input DataFrame containing text box positions.
    :param position_columns: List of column names representing positions (e.g., ['x0', 'x1'] or ['y0']).
    :param alignment_threshold: Maximum difference allowed for text boxes to be considered aligned.
    :param per_page: If True, only compares text boxes within the same page.
    :return: Pandas Series where each entry corresponds to the alignment value for that text box.
    """
    aligned_groups = []

    # Determine aligned text boxes per page or globally
    if per_page and 'page' in df.columns:
        for _, page_df in df.groupby('page'):
            for col in position_columns:
                related_groups = TextAlignment.group_nearby_positions(page_df[col], alignment_threshold)
                aligned_groups.extend([set(page_df.index[list(g)]) for g in related_groups])
    else:
        for col in position_columns:
            related_groups = TextAlignment.group_nearby_positions(df[col], alignment_threshold)
            aligned_groups.extend([set(df.index[list(g)]) for g in related_groups])

    aligned_text_groups = TextAlignment.merge_groups(aligned_groups)

    group_values = TextAlignment.compute_alignment_values(aligned_text_groups, df, position_columns)

    # Assign alignment values back to the original DataFrame row positions
    return df.index.to_series().map(lambda idx: group_values.get(idx, None))
