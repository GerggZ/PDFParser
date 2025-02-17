import bisect

import pandas as pd
import re
import warnings
from collections import deque


class TableExtractor:
    def __init__(self, df: pd.DataFrame):
        if "text" not in df.columns:
            raise ValueError("DataFrame must contain a 'text' column.")
        self.df = df.copy()  # Work with a copy to avoid modifying the original
        self.row_text_map = list(zip(df.index, df["text"]))
        self.full_text = "\n".join(text for _, text in self.row_text_map)

    def _find_position_str(self, regex_tuple):
        """Finds the row index of the first match based on the given regex pattern."""
        if not regex_tuple:
            return None

        offset, pattern = regex_tuple
        match = re.search(pattern, self.full_text, re.DOTALL)  # Multi-line search
        if match:
            start_pos = match.start()
            if start_pos < len(self.char_to_row):  # Ensure valid mapping
                found_row = self.char_to_row[start_pos]  # Convert character position to row index
                row_index = self._apply_offset(found_row, offset)
                return row_index
        return None

    def _find_position_dict(self, regex_tuple):
        """
        Uses a sliding window to find a block where the required regex patterns appear exactly as specified.

        :param regex_tuple: Tuple containing:
            - offset (row index offset)
            - regex_counts (dictionary where keys are regex patterns (including `\n`) and values are expected counts)
        :return: Row index if a valid match is found, otherwise None.
        """
        if not regex_tuple:
            return None

        offset, regex_counts = regex_tuple  # Extract offset and regex count dictionary
        if not regex_counts:
            return None

        matches = []

        # Convert regex keys to compiled regex objects
        compiled_regex = {re.compile(key): key for key in regex_counts.keys()}

        # Read the full text and split into lines
        lines = self.full_text.strip().split("\n")

        # If there are fewer than the required lines, return None early
        total_required_lines = sum(regex_counts.values())
        if len(lines) < total_required_lines:
            return None

        # Sliding window setup
        window = deque()  # Circular buffer holding the last N lines
        counts = {key: 0 for key in regex_counts}  # Track occurrences within the window

        # Scan through lines
        for i, line in enumerate(lines):
            line_with_newline = line + "\n"

            # Match the current line against known regexes
            matched = None
            for pattern, key in compiled_regex.items():
                if pattern.fullmatch(line_with_newline):
                    matched = key
                    break

            if matched is None:
                # Invalid line, reset the buffer
                window.clear()
                counts = {key: 0 for key in regex_counts}
                continue

            # Add the new line to the sliding window
            window.append(matched)
            counts[matched] += 1

            # Remove oldest line if window exceeds the required size
            if len(window) > total_required_lines:
                removed = window.popleft()
                counts[removed] -= 1

            # Check if we have a valid match
            if len(window) == total_required_lines and all(counts[key] == regex_counts[key] for key in regex_counts):
                start_pos = i - total_required_lines + 1
                if start_pos < len(self.row_text_map):
                    found_row = self.row_text_map[start_pos][0]
                    matches.append(self._apply_offset(found_row, offset))

        if not len(matches):
            # No valid match found – issue a warning
            warnings.warn(
                f"No valid match found for regex pattern block. Expected occurrences: {regex_counts}",
                RuntimeWarning
            )

        return matches

    def _apply_offset(self, row_index, offset):
        """Applies an offset to the found row index while ensuring it's within valid bounds."""
        row_loc = self.df.index.get_loc(row_index)
        new_loc = row_loc + offset
        return self.df.index[new_loc] if 0 <= new_loc < len(self.df) else None

    def _find_end_idx(self, end_idxs, start_idx):
        idx = bisect.bisect_right(end_idxs, start_idx)
        return end_idxs[idx] if idx < len(end_idxs) else None

    def extract_table_dict(
            self,
            start_regex_tuple: None | tuple[int, dict[str, int]],
            end_regex_tuple: None | tuple[int, dict[str, int]],
            drop_regex_tuples: None | list[tuple[int, dict[str, int]]]
    ):
        # Convert start_regex and end_regex to dicts if necessary
        if start_regex_tuple is not None:
            if isinstance(start_regex_tuple[1], str):
                start_regex = (start_regex_tuple[0], {start_regex_tuple[1]: 1})
        if end_regex_tuple is not None:
            if isinstance(end_regex_tuple[1], str):
                end_regex = (end_regex_tuple[0], {end_regex_tuple[1]: 1})

        start_idxs = self._find_position_dict(start_regex)
        end_idxs = self._find_position_dict(end_regex)

        extracted_table = self._extract_tables(start_idxs, end_idxs)
        return extracted_table

    def extract_table_str(
            self,
            start_regex_tuple: None | tuple[int, str],
            end_regex_tuple: None | tuple[int, str],
    ):
        start_idxs = self._find_position_str(start_regex_tuple)
        end_idxs = self._find_position_str(end_regex_tuple)

        extracted_table = self._extract_tables(start_idxs, end_idxs)
        return extracted_table

    def _extract_tables(self, start_regex, end_regex, start_idxs: list[int], end_idxs: list[int]):
        """Extracts a table from the DataFrame based on start, end, and drop regex patterns."""
        extracted_dfs = []
        for start_idx in start_idxs:
            end_idx = self._find_end_idx(end_idxs, start_idx)

            if start_idx is not None and end_idx is not None:
                extracted_df = self.df.loc[start_idx:end_idx]
            elif start_idx is not None and end_regex is None:
                extracted_df = self.df.loc[start_idx:]
            elif start_idx is None and end_idx is not None:
                extracted_df = self.df.loc[:end_idx]
            else:
                continue  # If no start/end criteria, there was nothing to extract :)

            extracted_dfs.append(extracted_df)

        return extracted_dfs


def extract_table_from_dataframe(df, start_regex=None, end_regex=None, drop_regexes=None):
    extractor = TableExtractor(df)
    return extractor.extract_table(start_regex, end_regex, drop_regexes)
