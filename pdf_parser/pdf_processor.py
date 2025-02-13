import pdfplumber
import pandas as pd
from tqdm import tqdm
from collections import Counter

from pdf_parser.cleaning import merge_sentences_by_position, remove_header_footer_text
from pdf_parser.alignment import align_text_boxes
from pdf_parser.utils import SectionHeader


class PDFTextProcessorUtilsBase:
    """
    Provides all the class specific actions for PDFTextProcessor
    """

    def __str__(self):
        """Returns the entire extracted text with each word on a new line."""
        return "\n".join(self.df["text"].astype(str))

    def __getitem__(self, key):
        """Allows DataFrame-like indexing: pdf_data['column'] or pdf_data[condition]"""
        return PDFTextProcessor.from_dataframe(self.word_data_df[key])

    def __getattr__(self, attr):
        """Delegates attribute access to the internal DataFrame"""
        return getattr(self.df, attr)

    @classmethod
    def from_dataframe(cls, df):
        """Creates a new PDFTextProcessor instance from an existing DataFrame."""
        instance = cls.__new__(cls)  # Create an instance without calling __init__
        instance.pdf_path = None
        instance.word_data_df = df
        return instance

    def text(self):
        """Returns the text of the dataframe, with returns between lines"""
        return self.__str__(self)

    def get_dataframe(self):
        """Returns the processed DataFrame."""
        return self.df

    def save_dataframe(self, save_path: str):
        """Saves the processed DataFrame."""
        self.df.to_csv(save_path, index=False)


class PDFTextProcessor(PDFTextProcessorUtilsBase):
    """
    Processes a PDF file by extracting, cleaning, and aligning text data.

    Attributes:
        pdf_path (str): Path to the PDF file.
        _char_spacing_x (float): Character-level spacing tolerance for `pdfplumber` (x-axis).
        _char_spacing_y (float): Character-level spacing tolerance for `pdfplumber` (y-axis).
        _word_spacing_x (float): Spacing threshold for merging words (x-axis).
        _word_spacing_y (float): Spacing threshold for merging words (y-axis).
        norm_columns (list): List of columns to use for normalization (e.g., ['x0', 'x1', 'x_c']).
        _align_threshold_x (float): Threshold for x-axis normalization.
        _align_threshold_y (float): Threshold for y-axis normalization.
        df (pd.DataFrame): Extracted and processed text data.
    """

    def __init__(
            self,
            file_path: str,

            char_spacing_x: float = 2.0,
            char_spacing_y: float = 1.0,
            word_spacing_x: float = 3.0,
            word_spacing_y: float = 1.5,
            norm_columns_x: list = ["x0", "x1"],
            norm_columns_y: list = ["y0"],
            per_page_x: bool = False,
            per_page_y: bool = True,
            norm_threshold_x: float = 1.5,
            norm_threshold_y: float = 0.75
    ):
        self._char_spacing_x = char_spacing_x
        self._char_spacing_y = char_spacing_y
        self._word_spacing_x = word_spacing_x
        self._word_spacing_y = word_spacing_y

        self._align_columns_x = norm_columns_x
        self._align_columns_y = norm_columns_y

        self._per_page_x = per_page_x
        self._per_page_y = per_page_y

        self._align_threshold_x = norm_threshold_x
        self._align_threshold_y = norm_threshold_y

        if file_path.endswith(".csv"):
            self.df = pd.read_csv(file_path)
        else:
            self.df = self._process_pdf(file_path)

    def _extract_words(self, file_path):
        """Extracts text with positions from the PDF."""
        word_data_dict = []

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in tqdm(enumerate(pdf.pages, start=1), desc="Parsing PDF Text", total=len(pdf.pages)):
                words = page.extract_words(
                    x_tolerance=self._char_spacing_x,
                    y_tolerance=self._char_spacing_y
                )
                char_data = page.chars  # Extract character-level data

                for word in words:
                    colors, font_names, font_sizes = [], [], []

                    # Populate the lists using list comprehension
                    [
                        (colors.append(char.get("non_stroking_color") or char.get("stroking_color")),
                         font_names.append(char.get("fontname")),
                         font_sizes.append(char.get("size")))
                        for char in char_data
                        if word["x0"] <= char["x0"] <= word["x1"] and word["top"] <= char["top"] <= word["bottom"]
                    ]

                    # Determine the dominant values using most_common(1)
                    dominant_color = Counter(colors).most_common(1)[0][0] if colors else None
                    dominant_font = Counter(font_names).most_common(1)[0][0] if font_names else None
                    dominant_size = Counter(font_sizes).most_common(1)[0][0] if font_sizes else None

                    word_data_dict.append({
                        "page": page_num,
                        "text": word["text"],
                        "x0": word["x0"], "x1": word["x1"],
                        "y0": word["top"], "y1": word["bottom"],
                        "color": dominant_color,
                        "font_name": dominant_font,
                        "font_size": dominant_size
                    })

        word_data_df = pd.DataFrame(word_data_dict)
        word_data_df["x_c"] = (word_data_df["x0"] + word_data_df["x1"]) / 2
        word_data_df["y_c"] = (word_data_df["y0"] + word_data_df["y1"]) / 2
        word_data_df["x_align"] = word_data_df["x0"]
        word_data_df["y_align"] = word_data_df["y0"]
        return word_data_df

    def _process_pdf(self, file_path):
        """Processes the PDF into a structured DataFrame."""
        df = self._extract_words(file_path)

        if df.empty:
            return df

        # Align positions of the words (i.e.align text)
        df["x_align"] = align_text_boxes(
            df, self._align_columns_x, alignment_threshold=self._align_threshold_x, per_page=False
        )
        df["y_align"] = align_text_boxes(
            df, self._align_columns_y, alignment_threshold=self._align_threshold_y, per_page=True
        )

        # Merge words that should be joined together
        df = merge_sentences_by_position(
            df, y_tol=self._word_spacing_y, x_tol=self._word_spacing_x
        )

        # Remove header and footer
        df = remove_header_footer_text(df)

        # Rearrange the column order to just be...well...nicer
        desired_order = [
            'page', 'section', 'text',
            'x_align', 'y_align', 'x0', 'x1', 'y0', 'y1', 'x_c', 'y_c',
            'color', 'font_name', 'font_size'
        ]
        return df.reindex(columns=desired_order)

    def assign_sections(self, section_headers: list[SectionHeader], initial_header: str = "Intro"):
        """
        Assigns sections to each row based on detected section titles.

        Parameters:
            header_names (list): List of known section headers.

        Returns:
            pd.DataFrame: Updated DataFrame with a 'section' column.
        """
        current_section = initial_header
        section_list = []

        for _, row in self.df.iterrows():
            for header in section_headers:
                if row["text"] == header.text and row["color"] == header.color:
                    current_section = row["text"]  # Update the section when a new header is found
                    break  # No need to check further, move to the next row

            section_list.append(current_section)

        self.df["section"] = section_list

    def generate_table(self, index_list: list[int]):
        """Generates a table using rows extracted from the dataframe"""
        #Yet to implement
        ...


