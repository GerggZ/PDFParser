import pdfplumber
import pandas as pd
from tqdm import tqdm
from collections import Counter

from pdf_parser.cleaning import merge_sentences_by_position, remove_header_footer_text
from pdf_parser.alignment import align_text_boxes
from pdf_parser.utils import SectionHeader


class _ILOCIndexer:
    """
    A helper class that allows integer-based row/column slicing just like Pandas' df.iloc.
    """
    def __init__(self, pdf_processor):
        self._pdf_processor = pdf_processor

    def __getitem__(self, key):
        # Pandas can handle fancy indexing: single row, slice of rows, or row/col combos
        # e.g. [4:8], [4:8, 1:3], single int, etc.
        selected_data = self._pdf_processor.df.iloc[key]

        # If the result is a single row (a pd.Series), return it directly
        if isinstance(selected_data, pd.Series):
            return selected_data

        # Otherwise it's multiple rows => return a new PDFTextProcessor
        # (or whichever derived class you want).
        return PDFTextProcessor.from_dataframe(selected_data)



class PDFTextProcessorUtilsBase:
    """
    Provides all the class specific actions for PDFTextProcessor
    """


    def __str__(self):
        """Returns the entire extracted text with each word on a new line."""
        return "\n".join(self.df["text"].astype(str))

    def __getitem__(self, key):
        """Allows DataFrame-like indexing: pdf_data['column'] or pdf_data[condition]"""
        if isinstance(key, str):
            return self.df[key]  # Return a Pandas Series directly
        elif isinstance(key, pd.Series) and key.dtype == bool:
            return PDFTextProcessor.from_dataframe(self.df[key].reset_index(drop=True))
        else:
            raise TypeError(f"Invalid indexing type: {type(key)}")

    def __getattr__(self, attr):
        """Delegates attribute access to the internal DataFrame"""
        return getattr(self.df, attr)

    def __eq__(self, other):
        """Allows filtering using '==' directly on PDFTextProcessor."""
        if isinstance(other, str):  # Check if filtering based on a string value
            return self.df.eq(other)  # Returns a DataFrame of boolean values
        return NotImplemented

    def __len__(self):
        return len(self.df)

    def __iter__(self):
        return iter(self.df)

    def __contains__(self, key):
        return key in self.df.columns

    def __setitem__(self, key, value):
        self.df[key] = value

    def __delitem__(self, key):
        del self.df[key]

    @classmethod
    def from_dataframe(cls, df):
        # Dynamically create an instance of whatever class called this method
        instance = cls.__new__(cls)
        instance.df = df.copy()
        instance.pdf_path = None
        return instance

    def text(self):
        """Returns the text of the dataframe, with returns between lines"""
        return self.__str__(self)

    def get_dataframe(self):
        """Returns the processed DataFrame."""
        return self.df

    def reset_index(self, drop=False, inplace=False):
        """Mimics Pandas reset_index while returning a PDFTextProcessor."""
        if inplace:
            self.df.reset_index(drop=drop, inplace=True)
            return self  # Return the *same* object
        else:
            return PDFTextProcessor.from_dataframe(self.df.reset_index(drop=drop))

    @property
    def iloc(self):
        """
        A property that returns an _ILOCIndexer instance to allow integer-based slicing
        like pdf_data.iloc[4:8].
        """
        return _ILOCIndexer(self)

    def drop(self, labels=None, axis=0, inplace=False, errors='raise'):
        """
        Mimics Pandas' drop method to drop specified rows or columns.
        """
        # If the user wants to do in-place, call self.df.drop(...) with inplace=True
        if inplace:
            self.df.drop(labels=labels, axis=axis, inplace=True, errors=errors)
            # For consistency with Pandas, you could return None here
            # but returning self can be convenient for chaining.
            return self
        else:
            # If not in-place, we must create a new PDFTextProcessor
            new_df = self.df.drop(labels=labels, axis=axis, errors=errors)
            return PDFTextProcessor.from_dataframe(new_df)

    def copy(self):
        """Return a new PDFTextProcessor with a copied DataFrame."""
        return PDFTextProcessor.from_dataframe(self.df.copy())

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
            pdf_data_df: pd.DataFrame,
    ):
        self.df = pdf_data_df

    @classmethod
    def from_pdf(
            cls,
            file_path: str,

            char_spacing_x: float = 1.8,
            char_spacing_y: float = 1.0,
            word_spacing_x: float = 2.8,
            word_spacing_y: float = 1.5,

            align_columns_x: list = ["x0", "x1"],
            align_columns_y: list = ["y0"],
            per_page_x: bool = False,
            per_page_y: bool = True,
            align_threshold_x: float = 1.5,
            align_threshold_y: float = 0.75,

            verbose: bool = False
    ):
        pdf_data_df = cls._process_pdf(
            file_path,
            char_spacing_x, char_spacing_y,
            word_spacing_x, word_spacing_y,
            align_columns_x, align_columns_y,
            per_page_x, per_page_y,
            align_threshold_x, align_threshold_y,
            verbose
        )
        return cls(pdf_data_df)

    @classmethod
    def from_csv(cls, file_path):
        pdf_data_df = pd.read_csv(file_path)
        return cls(pdf_data_df)

    @classmethod
    def _extract_words(cls, file_path, char_spacing_x, char_spacing_y, verbose):
        """Extracts text with positions from the PDF."""
        word_data_dict = []

        with pdfplumber.open(file_path) as pdf:
            pages_iter = enumerate(pdf.pages, start=1)
            if verbose:
                pages_iter = tqdm(pages_iter, desc="Parsing PDF Text", total=len(pdf.pages))

            for page_num, page in pages_iter:
                words = page.extract_words(
                    x_tolerance=char_spacing_x,
                    y_tolerance=char_spacing_y
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

    @classmethod
    def _process_pdf(
            cls,
            file_path,
            char_spacing_x, char_spacing_y,
            word_spacing_x, word_spacing_y,
            align_columns_x, align_columns_y,
            per_page_x, per_page_y,
            align_threshold_x, align_threshold_y,
            verbose
    ):
        """Processes the PDF into a structured DataFrame."""
        df = cls._extract_words(
            file_path,
            char_spacing_x, char_spacing_y,
            verbose
        )

        if df.empty:
            return df

        # Align positions of the words (i.e.align text)
        df["x_align"] = align_text_boxes(
            df, align_columns_x, alignment_threshold=align_threshold_x, per_page=per_page_x
        )
        df["y_align"] = align_text_boxes(
            df, align_columns_y, alignment_threshold=align_threshold_y, per_page=per_page_y
        )

        # Merge words that should be joined together
        df = merge_sentences_by_position(
            df, y_tol=word_spacing_y, x_tol=word_spacing_x
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
        # Create a boolean mask by checking if each row matches a SectionHeader object
        header_mask = self.df.apply(lambda row: SectionHeader(row["text"], row["color"]) in section_headers, axis=1)

        # Initialize `section` with NaN and set only the first row to `initial_header`
        self.df["section"] = [initial_header] + [pd.NA] * (len(self.df) - 1)

        # Assign headers where mask is True
        self.df.loc[header_mask, "section"] = self.df.loc[header_mask, "text"]

        # Forward-fill section values
        self.df["section"] = self.df["section"].ffill()

    def generate_table(self, index_list: list[int]):
        """Generates a table using rows extracted from the dataframe"""
        #Yet to implement
        ...
        print('we here!')
        return -7



