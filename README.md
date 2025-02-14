# **PDFTextProcessor**

## **Overview**  
`PDFTextProcessor` is a Python library for extracting, cleaning, and processing text from PDF documents. Built on top of `pdfplumber` and `pandas`, it extends `pd.DataFrame` to provide additional functionality for working with structured PDF text data.  

This project is designed for professionals working with document parsing, text alignment, and structured data extraction from PDFs, offering efficient methods for text processing while maintaining a familiar DataFrame interface.  

## **Features**  
- **PDF Extraction**: Parses text from PDFs while preserving position metadata.  
- **DataFrame Extension**: Behaves like a `pandas.DataFrame`, allowing for easy integration with existing workflows.  
- **Text Alignment**: Adjusts text box positioning using configurable alignment thresholds.  
- **Header & Footer Removal**: Cleans PDFs by removing unwanted header/footer text.  
- **Word Merging**: Merges words based on spacing constraints to reconstruct sentences.  
- **Section Assignment**: Automatically categorizes text into sections using predefined headers.  
- **CSV Support**: Load and save extracted data in CSV format.  

## **Installation**  
```bash
pip install pdfplumber pandas tqdm
```
or install directly from source:  
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/PDFTextProcessor.git
cd PDFTextProcessor
pip install -r requirements.txt
```

## **Usage**  
### **Basic Example**  
```python
from pdf_processor import PDFTextProcessor

# Load and process a PDF file
pdf_processor = PDFTextProcessor("example.pdf")

# View extracted text
print(pdf_processor.head())

# Get the entire document as a single text string
text = pdf_processor.get_text()

# Assign sections based on predefined headers
from pdf_parser.utils import SectionHeader
headers = [SectionHeader("Introduction", "black"), SectionHeader("Conclusion", "blue")]
pdf_processor.assign_sections(headers)

# Save processed data to CSV
pdf_processor.to_csv("output.csv", index=False)
```

## **Customization**  
### **Modifying Alignment Thresholds**  
```python
pdf_processor = PDFTextProcessor("example.pdf", norm_threshold_x=2.0, norm_threshold_y=1.0)
```
### **Extracting Only Certain Pages**  
```python
pdf_processor = PDFTextProcessor("example.pdf")
subset = pdf_processor[pdf_processor["page"] <= 5]  # Extract first five pages
```

## **Implementation Details**  
- **Extends `pd.DataFrame`** for seamless integration with Pandas operations.  
- **Uses `pdfplumber`** for precise text extraction.  
- **Overrides key Pandas methods** to maintain expected DataFrame behavior.  
- **Avoids recursion issues** by carefully handling attribute assignments.  

## **Development**  
### **Requirements**  
- Python **3.9+**  
- `pdfplumber`, `pandas`, `tqdm`  
# **PDFTextProcessor**

## **Overview**  
`PDFTextProcessor` is a Python library for extracting, cleaning, and processing text from PDF documents. Built on top of `pdfplumber` and `pandas`, it extends `pd.DataFrame` to provide additional functionality for working with structured PDF text data.  

This project is designed for professionals working with document parsing, text alignment, and structured data extraction from PDFs, offering efficient methods for text processing while maintaining a familiar DataFrame interface.  

## **Features**  
- **PDF Extraction**: Parses text from PDFs while preserving position metadata.  
- **DataFrame Extension**: Behaves like a `pandas.DataFrame`, allowing for easy integration with existing workflows.  
- **Text Alignment**: Adjusts text box positioning using configurable alignment thresholds.  
- **Header & Footer Removal**: Cleans PDFs by removing unwanted header/footer text.  
- **Word Merging**: Merges words based on spacing constraints to reconstruct sentences.  
- **Section Assignment**: Automatically categorizes text into sections using predefined headers.  
- **CSV Support**: Load and save extracted data in CSV format.  

## **Installation**  
```bash
pip install pdfplumber pandas tqdm
```
or install directly from source:  
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/PDFTextProcessor.git
cd PDFTextProcessor
pip install -r requirements.txt
```

## **Usage**  
### **Basic Example**  
```python
from pdf_processor import PDFTextProcessor

# Load and process a PDF file
pdf_processor = PDFTextProcessor("example.pdf")

# View extracted text
print(pdf_processor.head())

# Get the entire document as a single text string
text = pdf_processor.get_text()

# Assign sections based on predefined headers
from pdf_parser.utils import SectionHeader
headers = [SectionHeader("Introduction", "black"), SectionHeader("Conclusion", "blue")]
pdf_processor.assign_sections(headers)

# Save processed data to CSV
pdf_processor.to_csv("output.csv", index=False)
```

## **Customization**  
### **Modifying Alignment Thresholds**  
```python
pdf_processor = PDFTextProcessor("example.pdf", norm_threshold_x=2.0, norm_threshold_y=1.0)
```
### **Extracting Only Certain Pages**  
```python
pdf_processor = PDFTextProcessor("example.pdf")
subset = pdf_processor[pdf_processor["page"] <= 5]  # Extract first five pages
```

## **Implementation Details**  
- **Extends `pd.DataFrame`** for seamless integration with Pandas operations.  
- **Uses `pdfplumber`** for precise text extraction.  
- **Overrides key Pandas methods** to maintain expected DataFrame behavior.  
- **Avoids recursion issues** by carefully handling attribute assignments.  

## **Development**  
### **Requirements**  
- Python **3.9+**  
- `pdfplumber`, `pandas`, `tqdm`

### **Contributing**  
Contributions are welcome. Please open an issue or submit a pull request for improvements.  



### **Contributing**  
Contributions are welcome. Please open an issue or submit a pull request for improvements.