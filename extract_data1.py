from pdf_parser.pdf_processor import PDFTextProcessor
from pdf_parser.utils import SectionHeader
from pdf_parser.table_generator import generate_table_from_dataframe
from pdf_parser.table_extractor_2 import TableExtractor

import pandas as pd


# List of Section Headers (Now Without Repeating (1,1,1) Every Time)
EQUIFAX_FILE_HEADER_NAMES: list[SectionHeader] = [
    SectionHeader('Sabio Empresarial'),
    SectionHeader('Consulta Rápida'),
    SectionHeader('Información General'),
    SectionHeader('Datos Generales'),
    SectionHeader('Datos Principales'),
    SectionHeader('Otros Reportes Negativos'),
    SectionHeader('Detalle Variación'),
    SectionHeader('Posición Histórica'),
    SectionHeader('Gráficos'),
    SectionHeader('Reporte de Vencidos'),
    SectionHeader('Reporte SBS/Microfinanzas'),
    SectionHeader('Reporte SBS/Microfinanzas por entidades'),
    SectionHeader('Aval/Codeudor'),
    SectionHeader('Es Avalista de:'),
    SectionHeader('Quiénes lo Avalan:'),
    SectionHeader('Documentos Protestados como Deudor/Aceptante'),
    SectionHeader('Comercio Exterior')
]

def get_posicion_historica_table(df):
    posicion_historica_df = df[df['section'] == 'Posición Histórica'].reset_index(drop=True)

    extractor = TableExtractor(posicion_historica_df)
    table_df = extractor.extract_table_str(
        start_regex_tuple=(2, r"Cte.\nCréd.\n"),
        end_regex_tuple=None,
    )

    extracted_table_df = generate_table_from_dataframe(table_df, y_threshold=8, x_threshold=15)

    return extracted_table_df


def get_microfinanzas_table(pdf_data):
    microfinanzas_table_df = pdf_data.df[pdf_data.df['section'] == 'Reporte SBS/Microfinanzas'].reset_index(drop=True)

    start_regex = {
        r"Dias\n": 6,
        r"V\.\n": 6,
        r"Monto\n": 6,
        r"\*\n": 6
    }
    end_regex = r"Última fecha reportada por la SBS :\n"

    extractor = TableExtractor(microfinanzas_table_df)
    tables = extractor.extract_table_dict(
        start_regex_tuple=(24, start_regex),
        end_regex_tuple=(-1, end_regex),
    )

    extracted_1st_half = generate_table_from_dataframe(tables[0], y_threshold=8, x_threshold=15)
    extracted_2nd_half = generate_table_from_dataframe(tables[-1], y_threshold=8, x_threshold=15)\

    if not extracted_1st_half.iloc[:, 0].equals(extracted_2nd_half.iloc[:, 0]):
        raise ValueError("Microfinance data between last 6 months and previous 6 months not compatabale")

    extracted_1st_half = extracted_1st_half.iloc[:, 1:]  # Remove the first column from df2
    table_extracted = pd.concat([extracted_2nd_half, extracted_1st_half], axis=1)

    return table_extracted

# Initialize processor
# pdf_data = PDFTextProcessor.from_csv("sample.csv")

# Assign section titles dynamically
pdf_path = r"C:\Users\gregg\Documents\2024 Jobs\HipoTek\PDF2ALVARADOCASTROCARLOSAUGUSTO25012025201013[1].pdf"
# pdf_data = PDFTextProcessor.from_pdf(pdf_path, verbose=True)

# pdf_data = PDFTextProcessor.from_pdf("sample.pdf", verbose=True)
# pdf_data.assign_sections(section_headers=EQUIFAX_FILE_HEADER_NAMES, initial_header="Comienza")
# pdf_data.to_csv("sample.csv", index=False)
pdf_data = PDFTextProcessor.from_csv("sample.csv")

results_hist = get_posicion_historica_table(pdf_data)
results_micro = get_microfinanzas_table(pdf_data)

j = 7
