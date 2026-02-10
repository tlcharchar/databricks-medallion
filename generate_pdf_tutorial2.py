"""
Gera PDF do Tutorial 2 (ML) reutilizando a engine do generate_pdf.py.
"""
import sys
sys.path.insert(0, ".")
from generate_pdf import TutorialPDF, parse_and_render, sanitize_unicode

MD_FILE = "tutorial_02_ml_databricks.md"
PDF_FILE = "Tutorial_02_ML_Databricks.pdf"


def create_cover_ml(pdf):
    """Cria pagina de capa para Tutorial 2 (ML)."""
    pdf.add_page()

    # Fundo decorativo topo
    pdf.set_fill_color(46, 134, 193)  # Azul ML
    pdf.rect(0, 0, 210, 6, "F")
    pdf.set_fill_color(44, 62, 80)
    pdf.rect(0, 6, 210, 3, "F")

    # Titulo
    pdf.ln(50)
    pdf.set_font("Helvetica", "B", 30)
    pdf.set_text_color(44, 62, 80)
    pdf.multi_cell(0, 14, "Tutorial 2: Machine\nLearning", align="C")

    pdf.ln(5)
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "Databricks Free Edition", align="C")

    # Linha decorativa
    pdf.ln(15)
    pdf.set_draw_color(46, 134, 193)
    pdf.set_line_width(1)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())

    # Subtitulo
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 7, "Feature Engineering, MLflow, Model Registry\ne Batch Inference", align="C")

    # Diagrama
    pdf.ln(20)
    pdf.set_font("Courier", "B", 11)
    pdf.set_text_color(46, 134, 193)
    pdf.cell(0, 8, "Features >>> Train >>> Register >>> Predict", align="C")

    # Info
    pdf.ln(30)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, "Autor: Thiago Charchar & Claude AI", align="C")
    pdf.ln(7)
    pdf.cell(0, 7, "Data: Fevereiro 2026", align="C")
    pdf.ln(7)
    pdf.cell(0, 7, "github.com/tlcharchar/databricks-medallion", align="C")
    pdf.ln(7)
    pdf.cell(0, 7, "Pre-requisito: Tutorial 1 (Medallion Architecture)", align="C")

    # Rodape decorativo
    pdf.set_fill_color(44, 62, 80)
    pdf.rect(0, 287, 210, 3, "F")
    pdf.set_fill_color(46, 134, 193)
    pdf.rect(0, 290, 210, 7, "F")


def main():
    with open(MD_FILE, "r", encoding="utf-8") as f:
        md_content = f.read()

    # Pular ate Sumario
    lines = md_content.split("\n")
    start = 0
    for idx, line in enumerate(lines):
        if line.strip().startswith("## Sum"):
            start = idx
            break
    md_body = "\n".join(lines[start:]) if start > 0 else md_content

    pdf = TutorialPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    create_cover_ml(pdf)
    pdf.add_page()
    parse_and_render(pdf, md_body)

    pdf.output(PDF_FILE)
    print(f"PDF gerado com sucesso: {PDF_FILE}")
    print(f"Paginas: {pdf.page_no()}")


if __name__ == "__main__":
    main()
