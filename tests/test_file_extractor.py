from io import BytesIO

import pytest
from docx import Document
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.services.file_extractor import extract_text_from_file


def test_extract_plain_text_file() -> None:
    extracted = extract_text_from_file(
        "article.txt",
        "Ministerstwo opublikowało raport z danymi. Tekst zawiera liczby, źródła i spokojny opis sprawy.".encode(
            "utf-8"
        ),
    )

    assert extracted.file_type == "txt"
    assert extracted.extraction_method == "plain-text"
    assert "raport" in extracted.content


def test_extract_docx_file() -> None:
    document = Document()
    document.add_paragraph("Agencja opublikowała raport o jakości wody.")
    document.add_paragraph("Tekst zawiera dane, daty, ekspertów i linki do źródeł publicznych.")
    buffer = BytesIO()
    document.save(buffer)

    extracted = extract_text_from_file("article.docx", buffer.getvalue())

    assert extracted.file_type == "docx"
    assert extracted.extraction_method == "python-docx"
    assert "jakości wody" in extracted.content


def test_extract_pdf_file() -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=200)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 50 150 Td (Agency published a report with data and experts for verification.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = BytesIO()
    writer.write(buffer)

    extracted = extract_text_from_file("article.pdf", buffer.getvalue())

    assert extracted.file_type == "pdf"
    assert extracted.extraction_method == "pypdf"
    assert "Agency published a report" in extracted.content


def test_rejects_unsupported_file_type() -> None:
    with pytest.raises(ValueError, match="Nieobslugiwany typ pliku"):
        extract_text_from_file("archive.zip", b"not an article")
