#!/usr/bin/env python3
"""CS336 Lecture PDF to Markdown Parser.
Extracts text and tables from lecture PDFs, maintaining page breaks.
"""

import argparse
import os
import sys
import pdfplumber

def is_garbled(text: str, threshold: float = 0.30) -> bool:
    """Detect garbled text (math formulas often trigger this if not careful, so we set threshold high)."""
    if not text: return True
    normal = sum(1 for ch in text if 0x20 <= ord(ch) <= 0x7E or ord(ch) > 0x0100)
    return (normal / len(text)) < (1 - threshold)

def tables_to_markdown(tables: list) -> str:
    """Convert pdfplumber tables to markdown format."""
    parts = []
    for table in tables:
        if not table or len(table) < 2: continue
        cleaned = [[(str(cell) or "").replace("\n", " ").strip() for cell in row] for row in table]
        header = cleaned[0]
        md = "| " + " | ".join(header) + " |\n| " + " | ".join(["---"] * len(header)) + " |\n"
        for row in cleaned[1:]:
            while len(row) < len(header): row.append("")
            md += "| " + " | ".join(row[:len(header)]) + " |\n"
        parts.append(md)
    return "\n".join(parts)

def fallback_extract(pdf_path: str) -> list:
    """Fallback using PyMuPDF (fitz)."""
    import fitz
    doc = fitz.open(pdf_path)
    return [(i + 1, doc[i].get_text()) for i in range(len(doc))]

def parse_pdf_to_md(pdf_path: str, output_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    pages_text = []
    garbled_count = 0
    total = 0
    
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"Extracting {total} pages with pdfplumber...")
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables()
            if tables:
                text += "\n\n[TABLE]\n" + tables_to_markdown(tables)
            
            if is_garbled(text) and len(text) > 50:
                garbled_count += 1
            pages_text.append((i + 1, text))

    if total > 0 and garbled_count / total > 0.30:
        print("High garbled rate detected. Falling back to PyMuPDF...")
        pages_text = fallback_extract(pdf_path)

    # Write to Markdown with clear page boundaries for the LLM
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for page_num, text in pages_text:
            f.write(f"\n\n--- Page {page_num} ---\n\n")
            f.write(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    parse_pdf_to_md(args.pdf, args.output)
    print(f"Converted {args.pdf} to {args.output}")