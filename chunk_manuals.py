from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

DATA_DIR = Path("data")
OUT_DIR = Path("data/chunks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        try:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        except:
            continue
    return text

def get_metadata(pdf_path: Path):
    parts = pdf_path.parts
    return {
        "brand": parts[-3],
        "category": parts[-2],
        "model": pdf_path.stem,
        "source": str(pdf_path)
    }

def main():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    pdfs = list(DATA_DIR.rglob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs")

    all_chunks = []

    for pdf in pdfs:
        print(f"Processing: {pdf.name}")
        text = extract_text(pdf)

        if not text.strip():
            continue

        metadata = get_metadata(pdf)
        chunks = splitter.split_text(text)

        for c in chunks:
            all_chunks.append({
                "text": c,
                "metadata": metadata
            })

    # 🔑 SAVE TO DISK
    out_file = OUT_DIR / "chunks.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_chunks)} chunks → {out_file}")

if __name__ == "__main__":
    main()
