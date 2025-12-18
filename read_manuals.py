from pathlib import Path
from pypdf import PdfReader
import warnings

DATA_DIR = Path("data")

# -------------------------------
# Safe text extraction
# -------------------------------
def extract_text_safe(pdf_path: Path) -> str:
    text = ""

    try:
        reader = PdfReader(pdf_path, strict=False)

        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            except Exception as e:
                print(f"  ⚠️ Skipping page {i+1}: {e}")

    except Exception as e:
        print(f"  ❌ Failed to read {pdf_path.name}: {e}")

    return text


# -------------------------------
# Metadata from folder structure
# -------------------------------
def get_metadata(pdf_path: Path):
    parts = pdf_path.parts
    return {
        "brand": parts[-3],
        "category": parts[-2],
        "model": pdf_path.stem
    }


# -------------------------------
# Main
# -------------------------------
def main():
    pdf_files = list(DATA_DIR.rglob("*.pdf"))
    print(f"\nFound {len(pdf_files)} PDF manuals\n")

    for pdf in pdf_files:
        print(f"--- Reading {pdf.name} ---")

        text = extract_text_safe(pdf)
        metadata = get_metadata(pdf)

        # Show preview only (avoid flooding terminal)
        preview = text[:800].replace("\n", " ")
        print(preview)
        print("\nMetadata:", metadata)
        print("-" * 60)


if __name__ == "__main__":
    # Suppress noisy PDF warnings
    warnings.filterwarnings("ignore")
    main()
