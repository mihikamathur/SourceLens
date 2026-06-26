from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class TextRecord:
    text: str
    source: str          # filename
    page: int | None      # 1-indexed page, None if not paginated (e.g. txt)


@dataclass
class ImageRecord:
    image_bytes: bytes
    ext: str              # "png", "jpeg", ...
    source: str
    page: int
    image_index: int      # position of image within the page


def load_pdf(path: Path) -> tuple[list[TextRecord], list[ImageRecord]]:
    import fitz  # PyMuPDF

    text_records: list[TextRecord] = []
    image_records: list[ImageRecord] = []

    doc = fitz.open(path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        page_text = page.get_text("text").strip()
        if page_text:
            text_records.append(
                TextRecord(text=page_text, source=path.name, page=page_index + 1)
            )

        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue  # skip unreadable/corrupt embedded images
            image_records.append(
                ImageRecord(
                    image_bytes=base_image["image"],
                    ext=base_image.get("ext", "png"),
                    source=path.name,
                    page=page_index + 1,
                    image_index=img_idx,
                )
            )
    doc.close()
    return text_records, image_records


def load_docx(path: Path) -> list[TextRecord]:
    import docx  # python-docx
    d = docx.Document(path)
    full_text = "\n".join(p.text for p in d.paragraphs if p.text.strip())
    # DOCX has no native page concept pre-render; we treat the whole file as page=None
    return [TextRecord(text=full_text, source=path.name, page=None)] if full_text else []


def load_txt(path: Path) -> list[TextRecord]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [TextRecord(text=text, source=path.name, page=None)] if text else []


def load_directory(input_dir: Path) -> tuple[list[TextRecord], list[ImageRecord]]:
    """Walk a directory and load every supported file."""
    all_text: list[TextRecord] = []
    all_images: list[ImageRecord] = []

    for path in sorted(Path(input_dir).rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            t, i = load_pdf(path)
            all_text.extend(t)
            all_images.extend(i)
        elif suffix == ".docx":
            all_text.extend(load_docx(path))
        elif suffix == ".txt":
            all_text.extend(load_txt(path))
        # silently skip unsupported types; log if you want stricter behavior

    return all_text, all_images


def iter_supported_files(input_dir: Path) -> Iterator[Path]:
    for path in sorted(Path(input_dir).rglob("*")):
        if path.is_file() and path.suffix.lower() in {".pdf", ".docx", ".txt"}:
            yield path