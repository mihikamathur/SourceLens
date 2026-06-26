from src.ingest.loaders import TextRecord
from src.ingest.chunking import chunk_records


def test_chunking_respects_max_size():
    long_text = "Sentence number {}. " * 0
    long_text = " ".join([f"Sentence number {i}." for i in range(200)])
    records = [TextRecord(text=long_text, source="doc.txt", page=None)]

    chunks = chunk_records(records, chunk_size=200, overlap=20)

    assert len(chunks) > 1
    assert all(len(c.text) <= 220 for c in chunks)  # small slack for boundary packing


def test_chunk_ids_are_deterministic_and_unique():
    records = [TextRecord(text="A" * 50 + "\n\n" + "B" * 50, source="doc.txt", page=1)]
    c1 = chunk_records(records, chunk_size=40, overlap=5)
    c2 = chunk_records(records, chunk_size=40, overlap=5)

    assert [c.chunk_id for c in c1] == [c.chunk_id for c in c2]   # deterministic
    assert len({c.chunk_id for c in c1}) == len(c1)               # unique within run


def test_empty_input_produces_no_chunks():
    assert chunk_records([TextRecord(text="   ", source="doc.txt", page=None)]) == []