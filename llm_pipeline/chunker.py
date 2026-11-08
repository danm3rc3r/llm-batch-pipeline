"""Text chunking: overlapping character windows cut at whitespace boundaries."""


def chunk_text(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks of at most `size` characters.

    Cuts happen at whitespace boundaries (never mid-word) when possible.
    Consecutive chunks share up to `overlap` characters of context.
    """
    if not (0 <= overlap < size):
        raise ValueError(f"chunk_overlap must be in [0, size), got overlap={overlap}, size={size}")
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            space = text.rfind(" ", start + 1, end)
            if space != -1:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks
