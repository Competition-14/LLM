from typing import Iterator, List
from pathlib import Path

from tqdm import tqdm
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


class DirLoader(TextLoader):
    """Load all text files from a directory."""
    path: str
    docs: List[Document]

    def __init__(self, path: str):
        self.path = path

    def load(self) -> str:
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """Load documents lazily."""
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"Directory not found: '{self.path}'")
        if not p.is_dir():
            raise ValueError(f"Expected directory, got file: '{self.path}'")

        # load all files in the directory, and show progress
        files: List[Path] = list(p.iterdir())
        for file in tqdm(files, desc="Loading files"):
            if file.is_file():
                yield Document(page_content=file.read_text(), metadata={"source": str(file)})