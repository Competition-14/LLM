import re
from typing import List

from langchain_core.documents import Document
from overrides import override
from langchain_text_splitters import TextSplitter


class TableSplitter(TextSplitter):
    max_length: str

    def __init__(self, max_length: int = None) -> None:
        super().__init__()
        self.max_length = 10000
        if max_length is not None:
            self.max_length = max_length

    def check_length(self, text: str) -> bool:
        return len(text) <= self.max_length

    @override
    def split_text(self, text: str) -> List[str]:
        """Split text into multiple components."""
        # define regex pattern to match tables
        pattern = re.compile(r'(.{0,20})(<table.*?>.*?</table>)(.{0,20})', re.DOTALL)
        # find all tables in text
        matches: List[str] = pattern.findall(text)
        tables = []
        for match in matches:
            before, table, after = match

            if "</table>" in before:
                # 删除</table>和其之前的内容
                before = before.split("</table>")[-1]

            if "<table>" in after:
                # 删除<table>和其之后的内容
                after = after.split("<table>")[0]

            # 组合前10个字符、表格和后10个字符
            new_table = before + table + after
            if self.check_length(new_table):
                tables.append(new_table)
        return tables

    @override
    def create_documents(self, texts: List[str], metadatas: List[dict] | None = None) -> List[Document]:
        # set metadata "type" as "table"
        metadatas: List[dict] = TableSplitter.set_metadata(metadatas)
        documents: List[Document] = super().create_documents(texts, metadatas)
        for i, doc in enumerate(documents):
            doc.metadata["index"] = i
        return documents

    @staticmethod
    def set_metadata(metadatas: List[dict]) -> List[dict]:
        """Set metadata for the table splitter."""
        for metadata in metadatas:
            metadata["type"] = "table"
        return metadatas

