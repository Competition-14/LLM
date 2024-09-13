import re
from typing import List, Optional, Any

from langchain_core.documents import Document
from overrides import override
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .menu_splitter import MenuSplitter


class TextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = False,
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(separators=separators,
                         keep_separator=keep_separator,
                         is_separator_regex=is_separator_regex,
                         **kwargs)
        self.page_index: int = 0

    def set_page_index(self, page_index: int, check: bool = True) -> None:
        if check and self.page_index > page_index:
            raise ValueError("page_index must be greater than current page_index")
        self.page_index = page_index

    @override
    def split_text(self, text: str) -> List[str]:
        # if table exists, remove it
        if "<table>" in text:
            # remove tables
            text = re.sub(r"<table>(.*?)</table>", "", text, flags=re.DOTALL)
        # menu
        if "<menu>" in text:
            self.menu_splitter = MenuSplitter()
            menu_doc: Document = self.menu_splitter.create_documents([text])[0]
            self.menu_splitter.parse_menu(menu_doc)
            text = re.sub(r"<menu>(.*?)</menu>", "", text, flags=re.DOTALL)
        else:
            self.menu_splitter = None
        return super().split_text(text)

    @override
    def create_documents(self, texts: List[str], metadatas: List[dict] | None = None) -> List[Document]:
        # set metadata "type" as "text"
        metadatas: List[dict] = TextSplitter.set_metadata(metadatas)
        documents: List[Document] = super().create_documents(texts, metadatas)
        for doc in documents:
            self.set_page_index(0, check=False)
            # 使用正则找到内容格式是`Page4:`的标记
            # 获得标记的数字部分
            match = re.search(r"Page(\d+):", doc.page_content)
            if match:
                self.set_page_index(int(match.group(1)))
                # 删除标记内容
                doc.page_content = re.sub(r"Page\d+:", "", doc.page_content)
            # 如果内容是空的
            if doc.page_content.strip() == "":
                continue
            doc.metadata["index"] = self.page_index
            # set title by using menu splitter
            if self.menu_splitter:
                doc.page_content = self.menu_splitter.get_title_from_page(self.page_index) + doc.page_content
        return [doc for doc in documents if doc.page_content.strip() != ""]

    @staticmethod
    def set_metadata(metadatas: List[dict]) -> List[dict]:
        """Set metadata for the text splitter."""
        for metadata in metadatas:
            metadata["type"] = "text"
        return metadatas