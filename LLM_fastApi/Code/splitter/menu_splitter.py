import re
from typing import List, Dict, Any

from langchain_core.documents import Document
from overrides import override

from langchain_text_splitters import TextSplitter


class MenuSplitter(TextSplitter):
    """
    A class to split a menu into different sections based on the given pattern.
    """

    def __init__(self):
        """
        Initializes the MenuSplitter with the given pattern and section names.

        Args:
            pattern (str): The regex pattern to split the menu.
        """
        super().__init__()
        self.pattern = re.compile(r'<menu.*?>.*?</menu>', re.DOTALL)

    @override
    def split_text(self, text: str) -> List[str]:
        menu: str = self.pattern.findall(text)[0]
        return [menu]

    @override
    def create_documents(self, texts: List[str], metadatas: List[dict] | None = None) -> List[Document]:
        # set metadata "type" as "menu"
        if metadatas is None:
            metadatas = [{} for _ in texts]
        else:
            metadatas: List[dict] = MenuSplitter.set_metadata(metadatas)
        return super().create_documents(texts, metadatas)

    @staticmethod
    def set_metadata(metadatas: List[dict]) -> List[dict]:
        """Set metadata for the text splitter."""
        for metadata in metadatas:
            metadata["type"] = "menu"
        return metadatas

    def parse_menu(self, menu: Document):
        """
        Parse the menu and return a list of dictionaries.
        """
        # 正则表达式匹配规则
        pattern = re.compile(r'(.+节|\d+\.\d+)\s+([^\d]+)\s+(\d+)')
        # 查找所有匹配项
        matches = pattern.findall(menu.page_content)
        # 打印匹配结果
        self.title_info: List[Dict[str, Any]] = []
        title_index: int = -1
        for match in matches:
            title: str = match[0]
            content: str = match[1].strip('. ')
            page: int = int(match[2])
            # 如果满足`.+节`, 是1级标题
            if re.match(r'.+节', match[0]):
                title_index += 1
                self.title_info.append({"title": title,
                                        "content": content,
                                        "page": page,
                                        "children": []})
            elif re.match(r'\d+\.\d+', match[0]):
                # 如果满足`\d+\.\d+`, 是2级标题
                self.title_info[title_index]["children"].append({"title": title,
                                                                 "content": content,
                                                                 "page": page})

    def get_title_from_page(self, page: int) -> str:
        # 根据page返回title_info
        title_info: str = ""
        index_1level_title: int = 0
        # 根据page找对应的title_info中的page返回的title信息
        for i in range(len(self.title_info)):
            index_1level_title = i
            # if next title not exist
            if i >= len(self.title_info) - 1:
                break
            # if next title exist
            if self.title_info[i + 1]["page"] > page:
                break
        # add info into title info
        title_1level: Dict[str, Any] = self.title_info[index_1level_title]
        title_info += title_1level["title"] + " " + title_1level["content"] + "\n"
        # check 2 level title
        index_2level_title: int = -1
        for i in range(len(self.title_info[index_1level_title]["children"])):
            index_2level_title = i
            # if next title not exist
            if i >= len(self.title_info[index_1level_title]["children"]) - 1:
                break
            # if next title exist
            if self.title_info[index_1level_title]["children"][i + 1]["page"] > page:
                break
        # add info into title info
        if index_2level_title != -1:
            title_2level: Dict[str, Any] = self.title_info[index_1level_title]["children"][index_2level_title]
            title_info += title_2level["title"] + " " + title_2level["content"] + "\n"
        return "<title>\n" + title_info + "</title>"
