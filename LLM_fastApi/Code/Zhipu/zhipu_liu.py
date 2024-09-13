from langchain_core.prompts import ChatPromptTemplate


def get_llm_lc(self, app, r: ChatCompletionRequestStruct):

  """

  获取ai响应langchain版

  """

  return ChatOpenAI(

    temperature=0.5,

    openai_api_key=Config.ZHIPUAI_API_KEY,

    openai_api_base=Config.ZHIPUAI_OPENAI_API_URL,

    model=Config.ZHIPUAI_MODEL,

    streaming=r.streaming,

    callbacks=[StreamingStdOutCallbackHandler()],

  )

def get_prompt_lc(self):

  return ChatPromptTemplate.from_messages(

    [

    # ("system", "你是一个专业的AI助手。"),

    ("human", "{question}")

    ]

  )

llm = self.get_llm_lc(app, r)

prompt = self.get_prompt_lc()

llm_chain = prompt | llm

ret = llm_chain.stream({"question": r.question})

for _token in ret:

  token = _token.content

  finish_reason = ''

  if 'finish_reason' in _token.response_metadata:

  finish_reason = _token.response_metadata['finish_reason']

  reply = ChatCompletionResponseStruct()

  reply.text = token

  reply.finish_reason = finish_reason

  yield json.dumps(reply.to_dict(), ensure_ascii=False) + '\n'