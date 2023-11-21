import time
import chromadb
import openai
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers import ContextualCompressionRetriever, BM25Retriever
from langchain.retrievers.document_compressors import EmbeddingsFilter, DocumentCompressorPipeline
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from loguru import logger
from langchain.callbacks import FileCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import GoogleSerperRun, Tool
from langchain.vectorstores import Chroma
from transformers import GPT2Tokenizer
import logging
import gradio as gr

# 加载环境变量中的 OpenAI API 密钥
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY
# 打印 API 密钥
print(OPENAI_API_KEY)
logfile = "Rainbow_Agent_Search_V1.1_gradio_ui.log"
logger.add(logfile, colorize=True, enqueue=True)
handler = FileCallbackHandler(logfile)

# 创建 ChatOpenAI 实例作为底层语言模型
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

embeddings = OpenAIEmbeddings()

s# # 设置代理（替换为你的代理地址和端口）
proxy_url = 'http://localhost:7890'
os.environ['http_proxy'] = proxy_url
os.environ['https_proxy'] = proxy_url
Google_Search = GoogleSearchAPIWrapper()

persist_directory = ".chromadb/"
client = chromadb.PersistentClient(path=persist_directory)
# print(client.list_collections())
collections = client.list_collections()
list_collections_name = []
for collection in collections:
    collection_name = collection.name
    list_collections_name.append(collection_name)

# Local_Search Prompt模版
local_search_template = """
你作为一个AI问答助手。
必须通过以下双引号内的知识库内容进行问答:
“{combined_text}”

如果无法回答问题则回复:无法找到答案
我的问题是: {human_input}

"""

local_search_prompt = PromptTemplate(
    input_variables=["combined_text", "human_input"],
    template=local_search_template,
)
# 本地知识库工具
local_chain = LLMChain(
    llm=llm, prompt=local_search_prompt,
    verbose=True,
)

# 使用预训练的gpt2分词器
tokenizers = GPT2Tokenizer.from_pretrained("gpt2")

# 在文件顶部定义docsearch_db
docsearch_db = None


def echo(message, history, collection_name, load_existing_collection, new_collection_name, DirectoryLoader_path,
         print_speed_step):
    global docsearch_db

    if not load_existing_collection:
        # Collection exists, load it
        print("Collection exists, load it")
        docsearch_db = Chroma(client=client, embedding_function=embeddings, collection_name=collection_name)
    else:
        # 设置向量存储相关配置
        print("==========doc data vector search=======")
        print(DirectoryLoader_path)

        loader = DirectoryLoader(DirectoryLoader_path, show_progress=True, use_multithreading=True, silent_errors=True)

        documents = loader.load()
        print(documents)
        print("documents len= ", documents.__len__())

        input_chunk_size = 1536
        intput_chunk_overlap = 20
        embeddings.chunk_size = 1536
        embeddings.show_progress_bar = True
        embeddings.request_timeout = 20

        text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=int(input_chunk_size),
                                              chunk_overlap=int(intput_chunk_overlap))

        texts = text_splitter.split_documents(documents)
        print(texts)
        print("after split documents len= ", texts.__len__())
        # Collection does not exist, create it
        client.delete_collection(str(new_collection_name))
        docsearch_db = Chroma.from_documents(documents=texts, embedding=embeddings, collection_name=new_collection_name,
                                             persist_directory=persist_directory)

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

    # 初始化agent代理
    agent_open_functions = initialize_agent(
        tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=False,
        agent_kwargs=agent_kwargs,
        memory=memory,
        max_iterations=10,
        early_stopping_method="generate",
        handle_parsing_errors=True,  # 初始化代理并处理解析错误
        # handle_parsing_errors="Check your output and make sure it conforms!",
        callbacks=[handler],
        # return_intermediate_steps=True,
    )

    response = agent_open_functions.run(message)

    for i in range(0, len(response), int(print_speed_step)):
        # time.sleep(0.01)  # 可以根据需要调整休眠时间
        yield response[: i + int(print_speed_step)]
    # return response


def ask_local_vector_db(question):
    global docsearch_db

    # new docsearch_db 结合基础检索器+Embedding 压缩+BM25 关检词检索筛选
    chroma_retriever = docsearch_db.as_retriever(search_kwargs={"k": 50})
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter]
    )
    compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor,
                                                           base_retriever=chroma_retriever)
    compressed_docs = compression_retriever.get_relevant_documents(question, tools=tools)
    # 添加以下异常处理
    try:
        bm25_Retriever = BM25Retriever.from_documents(compressed_docs)
        bm25_Retriever.k = 30
        docs = bm25_Retriever.get_relevant_documents(question)
    except ValueError as e:
        print(f"Error while creating BM25Retriever: {e}")
        docs = []

    cleaned_matches = []
    total_toknes = 0
    # print(docs)
    for context in docs:
        cleaned_context = context.page_content.replace('\n', ' ').strip()

        cleaned_context = f"{cleaned_context}"
        tokens = tokenizers.encode(cleaned_context, add_special_tokens=False)
        if total_toknes + len(tokens) <= (1536 * 10):
            cleaned_matches.append(cleaned_context)
            total_toknes += len(tokens)
        else:
            break

    # 将清理过的匹配项组合合成一个字符串
    combined_text = " ".join(cleaned_matches)

    answer = local_chain.predict(combined_text=combined_text, human_input=question)
    return answer


# 创建工具列表
tools = [
    Tool(
        name="Google_Search",
        func=Google_Search.run,
        description="""
        当你用本地向量数据库问答后说无法找到答案的之后，你可以使用互联网搜索引擎工具进行信息查询,尝试直接找到问题答案。 
        将搜索到的按照问题的相关性和时间进行排序，并且你必须严格参照搜索到的资料和你自己的认识结合进行回答！
        如果搜索到一样的数据不要重复再搜索！
        注意你需要提出非常有针对性准确的问题和回答。
        """,
    ),
    Tool(
        name="Local_Search",
        func=ask_local_vector_db,
        description="""
        你可以首先通过本地向量数据知识库尝试寻找问答案。
        注意你需要提出非常有针对性准确的问题和回答。
        """
    )
]

with gr.Blocks() as demo:
    collection_name = gr.Dropdown(list_collections_name, label="Collection Name")

    # 将最上面的三个 UI 控件并排放置
    with gr.Row():
        # 在 Gradio UI 中添加一个复选框，用于控制加载已存在的集合
        load_existing_collection = gr.Checkbox(label="Load Existing Collection")
        new_collection_name = gr.Textbox("", label="New Collection Name")
        DirectoryLoader_path = gr.Textbox("", label="Directory Path")

    print_speed_step = gr.Slider(5, 20, render=False, label="Print Speed Step")
    gr.ChatInterface(
        echo, additional_inputs=[collection_name, load_existing_collection, new_collection_name, DirectoryLoader_path,
                                 print_speed_step],
        title="RainbowGPT-Agent",
        description="How to reach me: zhujiadongvip@163.com",
        css=".gradio-container {background-color: red}"
    )

demo.queue().launch(share=True)