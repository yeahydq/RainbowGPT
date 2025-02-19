import queue
import threading
import chromadb
import openai
import time
import os
from dotenv import load_dotenv
import gradio as gr
from langchain import hub
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.agents.format_scratchpad import format_log_to_str, format_to_openai_function_messages
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser, OpenAIFunctionsAgentOutputParser
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter, DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.utilities import WolframAlphaAPIWrapper, ArxivAPIWrapper
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import FileCallbackHandler
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool, render_text_description
from langchain_core.utils.function_calling import format_tool_to_openai_function
from langchain_text_splitters import CharacterTextSplitter
from loguru import logger
from langchain.chains import LLMMathChain
from langchain.callbacks.base import BaseCallbackHandler
from Rainbow_utils.model_config_manager import ModelConfigManager
import asyncio
from crawl4ai import AsyncWebCrawler

# Rainbow_utils
from Rainbow_utils.get_tokens_cal_filter import filter_chinese_english_punctuation, num_tokens_from_string, \
    truncate_string_to_max_tokens, concatenate_if_dissimilar
from Rainbow_utils import get_google_result
from Rainbow_utils import get_prompt_templates
from Rainbow_utils.image_genearation import ImageGen


class RainbowKnowledge_Agent:
    def __init__(self):
        self.load_dotenv()
        self.initialize_variables()
        self.create_interface()
        # 初始化模型配置管理器
        self.model_manager = ModelConfigManager()
        # 添加搜索历史记录器
        self.search_history = {
            "Google_Search": set(),
            "Local_Search": set(),
            "Calculator": set(),
            "Wolfram Alpha": set(),
            "Arxiv": set(),
            "Create_Image": set()
        }
        # 添加时间格式化工具
        self.time_formatter = lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def load_dotenv(self):
        load_dotenv()

    def initialize_variables(self):
        self.docsearch_db = None
        self.script_name = os.path.basename(__file__)
        self.logfile = "./logs/" + self.script_name + ".log"
        logger.add(self.logfile,
                   format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                   level="DEBUG",
                   rotation="1 MB",
                   compression="zip"
                   )
        self.handler = FileCallbackHandler(self.logfile)
        self.persist_directory = ".chromadb/"
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection_name_select_global = None
        # local private llm name
        self.local_private_llm_name_global = None
        # private llm apis
        self.local_private_llm_api_global = None
        # private llm api key
        self.local_private_llm_key_global = None
        # http proxy
        self.proxy_url_global = None
        # 创建 ChatOpenAI 实例作为底层语言模型
        self.llm = None
        self.llm_name_global = None
        self.embeddings = None
        self.Embedding_Model_select_global = 0
        self.temperature_num_global = 0
        # 文档切分的长度
        self.input_chunk_size_global = None
        # 本地知识库嵌入token max
        self.local_data_embedding_token_max_global = None
        # 在文件顶部定义docsearch_db
        self.docsearch_db = None
        self.human_input_global = None

        self.local_search_template = get_prompt_templates.local_search_template
        self.google_search_template = get_prompt_templates.google_search_template
        # 全局工具列表创建
        self.tools = []
        # memory
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            output_key="output"
        )

        self.Google_Search_tool = None
        self.Local_Search_tool = None
        self.llm_Agent_checkbox_group = None
        self.intermediate_steps_log = ""

        # 初始化LLM相关配置
        self.model_manager = ModelConfigManager()
        config = self.model_manager.get_active_config()
        
        # 使用全局配置初始化Calculator工具的LLM
        base_llm = ChatOpenAI(
            model_name=config.model_name,
            openai_api_base=config.api_base,
            openai_api_key=config.api_key,
            temperature=0
        )
        llm_math = LLMMathChain.from_llm(llm=base_llm)
        
        def get_rainbow_agent_help(input_str: str) -> str:
            """提供 RainbowKnowledge_Agent 的使用说明和帮助"""
            help_text = """
            🌈 RainbowKnowledge_Agent 使用指南

            1. 基本功能
               - 智能对话：支持自然语言交互
               - 多工具集成：Google搜索、本地知识库、Wolfram Alpha等
               - 知识库管理：支持多个知识库切换和查询
               - AI绘图：支持图片生成功能

            2. 主要工具说明
               - Google Search：实时网络搜索
               - Local Knowledge Search：本地知识库查询
               - Wolfram Alpha：数学和科学计算
               - Arxiv：学术论文搜索
               - Create Image：AI图片生成

            3. 使用技巧
               - 清晰表达问题需求
               - 合理选择工具组合
               - 灵活运用追问和澄清
               - 注意保持对话上下文

            4. 最佳实践
               - 复杂问题建议分步提问
               - 需要时可请求详细解释
               - 善用知识库功能
               - 适时切换Agent模式

            输入"help [工具名]"获取特定工具的详细说明
            例如：help Google Search、help Local Knowledge Search
            """
            
            # 处理特定工具的帮助请求
            if input_str.lower().startswith('help '):
                tool_name = input_str.lower().replace('help ', '').strip()
                tool_helps = {
                    'rainbowagent_help': """
                    🌟 RainbowAgent_Help 工具使用说明
                    
                    1. 基本用法
                       - 直接输入 "help" 获取总体使用指南
                       - 使用 "help [工具名]" 获取特定工具说明
                       - 例如: "help google search" 或 "help local knowledge search"
                    
                    2. 支持的工具说明查询
                       - Google Search
                       - Local Knowledge Search
                       - Wolfram Alpha
                       - Arxiv
                       - Create Image
                    
                    3. 帮助内容包括
                       - 工具功能介绍
                       - 适用场景说明
                       - 具体使用技巧
                       - 最佳实践建议
                    
                    4. 使用建议
                       - 优先查看总体指南了解系统功能
                       - 根据需求查询具体工具说明
                       - 遇到问题时查看相关工具的使用技巧
                    """,
                    'google search': """
                    🔍 Google Search 工具使用说明
                    - 功能：实时联网搜索最新信息
                    - 适用场景：查询实时资讯、获取网络信息
                    - 使用技巧：
                      1. 使用精确关键词
                      2. 可指定时间范围
                      3. 支持多语言搜索
                    """,
                    'local knowledge search': """
                    📚 Local Knowledge Search 工具使用说明
                    - 功能：搜索本地知识库内容
                    - 适用场景：专业领域查询、文档内容检索
                    - 使用技巧：
                      1. 选择合适的知识库
                      2. 使用准确的关键词
                      3. 可进行上下文关联搜索
                    """,
                    'wolfram alpha': """
                    🧮 Wolfram Alpha 工具使用说明
                    - 功能：数学计算与科学分析
                    - 适用场景：
                      1. 复杂数学计算
                      2. 科学数据查询
                      3. 工程计算问题
                    - 使用技巧：
                      1. 使用清晰的英文表达
                      2. 提供完整的计算条件
                      3. 适合处理定量分析
                    """,
                    'arxiv': """
                    📖 Arxiv 工具使用说明
                    - 功能：学术论文搜索和获取
                    - 适用场景：
                      1. 研究论文查询
                      2. 学术动态跟踪
                      3. 专业知识获取
                    - 使用技巧：
                      1. 使用准确的学术关键词
                      2. 可按领域筛选
                      3. 关注最新发表时间
                    """,
                    'create image': """
                    🎨 Create Image 工具使用说明
                    - 功能：AI图片生成
                    - 适用场景：
                      1. 创意图片生成
                      2. 视觉内容创作
                      3. 图片风格转换
                    - 使用技巧：
                      1. 提供详细的图片描述
                      2. 指定具体的风格要求
                      3. 可以参考样例图片
                    """
                }
                return tool_helps.get(tool_name, "未找到该工具的具体说明，请检查工具名称是否正确。\n可用的工具名称：rainbowagent_help, google search, local knowledge search, wolfram alpha, arxiv, create image")
            
            return help_text

        self.math_tool = Tool(
            name="RainbowAgent_Help",
            func=get_rainbow_agent_help,
            description="""RainbowKnowledge_Agent 帮助工具，可用于：
1. 获取系统整体使用说明
2. 查询特定工具的详细说明
3. 了解使用技巧和最佳实践
使用方式：
- 直接询问使用说明
- 输入"help [工具名]"获取特定工具说明"""
        )

        # 在初始化 math_tool 之后添加 wolfram_tool 的初始化
        try:
            wolfram = WolframAlphaAPIWrapper(
                wolfram_alpha_appid=os.getenv('WOLFRAM_ALPHA_APPID')
            )
            self.wolfram_tool = Tool(
                name="Wolfram Alpha",
                func=wolfram.run,
                description="""用于解决数学、科学和工程计算的强大工具。适用于:
1. 复杂数学计算和方程求解
2. 科学数据查询和分析
3. 单位换算和物理计算
4. 统计和数据分析
使用时请用清晰的英文描述问题。
示例: 'solve x^2 + 2x + 1 = 0' 或 'distance from Earth to Mars'"""
            )
        except Exception as e:
            logger.error(f"Failed to initialize Wolfram Alpha tool: {str(e)}")
            self.wolfram_tool = None

        # 添加爬虫模式控制
        self.use_async_crawler = False  # 默认使用同步爬虫
        self.crawler_mode = "同步爬虫"  # 用于UI显示

    def get_llm(self):
        """获取当前配置的LLM实例"""
        config = self.model_manager.get_active_config()
        return ChatOpenAI(
            model_name=config.model_name,
            openai_api_base=config.api_base,
            openai_api_key=config.api_key,
            temperature=config.temperature,
            streaming=True
        )

    def check_search_history(self, tool_name, query):
        """检查是否存在重复搜索"""
        if query in self.search_history[tool_name]:
            return True
        self.search_history[tool_name].add(query)
        return False

    def ask_local_vector_db(self, question):
        # 检查是否重复搜索
        if self.check_search_history("Local_Search", question):
            return "⚠️ 检测到重复搜索。请尝试使用不同的关键词或其他工具来获取新信息。"
        
        # 添加当前时间强调
        current_time = self.time_formatter()
        time_reminder = f"\n⏰ 当前查询时间: {current_time}\n请注意知识库内容的时效性。\n"
        
        # 使用模型配置管理器获取LLM
        self.llm = self.get_llm()
        
        local_search_prompt = PromptTemplate(
            input_variables=["combined_text", "human_input", "human_input_first", "current_time"],
            template=self.local_search_template + "\n当前时间: {current_time}\n请特别关注知识库内容的时效性。",
        )
        
        local_chain = LLMChain(
            llm=self.llm,
            prompt=local_search_prompt,
            verbose=True,
        )

        docs = []
        if self.Embedding_Model_select_global == 0:
            print("OpenAIEmbeddings Search")
            # 结合基础检索器+Embedding上下文压缩
            # 将稀疏检索器（如 BM25）与密集检索器（如嵌入相似性）相结合
            chroma_retriever = self.docsearch_db.as_retriever(search_kwargs={"k": 30})

            # 将缩器和文档转换器串在一起
            splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
            redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings)
            relevant_filter = EmbeddingsFilter(embeddings=self.embeddings, similarity_threshold=0.76)
            pipeline_compressor = DocumentCompressorPipeline(
                transformers=[splitter, redundant_filter, relevant_filter]
            )
            compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor,
                                                                   base_retriever=chroma_retriever)
            # compressed_docs = compression_retriever.get_relevant_documents(question, tools=tools)

            the_collection = self.client.get_collection(name=self.collection_name_select_global)
            the_metadata = the_collection.get()
            the_doc_llist = the_metadata['documents']
            bm25_retriever = BM25Retriever.from_texts(the_doc_llist)
            bm25_retriever.k = 30

            # 置次数
            max_retries = 3
            retries = 0
            while retries < max_retries:
                try:
                    # 初始化 ensemble 检索器
                    ensemble_retriever = EnsembleRetriever(
                        retrievers=[bm25_retriever, compression_retriever], weights=[0.5, 0.5]
                    )
                    docs = ensemble_retriever.get_relevant_documents(question)
                    break  # 如果成功执行，跳出循环
                except openai.error.OpenAIError as openai_error:
                    if "Rate limit reached" in str(openai_error):
                        print(f"Rate limit reached: {openai_error}")
                        # 如果是速率限制错误，等待一段时间后重试
                        time.sleep(20)
                        retries += 1
                    else:
                        print(f"OpenAI API error: {openai_error}")
                        docs = []
                        break  # 如果遇到其他错误，跳出循环
            # 处理循环结束后的情况
            if retries == max_retries:
                print(f"Max retries reached. Code execution failed.")
        elif self.Embedding_Model_select_global == 1:
            print("HuggingFaceEmbedding Search")
            chroma_retriever = self.docsearch_db.as_retriever(search_kwargs={"k": 30})
            # docs = chroma_retriever.get_relevant_documents(question)
            # chroma_vectorstore = Chroma.from_texts(the_doc_llist, embeddings)
            # chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": 10})

            the_collection = self.client.get_collection(name=self.collection_name_select_global)
            the_metadata = the_collection.get()
            the_doc_llist = the_metadata['documents']
            bm25_retriever = BM25Retriever.from_texts(the_doc_llist)
            bm25_retriever.k = 30

            # 初始化 ensemble 检索器
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
            )
            docs = ensemble_retriever.get_relevant_documents(question)

        cleaned_matches = []
        total_toknes = 0
        last_index = 0
        for index, context in enumerate(docs):
            cleaned_context = context.page_content.replace('\n', ' ').strip()
            cleaned_context = f"{cleaned_context}"
            tokens = num_tokens_from_string(cleaned_context, "cl100k_base")
            # tokens = tokenizers.encode(cleaned_context, add_special_tokens=False)
            if total_toknes + tokens <= (int(self.local_data_embedding_token_max_global)):
                cleaned_matches.append(cleaned_context)
                total_toknes += tokens
            else:
                last_index = index
                break
        print("Embedding了 ", str(last_index + 1), " 个知识库文档块")
        # 将清理过的匹配项组合合成一个字符串
        combined_text = " ".join(cleaned_matches)

        answer = local_chain.predict(
            combined_text=combined_text, 
            human_input=question,
            human_input_first=self.human_input_global,
            current_time=current_time
        )
        
        return time_reminder + answer

    def createImageByBing(self, input):
        auth_cooker = os.getenv('BINGCOKKIE')
        sync_gen = ImageGen(auth_cookie=auth_cooker)
        image_list = sync_gen.get_images(input)
        response = []
        if image_list is None:
            return "我无法为您生成对应的图片，请重试或者补充您的描述"
        else:
            for url in image_list:
                if not url.endswith(".svg"):
                    response.append(url)
            return response

    def get_google_answer(self, question, result_queue):
        try:
            logger.debug("Starting get_google_answer")
            google_answer_box = get_google_result.selenium_google_answer_box(
                question, "Rainbow_utils/chromedriver.exe")
            google_answer_box = filter_chinese_english_punctuation(google_answer_box)
            result_queue.put(("google_answer_box", google_answer_box))
            logger.debug("Completed get_google_answer successfully")
        except Exception as e:
            logger.exception(f"Error in get_google_answer: {str(e)}")
            result_queue.put(("google_answer_box", f"获取Google答案时出错: {str(e)}"))

    def process_data_title_summary(self, data_title_Summary, result_queue):
        data_title_Summary_str = ''.join(data_title_Summary)
        result_queue.put(("data_title_Summary_str", data_title_Summary_str))

    async def get_website_content_async(self, url):
        """Async function to get website content using AsyncWebCrawler"""
        try:
            async with AsyncWebCrawler(verbose=True, timeout=30) as crawler:  # 添加超时设置
                result = await crawler.arun(url=url)
                if result and result.markdown_v2 and result.markdown_v2.raw_markdown:
                    return result.markdown_v2.raw_markdown
                return None
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {str(e)}")
            return None

    def process_custom_search_link(self, custom_search_link, result_queue):
        """处理搜索链接并获取网页内容"""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        async def process_urls():
            """异步处理所有URL"""
            tasks = []
            for link in custom_search_link[:9]:  # 限制处理前9个链接
                if self.use_async_crawler:
                    tasks.append(self.get_website_content_async(link))
                else:
                    # 使用线程池处理同步爬取
                    with ThreadPoolExecutor() as executor:
                        future = executor.submit(get_google_result.get_website_content, link)
                        tasks.append(future)
            
            try:
                # 等待所有任务完成
                if self.use_async_crawler:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    # print("异步爬虫结果：", results)
                    # 保存到本地文件 utf-8  
                    with open('async_crawler_results.txt', 'w', encoding='utf-8') as file:
                        for result in results:
                            file.write(str(result) + '\n')
                else:
                    # 等待所有同步任务完成
                    results = [task.result() if hasattr(task, 'result') else task for task in tasks]
                    # print("同步爬虫结果：", results)
                    # 保存到本地文件 utf-8  
                    with open('sync_crawler_results.txt', 'w', encoding='utf-8') as file:
                        for result in results:
                            file.write(str(result) + '\n')  
                
                # 直接将结果放入队列
                result_queue.put(("link_detail_string", str(results)))
                
            except Exception as e:
                logger.error(f"Error in process_urls: {str(e)}")
                result_queue.put(("link_detail_string", f"爬取内容时发生错误: {str(e)}"))

        try:
            # 运行异步任务
            asyncio.run(process_urls())
            
        except Exception as e:
            error_msg = f"处理URL时发生错误: {str(e)}"
            logger.error(error_msg)
            result_queue.put(("link_detail_string", error_msg))

    def custom_search_and_fetch_content(self, question, result_queue):
        try:
            logger.debug("Starting custom_search_and_fetch_content")
            custom_search_link, data_title_Summary = get_google_result.google_custom_search(question)

            thread3 = threading.Thread(target=self.process_data_title_summary,
                                       args=(data_title_Summary, result_queue))
            thread4 = threading.Thread(target=self.process_custom_search_link,
                                       args=(custom_search_link, result_queue))

            thread3.start()
            thread4.start()

            # Add timeout to thread joins
            thread3.join(timeout=30)
            thread4.join(timeout=30)

            if thread3.is_alive() or thread4.is_alive():
                logger.error("Content processing threads timed out")
                raise TimeoutError("Content processing timed out")

            logger.debug("Completed custom_search_and_fetch_content successfully")
        except Exception as e:
            logger.exception(f"Error in custom_search_and_fetch_content: {str(e)}")
            result_queue.put(("data_title_Summary_str", f"获取搜索内容时出错: {str(e)}"))
            result_queue.put(("link_detail_string", ""))

    def Google_Search_run(self, question):
        try:
            # 检查是否重复搜索
            if self.check_search_history("Google_Search", question):
                return "⚠️ 检测到重复搜索。请尝试使用不同的关键词或其他工具来获取新信息。"
            
            logger.debug(f"Starting Google search for question: {question}")
            
            # 添加当前时间强调
            current_time = self.time_formatter()
            time_reminder = f"\n⏰ 当前搜索时间: {current_time}\n请注意时效性，确保获取最新信息。\n"
            
            # 使用模型配置管理器获取LLM
            self.llm = self.get_llm()
            
            # 在模板中添加时间信息
            local_search_prompt = PromptTemplate(
                input_variables=["combined_text", "human_input", "human_input_first", "current_time"],
                template=self.google_search_template + "\n当前时间: {current_time}\n请特别注意信息的时效性。",
            )
            
            local_chain = LLMChain(
                llm=self.llm,
                prompt=local_search_prompt,
                verbose=True,
            )

            # 创建一个队列来存储线程结果
            results_queue = queue.Queue()
            # 创建并启线程
            thread1 = threading.Thread(target=self.get_google_answer, args=(question, results_queue))
            thread2 = threading.Thread(target=self.custom_search_and_fetch_content, args=(question, results_queue))

            logger.debug("Starting search threads")
            thread1.start()
            thread2.start()

            # Add timeout to thread joins
            thread1.join(timeout=30)
            thread2.join(timeout=30)

            if thread1.is_alive() or thread2.is_alive():
                logger.error("Search threads timed out")
                raise TimeoutError("Search operation timed out")

            # 初始化变量
            google_answer_box = ""
            data_title_Summary_str = ""
            link_detail_string = ""

            # 提取并分配结果
            while not results_queue.empty():
                result_type, result = results_queue.get()
                if result_type == "google_answer_box":
                    google_answer_box = result
                elif result_type == "data_title_Summary_str":
                    data_title_Summary_str = result
                elif result_type == "link_detail_string":
                    link_detail_string = result

            finally_combined_text = f"""
            当前关键字搜索的答案框据：
            {google_answer_box}

            搜索结果相似度TOP10的网站标题和摘要数据：
            {data_title_Summary_str}

            搜索结果相似度TOP1-9的网站的详细内容数据:
            {link_detail_string}

            """

            truncated_text = truncate_string_to_max_tokens(finally_combined_text,
                                                           self.local_data_embedding_token_max_global,
                                                           "cl100k_base",
                                                           step_size=256)

            answer = local_chain.predict(
                combined_text=truncated_text, 
                human_input=question,
                human_input_first=self.human_input_global,
                current_time=current_time
            )

            return time_reminder + answer

        except Exception as e:
            logger.exception(f"Error in Google_Search_run: {str(e)}")
            return f"搜索过程中发生错误: {str(e)}。请检查网络连接或重试。"

    def echo(self, message, history, collection_name_select, print_speed_step,
             tool_checkbox_group, Embedding_Model_select, local_data_embedding_token_max,
             llm_Agent_checkbox_group, crawler_mode):
        """
        处理用户输入的主要方法
        """
        # 更新爬虫模式
        self.use_async_crawler = (crawler_mode == "异步爬虫")
        
        # 重置搜索历史
        self.search_history = {
            "Google_Search": set(),
            "Local_Search": set(),
            "Calculator": set(),
            "Wolfram Alpha": set(),
            "Arxiv": set(),
            "Create_Image": set()
        }
        
        self.human_input_global = message
        self.collection_name_select_global = str(collection_name_select)
        self.local_data_embedding_token_max_global = int(local_data_embedding_token_max)
        self.llm_Agent_checkbox_group = llm_Agent_checkbox_group  # 保留Agent类型设置

        # 获取当前配置
        config = self.model_manager.get_active_config()
        
        response = (f"{config.model_name} 模型加载中..... temperature={config.temperature}")
        for i in range(0, len(response), int(print_speed_step)):
            yield response[: i + int(print_speed_step)]

        # 初始化LLM
        self.llm = self.get_llm()

        self.tools = []  # 重置工具列表
        # Check if 'wolfram-alpha' is in the selected tools
        if "wolfram-alpha" in tool_checkbox_group:
            wolfram = WolframAlphaAPIWrapper()
            wolfram_tool = Tool(
                name="Wolfram Alpha",
                func=wolfram.run,
                description="Useful for when you need to answer questions about Math, Science, Technology, Culture, Society and Everyday Life"
            )
            self.tools.append(wolfram_tool)

        if "arxiv" in tool_checkbox_group:
            arxiv = ArxivAPIWrapper()
            arxiv_tool = Tool(
                name="Arxiv",
                func=arxiv.run,
                description="Useful for when you need to get information about scientific papers from arxiv.org. Input should be a search query."
            )
            self.tools.append(arxiv_tool)

        google_search_description = (
            """
                这是一个问题需要网络搜索的Google搜索工具。
                1.你先根据我的问题提取出最适合Google搜索引擎搜索的关键字进行搜索,可以选择英语或者中文搜索
                2.同时增加一些搜索提示词包括(引号，时间，关键字)
                3.如果问题比较复杂，你可以一步一步的思考去搜索和回答
                4.确保每个回答都不仅基于数据，输出的回答必须包含深入、完整，充分反映你对问题的全面理解。
                5.请注意时效性，如果问题中需要实效性，确保依据下面的当前时间加入到搜索关键字中来获取最新信息。
            """
            + f"\n当前时间: {self.time_formatter()}"
        )
        
        self.Google_Search_tool = Tool(
            name="Google_Search",
            func=self.Google_Search_run,
            description=google_search_description
        )

        self.Local_Search_tool = Tool(
            name="Local_Search",
            func=self.ask_local_vector_db,
            description="""
                这是一个本地知识库搜索工具，你可以优使用本地搜索并总结回答。
                1.你先根我的问题提取出最适合embedding模型向量匹配的关键字进行搜索。
                2.注意你需要提出非常有针对性准确的问题和回答。
                3.如果问题比较复杂，可以将复杂的问题进行行拆分，你可以一步一步的思考。
                4.确保每个回答都不仅基于数据，输出的回答必须包含深入、完整，充分反映你对问题的全面理解。
            """
        )

        self.Create_Image_tool = Tool(
            name="Create_Image",
            func=self.createImageByBing,
            description="""
                这是一个图片生成工具，当我的问题中明确需要画图，你就可以使用该工具并生成图片
                1。当你回关于需要使用bing来生成什么画图、照片时很有用，先提取生成图片的提示词，然后调用该工具。
                2.并严格按照Markdown语法: [![图片描述](图片链接)](图片链接)。
                3.如果生成的图片链接数量大于1，将其全部严格按照Markdown语法: [![图片描述](图片链接)](图片链接)。
                4.如果问题比较复杂，可以将复杂的问题进行拆分，你可以一步一步的思考。
                """
        )

        self.tools = [self.math_tool]  # 确保始终包含 Calculator 工具
        # Initialize flags for additional tools
        flag_get_Local_Search_tool = False
        # Check for additional tools and append them if not already in the list
        for tg in tool_checkbox_group:
            if tg == "wolfram-alpha" and self.wolfram_tool is not None:
                response = "Wolfram Alpha 工具加入 回答中..........."
                for i in range(0, len(response), int(print_speed_step)):
                    yield response[:i + int(print_speed_step)]
                self.tools.append(self.wolfram_tool)
                
            elif tg == "Google Search" and self.Google_Search_tool not in self.tools:
                response = "Google Search 工具加入 回答中..........."
                for i in range(0, len(response), int(print_speed_step)):
                    yield response[:i + int(print_speed_step)]
                self.tools.append(self.Google_Search_tool)

            elif tg == "Local Knowledge Search" and self.Local_Search_tool not in self.tools:
                response = "Local Knowledge Search 工具加入 回答中..........."
                for i in range(0, len(response), int(print_speed_step)):
                    yield response[:i + int(print_speed_step)]
                self.tools.append(self.Local_Search_tool)

                response = (f"{config.model_name} & {Embedding_Model_select} 模型加载中.....temperature="
                            + str(self.temperature_num_global))
                for i in range(0, len(response), int(print_speed_step)):
                    yield response[:i + int(print_speed_step)]

                if Embedding_Model_select in ["Openai Embedding", "", None]:
                    self.embeddings = OpenAIEmbeddings()
                    self.embeddings.show_progress_bar = True
                    self.embeddings.request_timeout = 20
                    self.Embedding_Model_select_global = 0
                elif Embedding_Model_select == "HuggingFace Embedding":
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-mpnet-base-v2",
                        cache_folder="models"
                    )
                    self.Embedding_Model_select_global = 1

                flag_get_Local_Search_tool = True

            elif tg == "Create Image" and self.Create_Image_tool not in self.tools:
                response = "Create Image 工具加入 回答中..........."
                for i in range(0, len(response), int(print_speed_step)):
                    yield response[:i + int(print_speed_step)]
                self.tools.append(self.Create_Image_tool)

        if message == "":
            response = "哎呀！好像有点小尴尬，您似乎忘记提出问题了。别着急，随时输入您的问题，我将尽力为您提供帮助！"
            for i in range(0, len(response), int(print_speed_step)):
                yield response[: i + int(print_speed_step)]
            return

        if flag_get_Local_Search_tool:
            if collection_name_select and collection_name_select != "...":
                print(f"{collection_name_select}", " Collection exists, load it")
                response = f"{collection_name_select}" + "知识库加载中，请等待我的回答......."
                for i in range(0, len(response), int(print_speed_step)):
                    yield response[: i + int(print_speed_step)]
                self.docsearch_db = Chroma(client=self.client, embedding_function=self.embeddings,
                                           collection_name=collection_name_select)
            else:
                response = "未选择知识库，回答中止。"
                for i in range(0, len(response), int(print_speed_step)):
                    yield response[: i + int(print_speed_step)]
                return

        if llm_Agent_checkbox_group == "openai-functions":
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

            llm_with_tools = self.llm.bind(
                functions=[format_tool_to_openai_function(t) for t in self.tools]
            )

            agent = (
                    {
                        "input": lambda x: x["input"],
                        "agent_scratchpad": lambda x: format_to_openai_function_messages(
                            x["intermediate_steps"]
                        ),
                    }
                    | prompt
                    | llm_with_tools
                    | OpenAIFunctionsAgentOutputParser()
            )

            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                return_intermediate_steps=True,
                handle_parsing_errors=True,
            )

            try:
                response = ""
                for chunk in agent_executor.stream({"input": message}):
                    if "output" in chunk:
                        response += chunk["output"]
                        yield response
                    elif "intermediate_step" in chunk:
                        self.intermediate_steps_log = str(chunk["intermediate_step"])

            except Exception as e:
                yield f"发生错误：{str(e)}"
                logger.error(f"Error in agent execution: {str(e)}")
        elif llm_Agent_checkbox_group == "ZeroShotAgent-memory":
            # 修改 prefix 和 suffix 以更好地处理对话
            prefix = """你是一个高效的AI助手。请遵循以下原则:

1. 避免重复查询
   - 不要重复使用相同的搜索关键词
   - 不要重复查询已获得的信息
   - 如需补充信息，使用不同角度的关键词

2. 工具使用规则
   - 每个工具最多使用1次
   - 仅在必要时使用工具
   - 已有足够信息时直接回答
   - 最多使用2次工具查询

3. 回答要求
   - 信息完整准确
   - 逻辑清晰连贯
   - 直接给出答案
   - 避免冗余内容

请按以下格式回复:

Thought: 简要分析当前情况

Action: 工具名称

Action Input: 查询内容

Observation: 工具返回结果

Final Answer: 完整答案

当前可用工具:"""

            suffix = """注意事项:
1. 禁止重复查询
2. 最多使用2次工具
3. 获得答案立即停止

历史对话:
{chat_history}

当前问题: {input}

{agent_scratchpad}"""

            prompt = ZeroShotAgent.create_prompt(
                self.tools,
                prefix=prefix,
                suffix=suffix,
                input_variables=["input", "chat_history", "agent_scratchpad"],
            )

            llm_chain = LLMChain(llm=self.llm, prompt=prompt)
            agent = ZeroShotAgent(llm_chain=llm_chain, tools=self.tools, verbose=True)

            # 创建一个回调处理器来捕获中间步骤
            class VerboseHandler(BaseCallbackHandler):
                def __init__(self):
                    self.steps = []
                    self.current_iteration = 0
                    self.current_output = ""
                    super().__init__()
                
                def on_agent_action(self, action, color=None, **kwargs):
                    try:
                        # 增加轮次计数和思考过程记录
                        self.current_iteration += 1
                        # 使用 Markdown 格式来突出显示轮次信息
                        step_text = f"\n### 🤔 第 {self.current_iteration} 轮思考过程\n\n"
                        
                        # 记录思考过程，使用更醒目的格式
                        if hasattr(action, 'log') and action.log:
                            step_text += f"🔍 **思考:** {action.log}\n\n"
                        
                        # 记录工具使用
                        if hasattr(action, 'tool'):
                            step_text += f"🛠️ **使用工具:** {action.tool}\n\n"
                        
                        # 记录工具输入
                        if hasattr(action, 'tool_input'):
                            step_text += f"📝 **输入参数:** {action.tool_input}\n\n"
                        
                        self.steps.append(step_text)
                        # 实时更新当前输出
                        self.current_output = "".join(self.steps)
                        
                    except Exception as e:
                        error_text = f"⚠️ **错误:** 行动记录出现问题: {str(e)}\n\n"
                        self.steps.append(error_text)
                        self.current_output = "".join(self.steps)
                
                def on_agent_observation(self, observation, color=None, **kwargs):
                    try:
                        if observation:
                            # 使用更醒目的格式显示观察结果
                            observation_text = f"📊 **观察结果:**\n```\n{observation}\n```\n\n"
                            self.steps.append(observation_text)
                            self.current_output = "".join(self.steps)
                            
                    except Exception as e:
                        error_text = f"⚠️ **错误:** 观察记录出现问题: {str(e)}\n\n"
                        self.steps.append(error_text)
                        self.current_output = "".join(self.steps)
                
                def on_agent_finish(self, finish, color=None, **kwargs):
                    try:
                        # 检查是否是最终答案
                        if hasattr(finish, 'return_values'):
                            if isinstance(finish.return_values, dict) and "output" in finish.return_values:
                                final_answer = finish.return_values['output']
                            else:
                                final_answer = str(finish.return_values)
                            
                            # 检查是否包含"思考:"前缀
                            if "思考:" in final_answer:
                                # 提取实际答案（去除思考部分）
                                answer_parts = final_answer.split("思考:")
                                if len(answer_parts) > 1:
                                    # 取最后一个分割部分作为实际答案
                                    actual_answer = answer_parts[0].strip()
                                else:
                                    actual_answer = final_answer
                            else:
                                actual_answer = final_answer
                            
                            # 格式化最终输出
                            formatted_output = f"### ✨ 最终答案:\n\n{actual_answer}\n\n"
                            self.steps.append(formatted_output)
                            
                        self.current_output = "".join(self.steps)
                        
                    except Exception as e:
                        error_text = f"⚠️ **错误:** 完成记录出现问题: {str(e)}\n\n"
                        self.steps.append(error_text)
                        self.current_output = "".join(self.steps)

                def get_current_output(self):
                    """返回当前的输出内容"""
                    return self.current_output

            handler = VerboseHandler()
            
            # 修改 agent_chain 配置
            agent_chain = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=self.tools,
                verbose=True,
                memory=self.memory,
                max_iterations=3,
                handle_parsing_errors=True,
                early_stopping_method="generate",
                callbacks=[handler],
                return_intermediate_steps=True
            )

            try:
                # 获取聊天历史
                chat_history = self.memory.load_memory_variables({})["chat_history"]
                
                # 创建处理器实例
                handler = VerboseHandler()
                
                # 运行agent_chain
                result = agent_chain(
                    {"input": message, "chat_history": chat_history},
                    include_run_info=True
                )
                
                # 使用处理器获取完整输出
                full_output = handler.get_current_output()
                
                if not full_output:  # 如果没有捕获到步骤，使用备用方案
                    steps = []
                    if "intermediate_steps" in result:
                        for step in result["intermediate_steps"]:
                            if len(step) >= 2:
                                action, observation = step
                                steps.append(f"**思考:** {action.log if hasattr(action, 'log') else ''}")
                                steps.append(f"**行动:** {action.tool if hasattr(action, 'tool') else ''}")
                                steps.append(f"**输入:** {action.tool_input if hasattr(action, 'tool_input') else ''}")
                                steps.append(f"**观察结果:**\n{observation}")
                    
                    if "output" in result:
                        steps.append(f"**最终答案:**\n{result['output']}")
                    
                    full_output = "\n".join(steps)
                
                yield full_output

            except Exception as e:
                error_msg = f"Agent执行过程中发生错误：{str(e)}"
                logger.error(error_msg)
                yield error_msg
        elif llm_Agent_checkbox_group == "Simple Chat":
            try:
                # 使用模型配置管理器获取LLM
                self.llm = self.get_llm()
                
                # 创建一个简单的提示模板，确保正确处理上下文
                simple_prompt = ChatPromptTemplate.from_messages([
                    ("system", """你是一个友好的AI助手。请：
1. 直接回答用户问题
2. 保持回答简洁明了
3. 如果不确定，诚实承认
4. 使用礼貌友好的语气
5. 只回答当前问题，不要重复之前的回答"""),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("user", "{input}")  # 将用户输入移到最后，确保是最新的问题
                ])
                
                # 创建简单的对话链
                chat_chain = simple_prompt | self.llm
                
                # 获取聊天历史并格式化
                chat_history = self.memory.load_memory_variables({})["chat_history"]
                
                # 执行对话
                response = ""
                for chunk in chat_chain.stream({
                    "input": message,
                    "chat_history": chat_history[-4:] if chat_history else []  # 只保留最近的2轮对话
                }):
                    response += chunk.content
                    yield response
                
                # 更新记忆前清除旧的上下文
                if len(chat_history) > 4:  # 如果历史记录超过2轮对话
                    self.memory.clear()  # 清除所有历史
                    # 只保存最近的一轮对话
                    self.memory.save_context({"input": message}, {"output": response})
                else:
                    # 正常保存上下文
                    self.memory.save_context({"input": message}, {"output": response})
                
            except Exception as e:
                error_msg = f"Simple Chat模式执行过程中发生错误：{str(e)}"
                logger.error(error_msg)
                yield error_msg

    def update_collection_name(self):
        # 获取已存在的collection的名称列表
        collections = self.client.list_collections()
        collection_name_choices = [collection.name for collection in collections]
        # 返回新的下拉列表组件
        return gr.Dropdown(
            choices=collection_name_choices,
            value=collection_name_choices[0] if collection_name_choices else None
        )

    def create_interface(self):
        with gr.Blocks(theme=gr.themes.Soft()) as self.interface:
            # 定义自定义CSS样式
            custom_css = """
                <style>
                    /* 标题样式 */
                    .gradio-header h1 {
                        text-align: center;
                        margin-bottom: 1rem;
                        font-family: "Courier New", monospace;
                        background: linear-gradient(135deg, #9400D3, #4B0082, #0000FF, #008000, #FFFF00, #FF7F00, #FF0000);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        background-clip: text;
                        color: transparent;
                    }

                    /* 聊天界面容器样式 */
                    .gradio-container {
                        min-height: 95vh !important;
                    }

                    /* 聊天记录区域样式 */
                    .chat-history {
                        height: calc(95vh - 200px) !important;
                        overflow-y: auto;
                    }

                    /* 帮助面板样式 */
                    .help-panel {
                        padding: 15px;
                        background: #f8f9fa;
                        border-radius: 8px;
                        height: calc(95vh - 40px);
                        overflow-y: auto;
                    }

                    /* 链接样式 */
                    a {
                        color: #007bff;
                        text-decoration: none;
                    }

                    a:hover {
                        text-decoration: underline;
                    }

                    /* 分割线样式 */
                    hr {
                        border: 0;
                        height: 1px;
                        background: #dee2e6;
                        margin: 1rem 0;
                    }

                    /* 工具选择组样式 */
                    .tool-group {
                        margin-bottom: 1rem;
                        padding: 10px;
                        border-radius: 5px;
                        background: #ffffff;
                    }
                </style>
            """

            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    # 左侧列: 所控件
                    with gr.Row():
                        with gr.Group():
                            gr.Markdown("### Agent Settings")
                            llm_Agent = ["Simple Chat", "openai-functions", "ZeroShotAgent-memory"]
                            llm_Agent_checkbox_group = gr.Dropdown(
                                llm_Agent, 
                                label="LLM Agent Type Options",
                                value=llm_Agent[0]
                            )

                    with gr.Group():
                        gr.Markdown("### Knowledge Collection Settings")
                        collection_name_select = gr.Dropdown(
                            choices=[],
                            label="Select existed Collection",
                            value=None
                        )
                        Refresh_button = gr.Button("Refresh Collection", variant="secondary")
                        Refresh_button.click(fn=self.update_collection_name, outputs=collection_name_select)

                        print_speed_step = gr.Slider(5, 10, label="Print Speed Step", step=1)

                    with gr.Row():
                        with gr.Group():
                            gr.Markdown("### Additional Tools")
                            tool_options = ["Google Search", "Local Knowledge Search", "wolfram-alpha", "arxiv",
                                          "Create Image"]
                            tool_checkbox_group = gr.CheckboxGroup(tool_options, label="Tools Select")
                            gr.Markdown("Note: Select the tools you want to use.")

                    with gr.Group():
                        gr.Markdown("### Embedding Data Settings")
                        Embedding_Model_select = gr.Radio(["Openai Embedding", "HuggingFace Embedding"],
                                                          label="Embedding Model Select",
                                                          value="HuggingFace Embedding")
                        local_data_embedding_token_max = gr.Slider(2048, 130048, step=2,
                                                                   label="Embeddings Data Max Tokens",
                                                                   value=2048)

                    with gr.Group():
                        gr.Markdown("### Crawler Settings")
                        crawler_mode = gr.Radio(
                            choices=["同步爬虫", "异步爬虫"],
                            label="爬虫模式选择",
                            value="同步爬虫",
                            info="同步爬虫(selenium)，异步爬虫(crawl4ai)更快速"
                        )
                        
                        def update_crawler_mode(mode):
                            self.use_async_crawler = (mode == "异步爬虫")
                            self.crawler_mode = mode
                            return f"已切换到{mode}模式"
                        
                        crawler_mode.change(
                            fn=update_crawler_mode,
                            inputs=[crawler_mode],
                            outputs=[gr.Textbox(label="状态", visible=False)]
                        )

                with gr.Column(scale=5):
                    # 中间聊天界面
                    chatbot = gr.ChatInterface(
                        self.echo,
                        additional_inputs=[
                            collection_name_select,
                            print_speed_step,
                            tool_checkbox_group,
                            Embedding_Model_select,
                            local_data_embedding_token_max,
                            llm_Agent_checkbox_group,
                            crawler_mode
                        ],
                        title="RainbowGPT-Agent",
                        css=custom_css,
                        theme="soft",
                        fill_height=True,
                        autoscroll=True,
                        type='messages'
                    )

            with gr.Column(scale=2):
                # 右侧帮助面板
                with gr.Group(elem_classes="help-panel"):
                    gr.Markdown("""
                        ### 🌈 RainbowGPT-Agent 使用指南
                                
                        #### 🎯 使用技巧
                        输入：help 即可调用帮助工具
                        
                        #### 📞 需要帮助？
                        - 遇到问题请联系：[zhujiadongvip@163.com](mailto:zhujiadongvip@163.com)
                        - 建议优先查看使用技巧解决问题
                        - 欢迎反馈使用体验
                    """)

    def launch(self):
        return self.interface
