#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Date: 2023/10/15 18:00
Desc: 同花顺-板块-概念板块
http://q.10jqka.com.cn/gn/detail/code/301558/
"""
from datetime import datetime
import time
from functools import lru_cache
from io import StringIO
import random

import pandas as pd
import requests
from bs4 import BeautifulSoup
from py_mini_racer import py_mini_racer
from tqdm import tqdm

from akshare.datasets import get_ths_js
from akshare.utils import demjson
from selenium import webdriver
import platform
from selenium.webdriver.chrome.service import Service
import numpy as np
import pickle
import gradio as gr

def stock_board_concept_graph_ths(symbol: str = "通用航空") -> pd.DataFrame:
    """
    同花顺-板块-概念板块-概念图谱
    http://q.10jqka.com.cn/gn/detail/code/301558/
    :param symbol: 板块名称
    :type symbol: str
    :return: 概念图谱
    :rtype: pandas.DataFrame
    """
    stock_board_ths_map_df = stock_board_concept_name_ths()
    symbol = (
        stock_board_ths_map_df[stock_board_ths_map_df["概念名称"] == symbol]["网址"]
        .values[0]
        .split("/")[-2]
    )
    url = f"http://q.10jqka.com.cn/gn/detail/code/{symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36",
    }
    r = requests.get(url, headers=headers)
    temp_df = pd.read_html(StringIO(r.text))[0]
    new_list = []
    for col in temp_df.columns:
        temp_list = temp_df[col].values[0].split("  ")
        for i, item in enumerate(temp_list):
            if i % 2 != 0:
                price_pct, pct = item.split(" ")
                price_pct = price_pct.strip("%").strip("+").strip("-")
                pct = pct.strip("-").strip("+")
                new_list.append([col, temp_list[i - 1], price_pct, pct])
    temp_df = pd.DataFrame(new_list, columns=["产业链", "名称", "涨跌幅", "现价"])
    temp_df["涨跌幅"] = pd.to_numeric(temp_df["涨跌幅"], errors="coerce")
    temp_df["现价"] = pd.to_numeric(temp_df["现价"], errors="coerce")
    return temp_df


def _get_file_content_ths(file: str = "ths.js") -> str:
    """
    获取 JS 文件的内容
    :param file:  JS 文件名
    :type file: str
    :return: 文件内容
    :rtype: str
    """
    setting_file_path = get_ths_js(file)
    with open(setting_file_path) as f:
        file_data = f.read()
    return file_data


@lru_cache()
def stock_board_concept_name_ths() -> pd.DataFrame:
    """
    同花顺-板块-概念板块-概念
    http://q.10jqka.com.cn/gn/detail/code/301558/
    :return: 所有概念板块的名称和链接
    :rtype: pandas.DataFrame
    """
    url = "http://q.10jqka.com.cn/gn/index/field/addtime/order/desc/page/1/ajax/1/"
    js_code = py_mini_racer.MiniRacer()
    js_content = _get_file_content_ths("ths.js")
    js_code.eval(js_content)
    v_code = js_code.call("v")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36",
        "Cookie": f"v={v_code}",
    }
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, features="lxml")
    total_page = soup.find(name="span", attrs={"class": "page_info"}).text.split("/")[1]
    big_df = pd.DataFrame()
    for page in tqdm(range(1, int(total_page) + 1), leave=False):
        url = f"http://q.10jqka.com.cn/gn/index/field/addtime/order/desc/page/{page}/ajax/1/"
        js_code = py_mini_racer.MiniRacer()
        js_content = _get_file_content_ths("ths.js")
        js_code.eval(js_content)
        v_code = js_code.call("v")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36",
            "Cookie": f"v={v_code}",
        }
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, features="lxml")
        url_list = []
        for item in (
            soup.find(name="table", attrs={"class": "m-table m-pager-table"})
            .find("tbody")
            .find_all("tr")
        ):
            inner_url = item.find_all("td")[1].find("a")["href"]
            url_list.append(inner_url)
        temp_df = pd.read_html(StringIO(r.text))[0]
        temp_df["网址"] = url_list
        big_df = pd.concat(objs=[big_df, temp_df], ignore_index=True)
    big_df = big_df[["日期", "概念名称", "成分股数量", "网址"]]
    big_df["日期"] = pd.to_datetime(big_df["日期"], errors="coerce").dt.date
    big_df["成分股数量"] = pd.to_numeric(big_df["成分股数量"], errors="coerce")
    big_df["代码"] = big_df["网址"].str.split("/", expand=True).iloc[:, 6]
    big_df.drop_duplicates(keep="last", inplace=True)
    big_df.reset_index(inplace=True, drop=True)

    # 处理遗漏的板块
    url = "http://q.10jqka.com.cn/gn/detail/code/301558/"
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "lxml")
    need_list = [
        item.find_all("a") for item in soup.find_all(attrs={"class": "cate_group"})
    ]
    temp_list = []
    for item in need_list:
        temp_list.extend(item)
    temp_df = pd.DataFrame(
        [
            [item.text for item in temp_list],
            [item["href"] for item in temp_list],
        ]
    ).T
    temp_df.columns = ["概念名称", "网址"]
    temp_df["日期"] = None
    temp_df["成分股数量"] = None
    temp_df["代码"] = temp_df["网址"].str.split("/", expand=True).iloc[:, 6].tolist()
    temp_df = temp_df[["日期", "概念名称", "成分股数量", "网址", "代码"]]
    big_df = pd.concat(objs=[big_df, temp_df], ignore_index=True)
    big_df.drop_duplicates(subset=["概念名称"], keep="first", inplace=True)
    return big_df


def _stock_board_concept_code_ths() -> dict:
    """
    同花顺-板块-概念板块-概念
    http://q.10jqka.com.cn/gn/detail/code/301558/
    :return: 所有概念板块的名称和链接
    :rtype: pandas.DataFrame
    """
    _stock_board_concept_name_ths_df = stock_board_concept_name_ths()
    name_list = _stock_board_concept_name_ths_df["概念名称"].tolist()
    url_list = [
        item.split("/")[-2] for item in _stock_board_concept_name_ths_df["网址"].tolist()
    ]
    temp_map = dict(zip(name_list, url_list))
    return temp_map


def stock_board_concept_cons_ths(symbol: str = "阿里巴巴概念") -> pd.DataFrame:
    """
    同花顺-板块-概念板块-成份股
    http://q.10jqka.com.cn/gn/detail/code/301558/
    :param symbol: 板块名称
    :type symbol: str
    :return: 成份股
    :rtype: pandas.DataFrame
    """
    stock_board_ths_map_df = stock_board_concept_name_ths()
    symbol = (
        stock_board_ths_map_df[stock_board_ths_map_df["概念名称"] == symbol]["网址"]
        .values[0]
        .split("/")[-2]
    )
    js_code = py_mini_racer.MiniRacer()
    js_content = _get_file_content_ths("ths.js")
    js_code.eval(js_content)
    v_code = js_code.call("v")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36",
        "Cookie": f"v={v_code}",
    }
    url = f"http://q.10jqka.com.cn/gn/detail/field/264648/order/desc/page/1/ajax/1/code/{symbol}"
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, features="lxml")
    try:
        page_num = int(soup.find_all(name="a", attrs={"class": "changePage"})[-1]["page"])
    except IndexError as e:
        page_num = 1
    big_df = pd.DataFrame()
    for page in tqdm(range(1, page_num + 1), leave=False):
        v_code = js_code.call("v")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36",
            "Cookie": f"v={v_code}",
        }
        url = f"http://q.10jqka.com.cn/gn/detail/field/264648/order/desc/page/{page}/ajax/1/code/{symbol}"
        r = requests.get(url, headers=headers)
        temp_df = pd.read_html(StringIO(r.text))[0]
        big_df = pd.concat(objs=[big_df, temp_df], ignore_index=True)
    big_df.rename(
        mapper={
            "涨跌幅(%)": "涨跌幅",
            "涨速(%)": "涨速",
            "换手(%)": "换手",
            "振幅(%)": "振幅",
        },
        inplace=True,
        axis=1,
    )
    del big_df["加自选"]
    big_df["代码"] = big_df["代码"].astype(str).str.zfill(6)
    big_df = big_df[big_df["代码"] != "暂无成份股数据"]
    return big_df


def stock_board_concept_info_ths(symbol: str = "阿里巴巴概念") -> pd.DataFrame:
    """
    同花顺-板块-概念板块-板块简介
    http://q.10jqka.com.cn/gn/detail/code/301558/
    :param symbol: 板块简介
    :type symbol: str
    :return: 板块简介
    :rtype: pandas.DataFrame
    """
    stock_board_ths_map_df = stock_board_concept_name_ths()
    symbol_code = (
        stock_board_ths_map_df[stock_board_ths_map_df["概念名称"] == symbol]["网址"]
        .values[0]
        .split("/")[-2]
    )
    url = f"http://q.10jqka.com.cn/gn/detail/code/{symbol_code}/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36",
    }
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, features="lxml")
    name_list = [
        item.text
        for item in soup.find(name="div", attrs={"class": "board-infos"}).find_all("dt")
    ]
    value_list = [
        item.text.strip().replace("\n", "/")
        for item in soup.find(name="div", attrs={"class": "board-infos"}).find_all("dd")
    ]
    temp_df = pd.DataFrame([name_list, value_list]).T
    temp_df.columns = ["项目", "值"]
    return temp_df


def stock_board_concept_hist_ths(
    start_year: str = "2000", symbol: str = "安防"
) -> pd.DataFrame:
    """
    同花顺-板块-概念板块-指数数据
    http://q.10jqka.com.cn/gn/detail/code/301558/
    :param start_year: 开始年份; e.g., 2019
    :type start_year: str
    :param symbol: 板块简介
    :type symbol: str
    :return: 板块简介
    :rtype: pandas.DataFrame
    """
    code_map = _stock_board_concept_code_ths()
    symbol_url = f"http://q.10jqka.com.cn/gn/detail/code/{code_map[symbol]}/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36",
    }
    r = requests.get(symbol_url, headers=headers)
    soup = BeautifulSoup(r.text, "lxml")
    symbol_code = soup.find("div", attrs={"class": "board-hq"}).find("span").text
    big_df = pd.DataFrame()
    current_year = datetime.now().year
    for year in tqdm(range(int(start_year), current_year + 1), leave=False):
        url = f"http://d.10jqka.com.cn/v4/line/bk_{symbol_code}/01/{year}.js"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36",
            "Referer": "http://q.10jqka.com.cn",
            "Host": "d.10jqka.com.cn",
        }
        r = requests.get(url, headers=headers)
        data_text = r.text
        try:
            demjson.decode(data_text[data_text.find("{") : -1])
        except:
            continue
        temp_df = demjson.decode(data_text[data_text.find("{") : -1])
        temp_df = pd.DataFrame(temp_df["data"].split(";"))
        temp_df = temp_df.iloc[:, 0].str.split(",", expand=True)
        big_df = pd.concat([big_df, temp_df], ignore_index=True)
    if big_df.columns.shape[0] == 12:
        big_df.columns = [
            "日期",
            "开盘价",
            "最高价",
            "最低价",
            "收盘价",
            "成交量",
            "成交额",
            "_",
            "_",
            "_",
            "_",
            "_",
        ]
    else:
        big_df.columns = [
            "日期",
            "开盘价",
            "最高价",
            "最低价",
            "收盘价",
            "成交量",
            "成交额",
            "_",
            "_",
            "_",
            "_",
        ]
    big_df = big_df[
        [
            "日期",
            "开盘价",
            "最高价",
            "最低价",
            "收盘价",
            "成交量",
            "成交额",
        ]
    ]
    big_df["日期"] = pd.to_datetime(big_df["日期"], errors="coerce").dt.date
    big_df["开盘价"] = pd.to_numeric(big_df["开盘价"], errors="coerce")
    big_df["最高价"] = pd.to_numeric(big_df["最高价"], errors="coerce")
    big_df["最低价"] = pd.to_numeric(big_df["最低价"], errors="coerce")
    big_df["收盘价"] = pd.to_numeric(big_df["收盘价"], errors="coerce")
    big_df["成交量"] = pd.to_numeric(big_df["成交量"], errors="coerce")
    big_df["成交额"] = pd.to_numeric(big_df["成交额"], errors="coerce")
    return big_df

def get_cookie_list():
    """
    :param : 从浏览器cv的或者去从requests里获取到的cookie
    :return:       反回cookie列表
    """
    cookie_list = []
    
    cookie = '_ga=GA1.1.1299022992.1734092265; _ga_KQBDS1VPQF=GS1.1.1734094444.2.0.1734094444.0.0.0; u_ukey=A10702B8689642C6BE607730E11E6E4A; u_uver=1.0.0; u_dpass=Oh%2F05wXd%2BzKE%2Bt4C17p14g4TIsHrt1s3VphvFWS8TkrbTgehAY6z2bnCqEyejttHHi80LrSsTFH9a%2B6rtRvqGg%3D%3D; u_did=56E5C5134B3D4853B9D974CC099BAF49; u_ttype=WEB; spversion=20130314; Hm_lvt_78c58f01938e4d85eaf619eae71b4ed1=1734752393,1735223285,1735349268; HMACCOUNT=870BE49DFDDF23DC; searchGuide=sg; historystock=601116%7C*%7C600839%7C*%7C600354%7C*%7C002085%7C*%7C831175; ttype=WEB; user_status=0; Hm_lvt_722143063e4892925903024537075d0d=1735381215; Hm_lvt_929f8b362150b1f77b477230541dbbc2=1735381215; log=; Hm_lpvt_722143063e4892925903024537075d0d=1735388207; Hm_lpvt_929f8b362150b1f77b477230541dbbc2=1735388207; Hm_lpvt_78c58f01938e4d85eaf619eae71b4ed1=1735388207; user=MDpteF82Njc3MTYxMDI6Ok5vbmU6NTAwOjY3NzcxNjEwMjo3LDExMTExMTExMTExLDQwOzQ0LDExLDQwOzYsMSw0MDs1LDEsNDA7MSwxMDEsNDA7MiwxLDQwOzMsMSw0MDs1LDEsNDA7OCwwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMSw0MDsxMDIsMSw0MDoxNjo6OjY2NzcxNjEwMjoxNzM1Mzg4MzM1Ojo6MTY3NzE2NTc4MDo2MDQ4MDA6MDoxYjM0MmNlMTc4MTM0OTNkZDMxNzIwMzQ1NWE2NmQwOTI6ZGVmYXVsdF80OjA%3D; userid=667716102; u_name=mx_667716102; escapename=mx_667716102; ticket=1b60761778a51f638f9b880c746bb02b; utk=abb8be3a9194c0a9843c8fb6f4f1a435; v=A4_BP9VONOmEsjAVfT4o2K8CGCictOI1fQvnyqGeLt8FUqHWqYRzJo3Ydxay'
    for i in cookie.split(';'):
        i_dict = {'name': i.split('=')[0].strip(), 'value': i.split('=')[1].strip()}
        cookie_list.append(i_dict)

    return cookie_list

def parse_stock_board_base_on_page(driver, symbol, page, url_flag):
    url = f"https://q.10jqka.com.cn/{url_flag}/detail/field/199112/order/desc/page/{page}/ajax/1/code/{symbol}"
    print(url)
    # 随机停顿几秒，你可以不停顿，或者改的更长/更短的时间
    sleepSeconds = random.randint(1,2)
    time.sleep(sleepSeconds)
    page_df=pd.DataFrame()
    seccessIdc = False

    # driver.get('https://q.10jqka.com.cn/')
    # time.sleep(3)
    try:
        # cookie_list = get_cookie_list()
        # for c_i in cookie_list:
        #     driver.add_cookie(c_i)
        # url = 'https://q.10jqka.com.cn/gn/detail/field/199112/order/desc/page/10/ajax/1/code/301558'
        driver.get(url)
        if "Nginx forbidden" in driver.page_source:
            print(url, driver.page_source)
            raise Exception("Nginx forbidden")
        page_df = pd.read_html(StringIO(driver.page_source))[0]
        if np.isnan(page_df.iloc[0,0]):
            raise Exception("NaN value found in page_df")
        # big_df = pd.concat([big_df, temp_df], ignore_index=True)
        page_df["代码"] = page_df["代码"].astype(str).str.zfill(6)
        page_df["概念"] = symbol
        seccessIdc=True
    except Exception as e:
        print(e)
    return page_df, seccessIdc


def stock_board_cons_ths(symbol: str = "301558") -> pd.DataFrame:
    """
    通过输入行业板块或者概念板块的代码获取成份股
    http://q.10jqka.com.cn/thshy/detail/code/881121/
    http://q.10jqka.com.cn/gn/detail/code/301558/
    :param symbol: 行业板块或者概念板块的代码
    :type symbol: str
    :return: 行业板块或者概念板块的成份股
    :rtype: pandas.DataFrame
    """
    # js_code = py_mini_racer.MiniRacer()
    # js_content = _get_file_content_ths("ths.js")
    # js_code.eval(js_content)
    # v_code = js_code.call("v")
    # headers = {
    #     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36",
    #     "Cookie": f"v={v_code}",
    # }
    driver=get_selenium_driver()
    url = f"http://q.10jqka.com.cn/thshy/detail/field/199112/order/desc/page/1/ajax/1/code/{symbol}"
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, "lxml")
    url_flag = "thshy" # 行业
    if soup.find("td", attrs={"colspan": "14"}):
        url = f"https://q.10jqka.com.cn/gn/detail/field/199112/order/desc/page/1/ajax/1/code/{symbol}"
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, "lxml")
        url_flag = "gn" # 概念
    try:
        page_num = int(soup.find_all("a", attrs={"class": "changePage"})[-1]["page"])
    except IndexError as e:
        page_num = 1
    big_df = pd.DataFrame()
    fail_Record ={}
    for page in tqdm(range(1, page_num + 1), leave=False):
        # v_code = js_code.call("v")
        # headers = {
        #     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36",
        #     "Cookie": f"v={v_code}",
        # }
        page_df, seccessIdc = parse_stock_board_base_on_page(driver, symbol, page, url_flag)
        if seccessIdc:
            big_df = pd.concat([big_df, page_df], ignore_index=True)
        else:
            fail_Record[page]=[symbol, page, url_flag, url]
            continue
        # url = f"https://q.10jqka.com.cn/{url_flag}/detail/field/199112/order/desc/page/{page}/ajax/1/code/{symbol}"
        # print(url)
        # # 随机停顿几秒，你可以不停顿，或者改的更长/更短的时间
        # sleepSeconds = random.randint(1,2)
        # time.sleep(sleepSeconds) 
        # try:
        #     driver.get(url)
        #     temp_df = pd.read_html(StringIO(driver.page_source))[0]
        #     if np.isnan(temp_df.iloc[0][0]):
        #         raise Exception("NaN value found in temp_df")
        #     big_df = pd.concat([big_df, temp_df], ignore_index=True)
        #     big_df["代码"] = big_df["代码"].astype(str).str.zfill(6)
        #     big_df["概念"] = symbol
        # except Exception as e:
        #     print(e)
        #     fail_Record[page]=url

    # big_df.rename(
    #     {
    #         "涨跌幅(%)": "涨跌幅",
    #         "涨速(%)": "涨速",
    #         "换手(%)": "换手",
    #         "振幅(%)": "振幅",
    #     },
    #     inplace=True,
    #     axis=1,
    # )
    # del big_df["加自选"]
    # big_df["代码"] = big_df["代码"].astype(str).str.zfill(6)
    # big_df["概念"] = symbol
    return big_df[['概念','代码', '名称']], fail_Record

def get_all_gn(gn_list=[]):

    gn_result=pd.DataFrame()
    fail_Record={}

    for symbol in gn_list:
        stock_board_cons_ths(symbol=str(symbol))
        result, fail_Record[symbol]=stock_board_cons_ths(symbol=str(symbol))
        gn_result=pd.concat([gn_result,result],ignore_index=True)

    return gn_result,fail_Record


def save_to_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def regen_failed_gn(gn_result_pkl_file, fail_Record_pkl_file,progress=gr.Progress()):
    progress(0, desc="重试开始...")
    big_df = load_pickle(gn_result_pkl_file)
    fail_Record = load_pickle(fail_Record_pkl_file)
    driver=get_selenium_driver(require_session=True)
    # driver.get('https://q.10jqka.com.cn/')
    # time.sleep(3)

    # # we need the session to be logged in
    # cookie_list = get_cookie_list()
    # for c_i in cookie_list:
    #     driver.add_cookie(c_i)
    # time.sleep(3)
    cnt=0
    for symbol in progress.tqdm(fail_Record.keys()):
        pages=list(fail_Record[symbol].keys())
        for page in pages:
            symbol, page, url_flag, url = fail_Record[symbol][page]
            # 每隔几次， 重置cookies https://github.com/akfamily/akshare/issues/4946
            if cnt%4==0:
                driver.quit()
                driver=get_selenium_driver(require_session=True)
            page_df, seccessIdc = parse_stock_board_base_on_page(driver, symbol, page, url_flag)
            if seccessIdc:
                big_df = pd.concat([big_df, page_df], ignore_index=True)
                fail_Record[symbol].pop(page)
            else:
                # TODO we can add a counter to retry the failed page
                print("url failed", url, cnt)
            cnt = cnt + 1
    return big_df[['概念','代码', '名称']], fail_Record

def get_selenium_driver(require_session=False):

    user_agent_list = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.3",
            "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Safari/605.1.1",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    ]

    options = webdriver.ChromeOptions()
    options.add_argument('lang=zh_CN.UTF-8')
    options.add_argument(f'--user-agent={random.choice(user_agent_list)}')
    options.add_argument('--ignore-certificate-errors')
    # linux下所需参数
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-extensions')
    options.add_argument('--headless')
    options.add_argument('--disable-blink-features=AutomationControlled')

    # options.add_argument('--disable-extensions')
    # options.add_argument('--disable-gpu')
    # options.add_argument('--disable-infobars')
    # options.add_argument('--disable-notifications')
    # options.add_argument('--disable-popup-blocking')
    # options.add_argument('--disable-web-security')
    # options.add_argument('--ignore-certificate-errors')
    # options.add_argument('--no-sandbox')
    # options.add_argument('--start-maximized')
    # options.add_experimental_option('excludeSwitches', ['enable-automation', 'useAutomationExtension'])
    # options.add_argument('--headless')

    if platform.system() == "Windows":
        # service = Service(chrome_driver_path)
        # driver = webdriver.Chrome(service=service, options=options)
        pass
    else:
        driver = webdriver.Chrome(options=options)

    driver.get('https://q.10jqka.com.cn/')
    time.sleep(3)

    if require_session:
        # we need the session to be logged in
        cookie_list = get_cookie_list()
        for c_i in cookie_list:
            driver.add_cookie(c_i)

    return driver

def get_data(symbol: str = "301558") -> pd.DataFrame:

    url = f"https://q.10jqka.com.cn/gn/detail/field/199112/order/desc/page/2/ajax/18/code/301558/"

    # url = "http://q.10jqka.com.cn/gn/index/field/addtime/order/desc/page/1/ajax/1/"
    js_code = py_mini_racer.MiniRacer()
    js_content = _get_file_content_ths("ths.js")
    js_code.eval(js_content)
    v_code = js_code.call("v")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36",
        # "Cookie": f"v={v_code}",
        "Cookie": f'_ga=GA1.1.1299022992.1734092265; _ga_KQBDS1VPQF=GS1.1.1734094444.2.0.1734094444.0.0.0; u_ukey=A10702B8689642C6BE607730E11E6E4A; u_uver=1.0.0; u_dpass=Oh%2F05wXd%2BzKE%2Bt4C17p14g4TIsHrt1s3VphvFWS8TkrbTgehAY6z2bnCqEyejttHHi80LrSsTFH9a%2B6rtRvqGg%3D%3D; u_did=56E5C5134B3D4853B9D974CC099BAF49; u_ttype=WEB; spversion=20130314; Hm_lvt_78c58f01938e4d85eaf619eae71b4ed1=1734752393,1735223285,1735349268; HMACCOUNT=870BE49DFDDF23DC; searchGuide=sg; historystock=601116%7C*%7C600839%7C*%7C600354%7C*%7C002085%7C*%7C831175; ttype=WEB; user_status=0; Hm_lvt_722143063e4892925903024537075d0d=1735381215; Hm_lvt_929f8b362150b1f77b477230541dbbc2=1735381215; log=; Hm_lpvt_722143063e4892925903024537075d0d=1735388207; Hm_lpvt_929f8b362150b1f77b477230541dbbc2=1735388207; Hm_lpvt_78c58f01938e4d85eaf619eae71b4ed1=1735388207; user=MDpteF82Njc3MTYxMDI6Ok5vbmU6NTAwOjY3NzcxNjEwMjo3LDExMTExMTExMTExLDQwOzQ0LDExLDQwOzYsMSw0MDs1LDEsNDA7MSwxMDEsNDA7MiwxLDQwOzMsMSw0MDs1LDEsNDA7OCwwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMSw0MDsxMDIsMSw0MDoxNjo6OjY2NzcxNjEwMjoxNzM1Mzg4MzM1Ojo6MTY3NzE2NTc4MDo2MDQ4MDA6MDoxYjM0MmNlMTc4MTM0OTNkZDMxNzIwMzQ1NWE2NmQwOTI6ZGVmYXVsdF80OjA%3D; userid=667716102; u_name=mx_667716102; escapename=mx_667716102; ticket=1b60761778a51f638f9b880c746bb02b; utk=abb8be3a9194c0a9843c8fb6f4f1a435; v=A4_BP9VONOmEsjAVfT4o2K8CGCictOI1fQvnyqGeLt8FUqHWqYRzJo3Ydxay'
    }

    # value='_ga=GA1.1.1299022992.1734092265; _ga_KQBDS1VPQF=GS1.1.1734094444.2.0.1734094444.0.0.0; u_ukey=A10702B8689642C6BE607730E11E6E4A; u_uver=1.0.0; u_dpass=Oh%2F05wXd%2BzKE%2Bt4C17p14g4TIsHrt1s3VphvFWS8TkrbTgehAY6z2bnCqEyejttHHi80LrSsTFH9a%2B6rtRvqGg%3D%3D; u_did=56E5C5134B3D4853B9D974CC099BAF49; u_ttype=WEB; spversion=20130314; Hm_lvt_78c58f01938e4d85eaf619eae71b4ed1=1734752393,1735223285,1735349268; HMACCOUNT=870BE49DFDDF23DC; searchGuide=sg; historystock=601116%7C*%7C600839%7C*%7C600354%7C*%7C002085%7C*%7C831175; ttype=WEB; user_status=0; Hm_lvt_722143063e4892925903024537075d0d=1735381215; Hm_lvt_929f8b362150b1f77b477230541dbbc2=1735381215; log=; Hm_lpvt_722143063e4892925903024537075d0d=1735388207; Hm_lpvt_929f8b362150b1f77b477230541dbbc2=1735388207; Hm_lpvt_78c58f01938e4d85eaf619eae71b4ed1=1735388207; user=MDpteF82Njc3MTYxMDI6Ok5vbmU6NTAwOjY3NzcxNjEwMjo3LDExMTExMTExMTExLDQwOzQ0LDExLDQwOzYsMSw0MDs1LDEsNDA7MSwxMDEsNDA7MiwxLDQwOzMsMSw0MDs1LDEsNDA7OCwwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMSw0MDsxMDIsMSw0MDoxNjo6OjY2NzcxNjEwMjoxNzM1Mzg4MzM1Ojo6MTY3NzE2NTc4MDo2MDQ4MDA6MDoxYjM0MmNlMTc4MTM0OTNkZDMxNzIwMzQ1NWE2NmQwOTI6ZGVmYXVsdF80OjA%3D; userid=667716102; u_name=mx_667716102; escapename=mx_667716102; ticket=1b60761778a51f638f9b880c746bb02b; utk=abb8be3a9194c0a9843c8fb6f4f1a435; v=A4_BP9VONOmEsjAVfT4o2K8CGCictOI1fQvnyqGeLt8FUqHWqYRzJo3Ydxay',

    r = requests.get(url, headers=headers)
    # r = requests.get(url, 
    #                 #  headers=headers
    #                 cookies={'u_ukey': 'A10702B8689642C6BE607730E11E6E4A',}
    #                  )
    # value='_ga=GA1.1.1299022992.1734092265; _ga_KQBDS1VPQF=GS1.1.1734094444.2.0.1734094444.0.0.0; u_ukey=A10702B8689642C6BE607730E11E6E4A; u_uver=1.0.0; u_dpass=Oh%2F05wXd%2BzKE%2Bt4C17p14g4TIsHrt1s3VphvFWS8TkrbTgehAY6z2bnCqEyejttHHi80LrSsTFH9a%2B6rtRvqGg%3D%3D; u_did=56E5C5134B3D4853B9D974CC099BAF49; u_ttype=WEB; spversion=20130314; Hm_lvt_78c58f01938e4d85eaf619eae71b4ed1=1734752393,1735223285,1735349268; HMACCOUNT=870BE49DFDDF23DC; searchGuide=sg; historystock=601116%7C*%7C600839%7C*%7C600354%7C*%7C002085%7C*%7C831175; ttype=WEB; user_status=0; Hm_lvt_722143063e4892925903024537075d0d=1735381215; Hm_lvt_929f8b362150b1f77b477230541dbbc2=1735381215; log=; Hm_lpvt_722143063e4892925903024537075d0d=1735388207; Hm_lpvt_929f8b362150b1f77b477230541dbbc2=1735388207; Hm_lpvt_78c58f01938e4d85eaf619eae71b4ed1=1735388207; user=MDpteF82Njc3MTYxMDI6Ok5vbmU6NTAwOjY3NzcxNjEwMjo3LDExMTExMTExMTExLDQwOzQ0LDExLDQwOzYsMSw0MDs1LDEsNDA7MSwxMDEsNDA7MiwxLDQwOzMsMSw0MDs1LDEsNDA7OCwwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMSw0MDsxMDIsMSw0MDoxNjo6OjY2NzcxNjEwMjoxNzM1Mzg4MzM1Ojo6MTY3NzE2NTc4MDo2MDQ4MDA6MDoxYjM0MmNlMTc4MTM0OTNkZDMxNzIwMzQ1NWE2NmQwOTI6ZGVmYXVsdF80OjA%3D; userid=667716102; u_name=mx_667716102; escapename=mx_667716102; ticket=1b60761778a51f638f9b880c746bb02b; utk=abb8be3a9194c0a9843c8fb6f4f1a435; v=A4_BP9VONOmEsjAVfT4o2K8CGCictOI1fQvnyqGeLt8FUqHWqYRzJo3Ydxay',

    return r.text

    user_agent_list = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.3",
            "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Safari/605.1.1",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    ]

    options = webdriver.ChromeOptions()
    options.add_argument('lang=zh_CN.UTF-8')
    options.add_argument(f'--user-agent={random.choice(user_agent_list)}')
    options.add_argument('--ignore-certificate-errors')
    # linux下所需参数
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-extensions')
    options.add_argument('headless')
    options.add_argument('--disable-blink-features=AutomationControlled')

    # options.add_argument('--disable-extensions')
    # options.add_argument('--disable-gpu')
    # options.add_argument('--disable-infobars')
    # options.add_argument('--disable-notifications')
    # options.add_argument('--disable-popup-blocking')
    # options.add_argument('--disable-web-security')
    # options.add_argument('--ignore-certificate-errors')
    # options.add_argument('--no-sandbox')
    # options.add_argument('--start-maximized')
    # options.add_experimental_option('excludeSwitches', ['enable-automation', 'useAutomationExtension'])
    # options.add_argument('--headless')

    # 当前文件夹里chromedriver路径

    if platform.system() == "Windows":
        # service = Service(chrome_driver_path)
        # driver = webdriver.Chrome(service=service, options=options)
        pass
    else:
        driver = webdriver.Chrome(options=options)
        # driver.implicitly_wait(5)

    driver.get(url)
    data_text = driver.page_source
    # import sys
    # sys.sleep(5)
    # 构建请求参数

    # # 隐藏navigator.webdriver标志，将其值修改为false或undefined
    # driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
    #     'source': 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'
    # })
    # # 设置user-agent，改变user-agent的值
    # user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    # driver.execute_cdp_cmd("Network.setUserAgentOverride", {"userAgent": user_agent})



    return data_text


if __name__ == "__main__":
    # stock_board_concept_graph_ths_df = stock_board_concept_graph_ths(symbol="通用航空")
    # print(stock_board_concept_graph_ths_df)

    # stock_board_concept_name_ths_df = stock_board_concept_name_ths()
    # print(stock_board_concept_name_ths_df)

    # stock_board_concept_cons_ths_df = stock_board_concept_cons_ths(symbol="小米概念")
    # print(stock_board_concept_cons_ths_df)

    # stock_board_concept_info_ths_df = stock_board_concept_info_ths(symbol="PVDF概念")
    # print(stock_board_concept_info_ths_df)

    # stock_board_concept_hist_ths_df = stock_board_concept_hist_ths(
    #     start_year="2023", symbol="新能源汽车"
    # )
    # print(stock_board_concept_hist_ths_df)

    print(get_data(symbol="309115"))
    # stock_board_cons_ths_df = stock_board_cons_ths(symbol="301558")
    # print(stock_board_cons_ths_df)

    # gn_result, fail_Record = get_all_gn()
    # save_to_pickle(gn_result, 'gn_result.pkl')
    # save_to_pickle(fail_Record, 'fail_record.pkl')

    # gn_result, fail_Record = regen_failed_gn('gn_result.pkl', 'fail_record.pkl')
    # save_to_pickle(gn_result, 'gn_result_new.pkl')
    # save_to_pickle(fail_Record, 'fail_record_new.pkl')

    # big_df = load_pickle('gn_result_new.pkl')
    # fail_Record = load_pickle('fail_record_new.pkl')
    # a=big_df[big_df["代码"] == '002312']["概念"].unique()
    # print(a)