#네이버 중고나라 명품 카테고리 별 크롤링

# -----------------------  Step 0. 준비
import sys  # 시스템
import os  # 시스템
import pandas as pd  # 판다스 : 데이터분석 라이브러리
import numpy as np  # 넘파이 : 숫자, 행렬 데이터 라이브러리
from bs4 import BeautifulSoup  # html 데이터 전처리
from selenium import webdriver  # 웹 브라우저 자동화
import time  # 시간 지연
import math
import datetime
df_titles = pd.DataFrame()
# 카테고리별 이름 입력하기
category = ['크롤링할 카테고리 이름']

# -----------------------  Step 1. 크롤링
# 크롤링할 데이터 양 입력
crawling_no = int(input())
# 크롬 웹브라우저 실행
driver = webdriver.Chrome("./chromedriver.exe")
# 사이트 주소
driver.get("https://cafe.naver.com/joonggonara")
time.sleep(2)
# 게시판 클릭
driver.get("https://cafe.naver.com/joonggonara")
time.sleep(2)
driver.find_element_by_xpath('//*[@id="menuLink791"]').click()
# //*[@id="menuLink782"]가방
# //*[@id="menuLink787"] 패션잡화
# //*[@id="menuLink785"]쥬얼리
# //*[@id="menuLink791"] 아동
# //*[@id="menuLink1007"]여성의류
# //*[@id="menuLink1008"]남성의류
# //*[@id="menuLink1009"]여성신발
# //*[@id="menuLink1010"]남성신발
# //*[@id="menuLink1011"] 시계

# 게시판 프레임
driver.switch_to.frame("cafe_main")

# 게시글 50개씩 보기 클릭
driver.find_element_by_xpath('//*[@id="listSizeSelectDiv"]').click()
driver.find_element_by_xpath('//*[@id="listSizeSelectDiv"]/ul/li[7]/a').click()

# 크롤링 리스트
no_data = []
title_data = []

# 크롤링 해야 할 페이지 계산
crawling_page = int(math.ceil(crawling_no / 50) + 1)

try:
    for page in range(1, crawling_page):
        # 페이지 클릭
        driver.find_element_by_link_text(str(page)).click()
        time.sleep(1)
        # 글 번호 수집
        no = [i.text for i in driver.find_elements_by_css_selector('.td_article')]
        no_split = [ni.split()[0] for ni in no]
        # 글 제목 수집
        title = [i.text for i in driver.find_elements_by_css_selector('.article')]
        # 수집 데이터 리스트화
        no_data.append(no_split)
        title_data.append(title)
        # 10페이지 마다 프린트 & 다음 페이지로 클릭
        if str(page)[-1] == '0':
            print(int(page), 'page 크롤링 완료')
            driver.find_element_by_link_text('다음').click()
# 더이상 페이지가 존재하지 않을 시
except:
    print('더이상 페이지가 존재하지 않음')
title_list = sum(title_data, [])
no_list = sum(no_data, [])

# -----------------------  Step 3. 데이터프레임 저장
df = pd.DataFrame({'번호': no_list, '제목': title_list, '분류': category[0]})
# 공지글 삭제
df = df.drop(df[df['번호'] == '필독'].index)
df = df.drop(df[df['번호'] == '공지'].index)
df = df.drop(columns=['번호'])
df = df.reset_index(drop=True)
# csv저장
df.to_csv('./crawling_data/crawling_data_{}.csv'.format(category[0]), index=False)

driver.close()
