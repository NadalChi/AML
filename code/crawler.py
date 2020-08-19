from bs4 import BeautifulSoup
import csv
import datetime
import fractions
import json
import os
import random
import re
import time
import urllib.parse

import js2py
import loguru
import pandas
import pyquery
import requests


now = datetime.datetime.now()
proxies = []
proxy = None

def getProxy():
    global proxies
    if len(proxies) == 0:
        getProxies()
    proxy = random.choice(proxies)
    loguru.logger.debug(f'getProxy: {proxy}')
    proxies.remove(proxy)
    loguru.logger.debug(f'getProxy: {len(proxies)} proxies is unused.')
    return proxy

def reqProxies(hour):
    global proxies
    #proxies = proxies + getProxiesFromProxyNova()
    #proxies = proxies + getProxiesFromGatherProxy()
    proxies = proxies + getProxiesFromFreeProxyList()
    proxies = list(dict.fromkeys(proxies))
    loguru.logger.debug(f'reqProxies: {len(proxies)} proxies is found.')

def getProxies():
    global proxies
    hour = f'{now:%Y%m%d%H}'
    filename = f'proxies-{hour}.csv'
    filepath = f'{filename}'
    if os.path.isfile(filepath):
        loguru.logger.info(f'getProxies: {filename} exists.')
        loguru.logger.warning(f'getProxies: {filename} is loading...')
        with open(filepath, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                proxy = row['Proxy']
                proxies.append(proxy)
        loguru.logger.success(f'getProxies: {filename} is loaded.')
    else:
        loguru.logger.info(f'getProxies: {filename} does not exist.')
        reqProxies(hour)
        loguru.logger.warning(f'getProxies: {filename} is saving...')
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Proxy'
            ])
            for proxy in proxies:
                writer.writerow([
                    proxy
                ])
        loguru.logger.success(f'getProxies: {filename} is saved.')

def getProxiesFromProxyNova():
    proxies = []
    countries = [
        'tw',
        'jp',
        'kr',
        'id',
        'my',
        'th',
        'vn',
        'ph',
        'hk',
        'uk',
        'us'
    ]
    for country in countries:
        url = f'https://www.proxynova.com/proxy-server-list/country-{country}/'
        loguru.logger.debug(f'getProxiesFromProxyNova: {url}')
        loguru.logger.warning(f'getProxiesFromProxyNova: downloading...')
        response = requests.get(url)
        if response.status_code != 200:
            loguru.logger.debug(f'getProxiesFromProxyNova: status code is not 200')
            continue
        loguru.logger.success(f'getProxiesFromProxyNova: downloaded.')
        d = pyquery.PyQuery(response.text)
        table = d('table#tbl_proxy_list')
        rows = list(table('tbody:first > tr').items())
        loguru.logger.warning(f'getProxiesFromProxyNova: scanning...')
        for row in rows:
            tds = list(row('td').items())
            if len(tds) == 1:
                continue
            js = row('td:nth-child(1) > abbr').text()
            js = 'let x = %s; x' % (js[15:-2])
            ip = js2py.eval_js(js).strip()
            port = row('td:nth-child(2)').text().strip()
            proxy = f'{ip}:{port}'
            proxies.append(proxy)
        loguru.logger.success(f'getProxiesFromProxyNova: scanned.')
        loguru.logger.debug(f'getProxiesFromProxyNova: {len(proxies)} proxies is found.')
        time.sleep(1)
    return proxies

def getProxiesFromGatherProxy():
    proxies = []
    countries = [
        'Taiwan',
        'Japan',
        'United States',
        'Thailand',
        'Vietnam',
        'Indonesia',
        'Singapore',
        'Philippines',
        'Malaysia',
        'Hong Kong'
    ]
    for country in countries:
        url = f'http://www.gatherproxy.com/proxylist/country/?c={urllib.parse.quote(country)}'
        loguru.logger.debug(f'getProxiesFromGatherProxy: {url}')
        loguru.logger.warning(f'getProxiesFromGatherProxy: downloading...')
        response = requests.get(url)
        if response.status_code != 200:
            loguru.logger.debug(f'getProxiesFromGatherProxy: status code is not 200')
            continue
        loguru.logger.success(f'getProxiesFromGatherProxy: downloaded.')
        d = pyquery.PyQuery(response.text)
        scripts = list(d('table#tblproxy > script').items())
        loguru.logger.warning(f'getProxiesFromGatherProxy: scanning...')
        for script in scripts:
            script = script.text().strip()
            script = re.sub(r'^gp\.insertPrx\(', '', script)
            script = re.sub(r'\);$', '', script)
            script = json.loads(script)
            ip = script['PROXY_IP'].strip()
            port = int(script['PROXY_PORT'].strip(), 16)
            proxy = f'{ip}:{port}'
            proxies.append(proxy)
        loguru.logger.success(f'getProxiesFromGatherProxy: scanned.')
        loguru.logger.debug(f'getProxiesFromGatherProxy: {len(proxies)} proxies is found.')
        time.sleep(1)
    return proxies

def getProxiesFromFreeProxyList():
    proxies = []
    url = 'https://free-proxy-list.net/'
    loguru.logger.debug(f'getProxiesFromFreeProxyList: {url}')
    loguru.logger.warning(f'getProxiesFromFreeProxyList: downloading...')
    response = requests.get(url)
    if response.status_code != 200:
        loguru.logger.debug(f'getProxiesFromFreeProxyList: status code is not 200')
        return
    loguru.logger.success(f'getProxiesFromFreeProxyList: downloaded.')
    d = pyquery.PyQuery(response.text)
    trs = list(d('table#proxylisttable > tbody > tr').items())
    loguru.logger.warning(f'getProxiesFromFreeProxyList: scanning...')
    for tr in trs:
        tds = list(tr('td').items())
        ip = tds[0].text().strip()
        port = tds[1].text().strip()
        proxy = f'{ip}:{port}'
        proxies.append(proxy)
    loguru.logger.success(f'getProxiesFromFreeProxyList: scanned.')
    loguru.logger.debug(f'getProxiesFromFreeProxyList: {len(proxies)} proxies is found.')
    return proxies

def main():
    global proxy

    with open('tbrain_train_final_0610.csv', newline='') as csvfile:
        rows = csv.reader(csvfile)
        header = next(rows)
        for i in range(60):
            header = next(rows)
        temp = 0
        for row in rows:

            temp += 1
            print(temp)
            print(row[1])
#            try:
#             s = requests.session()
#             s.keep_alive = False
#             if proxy is None:
#                 proxy = getProxy()
#             proxyDict = {'https':f'https://{proxy}'}
#             print(proxyDict)
            
            response = requests.get(
                row[1]#, proxies=proxyDict)#,timeout = 3)
#                 ,
#                 proxies = proxyDict,
#                 timeout=10
#                proxies={'http':f'http://{proxy}','https':f'https://{proxy}'},
#                     proxies={
#                         'https': f'https://{proxy}'
#                     },
                )
            proxy = None
            if response.status_code != 200:
                loguru.logger.success(f'status code is not 200.')
                proxy = None
                break
            response.encoding = 'utf8'
            soup = BeautifulSoup(response.text, 'lxml')

            articles = soup.find_all('p')# or 
            articles_pre = soup.find_all('pre')
            titles = soup.find_all('h1')
            with open("./home/huangchingchi/training_set" + row[0] + ".txt","w") as f:
                #print(row[3])
                f.write(row[3])
                f.write('\n')
                if titles:
                    title = titles[-1].get_text()#max([i.get_text()for i in titles], key = len)
                    f.write(title)
                f.write('\n')
                for article in articles:
                    if len(article.get_text()) > 40 and article.get_text()[-1] in ['。', '？', '！', '」', '……']:
                        f.write(article.get_text().encode('utf8').decode('utf8'))
                for article in articles_pre:
                    if len(article.get_text()) > 20:# and article.get_text()[-1] in ['。', '？', '！', '」', '……']:
                        f.write(article.get_text().encode('utf8').decode('utf8'))
                        #print(article.get_text())
            if temp > 16:
                break
#             except requests.exceptions.ConnectionError:
#                 requests.status_code = "Connection refused"
#                 #time.sleep(5)
#                 continue
#             except requests.exceptions.ConnectionError:
#                 loguru.logger.error(f'getTaiexs: proxy({proxy}) is not working (connection error).')
#                 time.sleep(1)
#                 proxy = None
#                 continue
#             except requests.exceptions.ConnectTimeout:
#                 loguru.logger.error(f'getTaiexs: proxy({proxy}) is not working (connect timeout).')
#                 time.sleep(1)
#                 proxy = None
#                 continue
#             except requests.exceptions.ProxyError:
#                 loguru.logger.error(f'getTaiexs: proxy({proxy}) is not working (proxy error).')
#                 time.sleep(1)
#                 proxy = None
#                 continue
#             except requests.exceptions.SSLError:
#                 loguru.logger.error(f'getTaiexs: proxy({proxy}) is not working (ssl error).')
#                 time.sleep(1)
#                 proxy = None
#                 continue
#             except Exception as e:
#                 loguru.logger.error(f'getTaiexs: proxy({proxy}) is not working.')
#                 loguru.logger.error(e)
#                 time.sleep(1)
#                 proxy = None
#                 continue
                    
if __name__ == '__main__':
    loguru.logger.add(
        f'{datetime.date.today():%Y%m%d}.log',
        rotation='1 day',
        retention='7 days',
        level='DEBUG'
    )
    main()
# resp = requests.get("https://www.chinatimes.com/newspapers/20190827000246-260202?chdtv")
# soup = BeautifulSoup(resp.text, 'lxml')
# articles = soup.find_all('p')

# for article in articles:
#     if len(article.get_text()) > 30 and article.get_text()[-1] in ['。', '？', '！', '」', '……']:
#         print(article.get_text())