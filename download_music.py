from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from fake_useragent import UserAgent
import json

url = 'https://www.looperman.com/loops?page=7&keys=Boom%20Bap&cid=1&dir=d'
options = Options()
ua = UserAgent()
userAgent = ua.random
print(userAgent)
options.add_argument(f'user-agent={userAgent}')
chrome = webdriver.Chrome('/usr/local/bin/chromedriver', options=options)
chrome.maximize_window()

account = ""
password = ""
chrome.get(url)
with open("cookies.json",'r') as f:
    data = json.load(f)
for d in data:
    chrome.add_cookie(d)
chrome.refresh()


while True:
    for i in range(8, 124, 5):
        # elm = chrome.find_element(By.XPATH, '// *[ @ id = "body-left"] / div[{0}]'.format(str(i)))
        print(i)
        btn = chrome.find_element(By.XPATH, '// *[ @ id = "body-left"] / div[{0}]/div[2]/div[4]/a[3]'.format(str(i)))
        chrome.execute_script("arguments[0].click();", btn)
        btn = chrome.find_element(By.XPATH, '// *[ @ id = "body-left"] / div[1]/div[2]/div[4]/a[3]'.format(str(i)))
        chrome.execute_script("arguments[0].click();", btn)
        chrome.back()
    chrome.find_element(By.XPATH, '//*[@id="body-left"]/div[128]/div/a[last()-1]').click()

