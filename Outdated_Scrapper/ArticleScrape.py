from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

# Setup the driver. This one uses chrome with some options and a path to the chromedriver
driver = webdriver.Chrome(ChromeDriverManager().install())

# implicitly wait for 5 seconds
driver.implicitly_wait(5)

# load the webpage
driver.get("https://www.coindesk.com/business/2024/03/05/bitcoin-etf-giant-grayscale-introduces-a-crypto-staking-fund/")

# get all the paragraphs in the div class
paragraphs = driver.find_elements(By.CSS_SELECTOR, ".typography__StyledTypography-sc-owin6q-0.eycWal.at-text p")

# print each paragraph
for paragraph in paragraphs:
    print(paragraph.text)

# close the driver
driver.quit()
