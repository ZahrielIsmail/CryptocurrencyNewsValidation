from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time
import openpyxl
from openpyxl import load_workbook



file_name = 'Content.xlsx'
tab_name = 'Bitcoin_V2'


rows = []
counter = 1
df_to_append = pd.DataFrame(columns=["Website", "Content"])
df = pd.read_excel('Index.xlsx', sheet_name=tab_name)
df = df.drop_duplicates(subset=df.columns[0])
df = df[df['Links'].str.count('/') > 4]


chrome_options = Options()
chrome_options.add_argument("--ignore-certificate-errors")
chrome_options.add_argument('--ignore-ssl-errors')



s=Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=s, options=chrome_options)

for website in df.iloc[:,0]:
	# load the webpage
	print("Index",counter,"Working on this website:",website)
	driver.get(website)
	
	print("------ Preventing rate limit, Sleep 5 seconds ------")
	time.sleep(5)

	# get all the paragraphs in the div class
	paragraphs = driver.find_elements(By.CSS_SELECTOR, ".typography__StyledTypography-sc-owin6q-0.eycWal.at-text p")
	headers = driver.find_elements(By.TAG_NAME, "h1")
	subheaders = driver.find_elements(By.TAG_NAME, "h2")


	all_paragraphs = ''
	row = ''

	all_paragraphs = ' '.join([paragraph.text for paragraph in paragraphs])
	all_header = ' '.join([headers.text for headers in headers])
	all_subheaders = ' '.join([subheaders.text for subheaders in subheaders])

	row = {"Website":website,"Header":all_header,"SubHeader":all_subheaders,"Content":all_paragraphs}
	rows.append(row)
	counter+=1
	if counter%50 ==0:
		print("------ Preventing rate limit, Sleep 30 seconds ------")
		time.sleep(30)
	
df_to_append = pd.concat([df_to_append, pd.DataFrame(rows)], ignore_index=True)
df_to_append.to_excel('Contents.xlsx', index=False)
# close the driver
driver.quit()