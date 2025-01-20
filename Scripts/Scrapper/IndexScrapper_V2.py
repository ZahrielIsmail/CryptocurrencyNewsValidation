from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time

# Setup webdriver
s=Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=s)


website = "https://www.coindesk.com/tag/bitcoin/"
links = []

for i in range(1, 3):
	#url = f"https://www.coindesk.com/tag/bitcoin/{i}/"
	#url = f"https://www.coindesk.com/tag/ada/{i}/"
	#url = f"https://www.coindesk.com/tag/ethereum/{i}/"
	#url = f"https://www.coindesk.com/tag/shiba-inu/{i}/"
	driver.get(url)
	print(driver.title)

	elements = driver.find_elements(By.TAG_NAME, 'a')
	links.extend([element.get_attribute('href') for element in elements])

# Save the links to a DataFrame
df = pd.DataFrame(links, columns=['Links'])

urls_to_keep = [
    'https://www.coindesk.com/markets/',
    'https://www.coindesk.com/finance/',
    'https://www.coindesk.com/business/',
    'https://www.coindesk.com/news-analysis/',
    'https://www.coindesk.com/policy/',
    'https://www.coindesk.com/tech/'
]

# Filter the DataFrame to keep only the desired URLs
filtered_df = df[df['Links'].str.contains('|'.join(urls_to_keep), na=False)]

# Save the DataFrame to an Excel file
filtered_df.to_excel('links2.xlsx', index=False)

# Print a success message
print("The links were successfully saved to links2.xlsx")

driver.quit()