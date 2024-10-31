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

# Initialize an empty list to store the links
links = []

# Get the webpage
driver.get('https://www.coindesk.com/search?s=cardano&sort=1')

# Find elements by tag name
elements = driver.find_elements(By.TAG_NAME, 'a')

# Get href attribute from elements
links.extend([element.get_attribute('href') for element in elements])

# Find the next page button and click it
button = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, 'Button__ButtonBase-sc-1sh00b8-0 Button__TextButton-sc-1sh00b8-1 gjtbfz ccioqw searchstyles__PageButton-sc-ci5zlg-17 dZKZem')]"))
)

counter = 0
while counter != 5:
	driver.execute_script("arguments[0].scrollIntoView();", button)
	driver.execute_script("arguments[0].click();", button)


	# Wait for the next page to load
	time.sleep(10)

	# Scrape the next page
	elements = driver.find_elements(By.TAG_NAME, 'a')
	links.extend([element.get_attribute('href') for element in elements])
	counter +=1

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
filtered_df.to_excel('links.xlsx', index=False)

# Print a success message
print("The links were successfully saved to links.xlsx")

# Close the driver
driver.quit()

