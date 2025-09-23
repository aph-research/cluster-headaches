from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Ensures GUI is off
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Set up webdriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# Navigate to your Streamlit app
driver.get("https://cluster-headaches.streamlit.app/")

try:
    # Wait for a specific element to be present (adjust as needed)
    element = WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )
    print("Page is loaded. Title is: ", driver.title)
    
except Exception as e:
    print("An error occurred:", str(e))

finally:
    driver.quit()