import requests
from bs4 import BeautifulSoup
import re

# Making a GET request
base_url = "https://bulletin.case.edu/course-descriptions/"
r = requests.get(base_url)

# check status code for response received
# success code - 200
print(r)

# Create the web scraper object
soup = BeautifulSoup(r.content, 'html.parser')


# Extract all links
def to_course_page(href):
    return href and re.compile("course-descriptions/....").search(href)


links = soup.find_all(href=to_course_page)

# Store all course codes so we can scrape each page individually
course_codes = []
for link in links:
    link = str(link)
    match = re.search("course-descriptions/(....)", link)
    if match:
        whole_link = match.group()
        course_code = match.group(1)
        course_codes.append(course_code)

print(course_codes)

