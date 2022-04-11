import requests
from bs4 import BeautifulSoup
import re
import os


# Extract all links that are course descriptions (soup search function)
def to_course_page(href):
    return href and re.compile("course-descriptions/....").search(href)


# Making a GET request
base_url = "https://bulletin.case.edu/course-descriptions/"
r = requests.get(base_url)

# check status code for response received
# success code - 200
print(r)

# Create the web scraper object
soup = BeautifulSoup(r.content, 'html.parser')

links = soup.find_all(href=to_course_page)

# Store all course codes so we can scrape each page individually
course_codes = []
for link in links:
    link = str(link)
    match = re.search("course-descriptions/(....)/", link)

    if match:
        whole_link = match.group()
        course_code = match.group(1)
        course_codes.append(course_code)

print(course_codes)


def scrape_course_page(base_url, course):
    r = requests.get(base_url + course)

    soup = BeautifulSoup(r.content, 'html.parser')

    titles = soup.find_all(class_="courseblocktitle")
    contents = soup.find_all(class_="courseblockdesc")

    ret = ""
    for title, content in zip(titles, contents):
        ret += title.get_text() + content.get_text()

    return ret


if not os.path.isdir("input"):
    os.mkdir("input")

for course in course_codes:
    print("Scraping " + course)
    text = scrape_course_page(base_url, course)

    with open("input/" + course + ".txt", 'w') as f:
        f.write(text)
