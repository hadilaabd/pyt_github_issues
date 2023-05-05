import requests
import csv

# Set up the GitHub API endpoint and headers
endpoint = "https://api.github.com/search/issues?q=pytorch+repo:pytorch/pytorch+is:issue&per_page=100&page="
headers = {"Accept": "application/vnd.github.v3+json"}

# Initialize the CSV file and header row
with open("pytorch_issues.csv", mode="w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Title", "Body", "Tags"])

# Loop through the pages of search results
for page in range(1, 11):  # 10 pages of 100 results each
    response = requests.get(endpoint + str(page), headers=headers)

    # If the request was successful, extract the issues and save them to the CSV file
    if response.ok:
        data = response.json()
        with open("pytorch_issues.csv", mode="a", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, quoting=csv.QUOTE_NONE, delimiter='|', quotechar='',escapechar='\\')
            for issue in data["items"]:
                # Ignore issues with empty fields
                if issue["title"] and issue["body"] and issue["labels"]:
                    # Extract the title, body, and tags from the issue
                    title = issue["title"]
                    body = issue["body"]
                    tags = [label["name"] for label in issue["labels"]]

                    # Write the data to the CSV file
                    writer.writerow([title, body, tags])
    else:
        print(f"Error: {response.status_code}")