import requests
import os

urls_to_scrape = [
    "https://www.ashoka.org/es-es/country/india",
    "https://www.ashoka.org/en-ng/program/nominate-ashoka-young-changemaker",
    "https://www.ashoka.org/en-ng/about-ashoka",
    "https://en.wikipedia.org/wiki/Ashoka_(non-profit_organization)",
    "https://www.ashoka.org/en-ca/program/ashoka-changemakers",
    "https://www.ashoka.org/de-de/collection/stories-india?page=1",
    "https://www.linkedin.com/company/ashoka/posts/?feedView=all",
    "https://www.linkedin.com/company/ashokayoungchangemakers",
]

output_path = os.path.join("data", "ashoka_web.html")

def fetch_all_to_file(urls, path):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # --- Step 2: Loop through each URL ---
    for i, url in enumerate(urls):
        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()  

            write_mode = "w" if i == 0 else "a"
            
            with open(path, write_mode, encoding="utf-8") as f:
                if i > 0:
                    f.write("\n\n<!-- CONTENT FROM NEW PAGE -->\n\n")
                f.write(r.text)
            print(f"Successfully saved content from {url} to {path}")

        except requests.exceptions.RequestException as e:
            print(f"Could not fetch {url}. Error: {e}")

# --- Step 4: Run the function ---
fetch_all_to_file(urls_to_scrape, output_path)

print("\nAll websites have been scraped.")




