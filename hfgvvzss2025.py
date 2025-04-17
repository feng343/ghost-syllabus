import requests
import json
import csv
import re
    
def scrapePage(target_url):
    headers = {
    "Accept-Language": "*",
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Version/15.1 Safari/537.36'
}
    req = requests.get(target_url, headers = headers)

    # check if request is a success
    if req.status_code != 200:
        print("error:", req.status_code)
        return []
                
    data = req.json() 

    results = []

    for t in data:
        
        if"title" in t and "idx" in t and "focus" in t and "en" in t["focus"] and "location" in t and "description" in t and "lecturer" in t and "start" in t and "finish" in t and "time" in t and "appointment_description" in t and "credit" in t and "contact" in t:
            title = t["title"]
            lecturer = ", ".join(t["lecturer"]) if t["lecturer"] else "N/A"
            idx = t["idx"]  
            focus = t["focus"]["en"][0] if t["focus"]["en"] else "N/A"
            location = t["location"]
            description =t["description"]
            start =t["start"]
            end =t["finish"]
            time =t["time"]
            appointment =t["appointment_description"]
            credit=t["credit"]["de"] if t["credit"]["de"] else "N/A"
            contact=t["contact"]

            results.append((idx, title, lecturer, focus, location, description, start, end, time, appointment, credit, contact))
 
    return results

s = ["SS2025"]
   
all_data = []
for semester in s:
    res = scrapePage(f"https://vvz.hfg-karlsruhe.de/api/v1/v/?f=&q=*&s={semester}&t=")
    all_data += res 

for idx, title, lecturer, focus, location, description, start, end, time, appointment, credit, contact in all_data:
    print(f"YEAR:\n{idx}\n")
    print(f"TITLE:\n{title}\n")
    print(f"lecturer:\n{lecturer}\n")
    print(f"MAJOR:\n{focus}\n")
    print(f"location:\n{location}\n")
    print(f"DESCRIPTION:\n{description}\n")
    print(f"Start:\n{start}\n")
    print(f"End:\n{end}\n")
    print(f"Time:\n{time}\n")
    print(f"appointment:\n{appointment}\n")
    print(f"Start:\n{credit}\n")
    print(f"End:\n{contact}\n")
    print("="*80)

print(f"found {len(all_data)} results")


# 导出到CSV文件
csv_file = "ss2025.csv"
fields = ["YEAR", "TITLE", "LECTURER", "MAJOR", "LOCATION", "DESCRIPTION", "START", "END", "TIME", "APPOINTMENT", "CREDIT", "CONTACT"]

def clean_text(text):
    return re.sub(r'\s+', ' ', str(text)).strip()

all_data = [(idx, clean_text(title), clean_text(lecturer), 
             clean_text(focus), clean_text(location), 
             clean_text(description), clean_text(start), 
             clean_text(end), clean_text(time), clean_text(appointment),
             clean_text(credit), clean_text(contact)) for idx, title, lecturer, 
             focus, location, description, start, end, time, appointment, credit, contact in all_data]

with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(fields)
    writer.writerows(all_data)

print(f"The data has been exported to  {csv_file}")