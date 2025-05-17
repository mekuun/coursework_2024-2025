import pandas as pd
import time
import google.generativeai as genai

genai.configure(api_key='AIzaSyA6YDQW8m6mozHWzINh0vfCgUk5P7luQaI')

model = genai.GenerativeModel('gemini-2.0-flash')

df = pd.read_csv("all_topics.csv")
alltopicslist = df['words'].to_list()
titles = []

for topic in alltopicslist:
    prompt = f"""The following words describe a topic in the field of particle physics.
Give a short and meaningful title in English, use 3-5 words.
IMPORTANT: Answer ONLY with the title, without quotations, without explanation.
If you can't generate a reasonable answer, just reply with a minus sign (-).

Words: {topic}
"""

    try:
        response = model.generate_content(prompt)
        title = response.text.strip()
        print(title)
        titles.append(title)
        
    except Exception as e:
        print(f"Request error: {e}")
        titles.append(None)

    time.sleep(4)

df['Titles'] = titles
df.to_csv("alltopics_with_titles_gemini.csv", index=False)
