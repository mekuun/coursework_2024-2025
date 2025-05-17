import pandas as pd

df = pd.read_csv("file2_with_topics.csv")
topics = pd.read_csv("alltopics_with_titles_gemini.csv")

id_to_title = dict(zip(topics["topic_id"], topics["Titles"]))

df["topic_title"] = df["topic"].map(id_to_title)

df.to_csv("file2_with_topic_titles.csv", index=False)
