import urllib.parse
import time
import os
import csv
import string
import re
from collections import Counter, defaultdict
import datetime
import json
from bs4 import BeautifulSoup
import feedparser
from textblob import TextBlob
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

print("Loading Advanced NLP Models...")
nlp = spacy.load("en_core_web_sm")
noise_words = {"strait", "hormuz", "iran", "us", "u.s.", "said", "says", "new", "news", "will", "year", "one", "also", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "reuters", "cnn", "jazeera", "times", "post", "bbc", "cnbc", "bloomberg", "guardian", "fox", "ap", "press", "daily", "sabah", "hill", "cbs", "nbc", "briefing", "today", "update", "live", "video", "watch"}
stopwords = nlp.Defaults.stop_words.union(noise_words)
entity_blacklist = {"reuters", "cnn", "al jazeera", "the new york times", "cbs news", "hill", "cnbc", "bbc", "fox news", "the jerusalem post", "daily sabah", "nbc news", "bloomberg", "the guardian", "associated press", "news", "times", "post", "ap", "al", "jazeera", "washington post", "wall street journal"}

# Narrative framing lexicons
fear_keywords = {"fear", "panic", "threat", "crisis", "warning", "escalation", "danger", "risk", "conflict", "tension", "attack", "strike", "war"}
deescalation_keywords = {"peace", "calm", "talks", "diplomacy", "negotiation", "ceasefire", "truce", "agreement", "resolution", "pact"}

# Geocoding Dictionary (Static approximations for top players)
geo_dict = {
    "Iran": [32.4279, 53.6880], "United States": [37.0902, -95.7129], "Israel": [31.0461, 34.8516], 
    "Yemen": [15.5527, 48.5164], "Saudi Arabia": [23.8859, 45.0792], "Uae": [23.4241, 53.8478],
    "United Arab Emirates": [23.4241, 53.8478], "China": [35.8617, 104.1954], "Russia": [61.5240, 105.3188],
    "Qatar": [25.3548, 51.1839], "Oman": [21.5126, 55.9233], "Iraq": [33.2232, 43.6793], "Syria": [34.8021, 38.9968],
    "Uk": [55.3781, -3.4360], "United Kingdom": [55.3781, -3.4360], "Tehran": [35.6892, 51.3890], "Washington": [38.9072, -77.0369]
}

SEARCH_QUERY = '"Hormuz"'
start_date = datetime.date(2026, 2, 27)
end_date = datetime.date.today()
delta = datetime.timedelta(days=1)

articles = []
all_text_corpus = []

# Network Graph
co_occurrence = defaultdict(int)
source_stats = defaultdict(lambda: {"count": 0, "sentiment_sum": 0, "fear_sum": 0})
quote_sentiment_sum = 0
quote_count = 0
text_sentiment_sum = 0

themes = {"Military": 0, "Economic": 0, "Diplomatic": 0, "Maritime": 0}
military_keywords = {"missile", "navy", "strike", "military", "war", "attack", "fleet", "drone", "guard", "ship", "vessel"}
economic_keywords = {"oil", "barrel", "trade", "economy", "price", "market", "export", "sanctions", "cargo"}
diplomatic_keywords = {"talks", "summit", "un", "diplomatic", "negotiation", "treaty", "ambassador", "peace"}
maritime_keywords = {"strait", "chokepoint", "route", "navigation", "sea", "gulf", "waters"}

persons = Counter()
organizations = Counter()
locations = Counter()
geospatial_hits = Counter()
word_counter = Counter()
bigrams_counter = Counter()
sentiment_summary = {"Positive": 0, "Negative": 0, "Neutral": 0}
framing = {"Fear & Escalation": 0, "Diplomacy & Calm": 0}

trajectory_data = defaultdict(lambda: {"volume": 0, "sentiment_sum": 0, "fear_count": 0})
subjectivity_sum = 0

current_date = start_date
print(f"Commencing Deep OSINT Scraping from {start_date} to {end_date}...")

while current_date <= end_date:
    d1 = current_date.strftime('%Y-%m-%d')
    d2 = (current_date + delta).strftime('%Y-%m-%d')
    
    q = urllib.parse.quote(f'{SEARCH_QUERY} after:{d1} before:{d2}')
    RSS_URL = f'https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en'
    
    try:
        feed = feedparser.parse(RSS_URL)
    except Exception as e:
        current_date += delta
        continue
        
    for entry in feed.entries:
        title = entry.title.split(" - ")[0] # Strip news agency from title
        source_name = entry.source.title if hasattr(entry, 'source') else "Unknown"
        pub_date_str = entry.published
        
        if any(a['title'] == title for a in articles): continue
            
        soup = BeautifulSoup(entry.description, "html.parser")
        desc_text = soup.get_text(separator=' ')
        full_text = f"{title}. {desc_text}"
        all_text_corpus.append(full_text)
        
        # 4. Quote Extraction
        quotes = re.findall(r'"([^"]*)"', full_text)
        for q_str in quotes:
            if len(q_str.split()) > 3:
                quote_sentiment_sum += TextBlob(q_str).sentiment.polarity
                quote_count += 1

        # Sentiment
        blob = TextBlob(full_text)
        sentiment_score = blob.sentiment.polarity
        text_sentiment_sum += sentiment_score
        subjectivity_sum += blob.sentiment.subjectivity
        
        if sentiment_score > 0.1: sentiment_summary["Positive"] += 1
        elif sentiment_score < -0.1: sentiment_summary["Negative"] += 1
        else: sentiment_summary["Neutral"] += 1
        
        # Media Bias
        source_stats[source_name]["count"] += 1
        source_stats[source_name]["sentiment_sum"] += sentiment_score
        
        # Framing & Themes
        words_list = full_text.lower().translate(str.maketrans('', '', string.punctuation)).split()
        words_set = set(words_list)
        fear_hits = len(words_set.intersection(fear_keywords))
        dip_hits = len(words_set.intersection(deescalation_keywords))
        
        if fear_hits >= dip_hits: framing["Fear & Escalation"] += 1
        else: framing["Diplomacy & Calm"] += 1
        
        # Theme assignment
        m_hits = len(words_set.intersection(military_keywords))
        e_hits = len(words_set.intersection(economic_keywords))
        d_hits = len(words_set.intersection(diplomatic_keywords))
        ma_hits = len(words_set.intersection(maritime_keywords))
        
        theme_scores = {"Military": m_hits, "Economic": e_hits, "Diplomatic": d_hits, "Maritime": ma_hits}
        dom_theme = max(theme_scores, key=theme_scores.get)
        if theme_scores[dom_theme] > 0:
            themes[dom_theme] += 1
        
        clean_words = [w for w in words_list if w not in stopwords and len(w) > 3]
        for w in clean_words:
            word_counter[w] += 1
            
        for i in range(len(clean_words)-1):
            bg = f"{clean_words[i]} {clean_words[i+1]}"
            bigrams_counter[bg] += 1

        source_stats[source_name]["fear_sum"] += fear_hits
        
        # Trajectory
        trajectory_data[d1]["volume"] += 1
        trajectory_data[d1]["sentiment_sum"] += sentiment_score
        trajectory_data[d1]["fear_count"] += fear_hits
        
        # NER & Network
        doc = nlp(full_text)
        ents_in_article = set()
        
        for ent in doc.ents:
            clean_text = ent.text.strip().title()
            if len(clean_text) < 3 or clean_text.lower() in stopwords: continue
            if any(char.isdigit() for char in clean_text): continue
            clean_lower = clean_text.lower()
            if "strait" in clean_lower or "hormuz" in clean_lower or "iran" == clean_lower: continue
            if clean_lower in entity_blacklist or any(noise in clean_lower for noise in ["news", "times", "post", "daily", "journal", "telegraph"]): continue
            
            valid_ent = False
            
            # Fix Trump classification
            if "trump" in clean_lower:
                persons["Donald Trump"] += 1
                valid_ent = True
            elif ent.label_ == "PERSON" and " " in clean_text and not any(w in clean_lower for w in ["admin", "gov", "news"]):
                persons[clean_text] += 1
                valid_ent = True
            elif ent.label_ == "ORG" and len(clean_text.split()) < 4:
                organizations[clean_text] += 1
                valid_ent = True
            elif ent.label_ == "GPE":
                locations[clean_text] += 1
                valid_ent = True
                # Geospatial Mapping
                for key in geo_dict.keys():
                    if key.lower() in clean_lower:
                        geospatial_hits[key] += 1
            
            if valid_ent: ents_in_article.add(clean_text)
        
        # 1. Co-occurrence Network
        ents_list = list(ents_in_article)
        for i in range(len(ents_list)):
            for j in range(i+1, len(ents_list)):
                pair = tuple(sorted([ents_list[i], ents_list[j]]))
                co_occurrence[pair] += 1

        articles.append({"title": title, "date": pub_date_str, "sentiment_score": sentiment_score})
        
    current_date += delta

print("Performing Topic Modeling (LDA/KMeans)...")
# 2. Topic Modeling (KMeans on TF-IDF)
topics = []
try:
    vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(all_text_corpus)
    k = min(4, X.shape[0]) if X.shape[0] > 0 else 4
    if k > 0:
        model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
        model.fit(X)
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        for i in range(k):
            top_words = [terms[ind] for ind in order_centroids[i, :7]]
            topics.append({"cluster": i, "keywords": top_words})
except ValueError:
    print("Warning: Vocabulary empty or rate limited. Using fallback topics.")
    topics = [
        {"cluster": 0, "keywords": ["military", "strike", "iran", "navy", "drone"]},
        {"cluster": 1, "keywords": ["oil", "price", "economy", "market", "trade"]},
        {"cluster": 2, "keywords": ["diplomacy", "talks", "peace", "un", "treaty"]},
        {"cluster": 3, "keywords": ["strait", "ship", "vessel", "cargo", "route"]}
    ]

print("Calculating Analytics...")

# Trajectory & 6. Predictive Escalation Index
timeline_series = []
for d_str, stats in sorted(trajectory_data.items()):
    vol = stats["volume"]
    avg_sent = stats["sentiment_sum"] / vol if vol > 0 else 0
    # Escalation Index: Volume + Fear - Sentiment (normalized roughly 0-100)
    escalation_index = min(100, max(0, (vol * 0.5) + (stats["fear_count"] * 2) - (avg_sent * 50)))
    timeline_series.append({
        "date": d_str, "volume": vol, "avg_sentiment": round(avg_sent, 2),
        "fear_count": stats["fear_count"], "escalation_index": round(escalation_index, 1)
    })

# Media Bias
top_sources = []
for src, stats in sorted(source_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:10]:
    if src == "Unknown": continue
    avg_sent = stats["sentiment_sum"] / stats["count"]
    avg_fear = stats["fear_sum"] / stats["count"]
    top_sources.append({"source": src, "count": stats["count"], "avg_sentiment": round(avg_sent, 2), "avg_fear": round(avg_fear, 2)})

# Network Graph Data
network_edges = [{"source": pair[0], "target": pair[1], "weight": count} for pair, count in sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)[:50]]

# Quotes
avg_quote_sentiment = quote_sentiment_sum / quote_count if quote_count > 0 else 0
avg_text_sentiment = text_sentiment_sum / len(articles) if articles else 0
avg_subj = subjectivity_sum / len(articles) if articles else 0

# Geospatial
geo_data = [{"name": k, "lat": geo_dict[k][0], "lon": geo_dict[k][1], "count": v} for k, v in geospatial_hits.items()]

dashboard_data = {
    "metadata": {
        "last_updated": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "total_articles": len(articles),
        "date_range": f"{start_date.strftime('%b %d, %Y')} to {end_date.strftime('%b %d, %Y')}"
    },
    "timeline": timeline_series,
    "themes": themes,
    "sentiment_summary": sentiment_summary,
    "framing": framing,
    "avg_subjectivity": round((avg_subj * 100), 1),
    "top_words": [{"text": w[0], "value": w[1]} for w in word_counter.most_common(20)],
    "top_bigrams": [{"phrase": b[0].title(), "count": b[1]} for b in bigrams_counter.most_common(5)],
    "top_orgs": [{"name": o[0], "count": o[1]} for o in organizations.most_common(10)],
    "top_locations": [{"name": l[0], "count": l[1]} for l in locations.most_common(10)],
    "network_edges": network_edges,
    "topics": topics,
    "media_bias": top_sources,
    "quote_analysis": {"journalist_sentiment": round(avg_text_sentiment, 2), "quote_sentiment": round(avg_quote_sentiment, 2)},
    "geospatial": geo_data,
    "top_persons": [{"name": p[0], "count": p[1]} for p in persons.most_common(10)]
}

output_js_path = 'hormuz_data.js'
with open(output_js_path, 'w', encoding='utf-8') as f:
    f.write('const HORMUZ_DATA = ')
    json.dump(dashboard_data, f, ensure_ascii=False, indent=4)
    f.write(';')

print(f"Deep Analysis Complete! Saved to {output_js_path}")

output_csv_path = 'hormuz_articles.csv'
# Generate CSV
with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        "Date", "Source", "Headline", "Sentiment_Score", "Sentiment_Label"
    ])
    for a in articles:
        sent_label = "Negative" if a['sentiment_score'] < -0.2 else ("Positive" if a['sentiment_score'] > 0.2 else "Neutral")
        writer.writerow([
            a['date'], a.get('source', 'Unknown'), a['title'], 
            round(a['sentiment_score'], 2), sent_label
        ])
print(f"CSV Export Complete! Saved to {output_csv_path}")
