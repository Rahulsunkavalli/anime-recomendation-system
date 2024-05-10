from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences # type: ignore
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


anime_data = pd.read_csv("Anime.csv")


anime_data = anime_data.dropna(subset=['Description', 'Tags'])
anime_data = anime_data.drop_duplicates(subset=['Name'])
anime_data['Content_Warning'] = anime_data['Content_Warning'].fillna("N/A")


anime_data['Content'] = anime_data['Tags'] + ' ' + anime_data['Description'] + ' ' + anime_data['Content_Warning']

tokenizer = Tokenizer()
tokenizer.fit_on_texts(anime_data['Content'])

sequences = tokenizer.texts_to_sequences(anime_data['Content'])


max_len = max([len(x) for x in sequences])
sequences = pad_sequences(sequences, maxlen=max_len)


cosine_sim = cosine_similarity(sequences, sequences)


class AnimeName(BaseModel):
    anime_name: str


def get_recommendations(anime_name, cosine_sim=cosine_sim, anime_data=anime_data):
    anime_name = anime_name.lower()
    anime_names_lower = anime_data['Name'].str.lower()
    if anime_name not in anime_names_lower.values:
        raise HTTPException(status_code=404, detail="Anime not found")
    anime1_idx = anime_data.index[anime_names_lower == anime_name].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[anime1_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    anime_indices = [i[0] for i in sim_scores]
    recommendations = anime_data.iloc[anime_indices][['Name', 'Tags']]
    return recommendations.to_dict(orient='records')


@app.post("/predict")
async def predict(anime: AnimeName):
    return get_recommendations(anime.anime_name)


@app.options("/predict")
async def handle_options():
    return {}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
