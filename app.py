import streamlit as st
import pandas as pd
from surprise import Dataset, Reader
from surprise import KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# Fungsi untuk memuat dataset
@st.cache_data
def load_data():
    df = pd.read_csv('IMDb_Top_250_Cleaned.csv')
    return df

# Fungsi untuk melatih model
@st.cache_data
def train_model(df):
    # Rename kolom agar sesuai dengan Surprise
    df_surprise = df.rename(columns={
        "Age Rating": "user_id",  # Age Rating menggantikan user_id
        "Title": "item_id",       # Title menggantikan item_id
        "IMDb Rating": "rating"   # IMDb Rating tetap sebagai rating
    })

    # Mengubah data ke format Surprise
    reader = Reader(rating_scale=(df_surprise["rating"].min(), df_surprise["rating"].max()))
    data = Dataset.load_from_df(df_surprise[["user_id", "item_id", "rating"]], reader)
    
    # Train-Test Split
    trainset, testset = train_test_split(data, test_size=0.25)
    
    # Model dengan Collaborative Filtering
    sim_options = {
        "name": "cosine",
        "user_based": True,  # Berbasis pengguna
    }
    algo = KNNBasic(sim_options=sim_options)
    algo.fit(trainset)
    
    # Evaluasi Model
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    
    return algo, rmse

# Fungsi untuk memberikan rekomendasi
def recommend(algo, user_id, movie_title):
    pred = algo.predict(uid=user_id, iid=movie_title)
    return pred.est

# Streamlit UI
st.title("IMDb Top 250 Movie Recommender")
st.write("Aplikasi ini memberikan rekomendasi berdasarkan age rating dan film yang Anda pilih.")

# Load data
df = load_data()

# Sidebar untuk memilih pengguna dan film
st.sidebar.header("Input Data")
user_id = st.sidebar.selectbox("Pilih Age Rating (User ID):", df["Age Rating"].unique())
movie_title = st.sidebar.selectbox("Pilih Film (Item ID):", df["Title"].unique())

# Train model
with st.spinner("Melatih model..."):
    algo, rmse = train_model(df)

st.success(f"Model telah dilatih! RMSE: {rmse:.2f}")

# Prediksi
if st.sidebar.button("Prediksi Rating"):
    with st.spinner("Memproses prediksi..."):
        predicted_rating = recommend(algo, user_id, movie_title)
    st.success(f"Prediksi rating untuk Age Rating '{user_id}' pada film '{movie_title}' adalah: {predicted_rating:.2f}")

# Tampilkan dataset
if st.checkbox("Tampilkan dataset"):
    st.write(df)
