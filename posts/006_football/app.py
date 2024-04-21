import pickle
import pandas as pd
import streamlit as st

# Load MLflow model and scaler
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("rf_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load data
df = pd.read_csv("2022_2023_Football_Player_Stats.csv", sep=";", encoding="latin1")

# Positions
positions = {0: "Defender", 1: "Forward", 2: "Goalkeeper", 3: "Midfielder"}


# Helper function to preprocess input data
def preprocess_input(data):
    # Scale data
    data = scaler.transform(data)
    
    return data


# Function to predict using the model
def predict(data):
    # Preprocess input data
    X = preprocess_input(data)
    
    # Make predictions
    predictions = model.predict(X)
    
    return predictions


# Function to compare input with existing players
def compare(data, n):
    cols = ["Shots", "PasMedAtt", "Pas3rd", "Clr"]
    mse = ((df[["Shots", "PasMedAtt", "Pas3rd", "Clr"]] - data.iloc[0]) ** 2).mean(axis=1)
    
    idxs = mse.nsmallest(n).index
    return df.loc[idxs][["Player", "Age", "Pos", "Squad", "Comp"] + cols]


# Function to format a line of player information
def format_player(row):
    return f"*{row['Player']}*, a {row['Age']} year old {row['Pos']} playing for {row['Squad']} in {row['Comp']}."


# Function that provides text for players similar to input
def text_similar_players(data):
    data = data.reset_index()
    
    text = 'The players closest to your input are:\n'
    for index, row in data.iterrows():
        player_info = format_player(row)
        text += f"{index + 1}. {player_info}\n"
    
    return text


# Streamlit UI
def main():
    st.set_page_config(page_title="Classifying Footballers", layout="wide")
    st.markdown("""
        <style>
            .reportview-container {
                margin-top: -2em;
            }
            #MainMenu {visibility: hidden;}
            .stDeployButton {display:none;}
            footer {visibility: hidden;}
            #stDecoration {display:none;}
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Predicting Football player positions based on 2022/23 stats")
    
    st.sidebar.success(f"Visit [blog.panliyong.nl](https://blog.panliyong.nl/posts/006_football) for the post!")
    
    # Create input form
    st.sidebar.header("Input Features")
    shots = st.sidebar.number_input("Shots", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
    pas_med_att = st.sidebar.number_input("Passes Medium Att", min_value=0.0, max_value=70.0, step=1.0, value=20.0)
    pas_3rd = st.sidebar.number_input("3rd Passes", min_value=0.0, max_value=20.0, step=1.0, value=3.0)
    clr = st.sidebar.number_input("Clearances", min_value=0.0, max_value=15.0, step=0.1, value=3.0)
    
    # Create column layout
    left_column, right_column = st.columns(2)
    
    with left_column:
        # Players compare
        st.markdown("### Find similar players")
        n_players = st.slider("How many similar players do you want to find?", min_value=1, max_value=20, step=1,
                              value=5)
    
    # Create prediction button
    if st.sidebar.button("Predict"):
        # Create a dataframe from the input
        input_data = pd.DataFrame({
            "Shots": [shots],
            "PasMedAtt": [pas_med_att],
            "Pas3rd": [pas_3rd],
            "Clr": [clr]
        })
        
        # Predict
        prediction = predict(input_data)
        
        # Display prediction
        position = positions.get(prediction[0])
        position_styled = f"<span style='color:purple'>{position}</span>"
        
        st.sidebar.info(f"Predicted Position: {position}")
        
        # Check which player is closest to input
        players = compare(input_data, n=n_players)
        
        text = text_similar_players(players)
        
        with left_column:
            st.info(text)
        
        with right_column:
            st.markdown(f"# **Predicted a** {position_styled}",
                        unsafe_allow_html=True)
            st.markdown("## Data of similar players:")
            st.dataframe(players.reset_index(drop=True))


# Run the app
if __name__ == "__main__":
    main()
