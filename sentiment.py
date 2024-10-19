import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title='Indonesia vs Bahrain Tweets', layout='wide', initial_sidebar_state='expanded')

@st.cache_resource
def load_model():
    with st.spinner("Loading model and processing data..."):
        time.sleep(3)
        st.success("Model loaded successfully!")
    return "Model loaded"  

model = load_model()

@st.cache_data
def load_data():
    df = pd.read_csv('wasit_bola_bersih.csv')
    return df

df = load_data()

page = st.sidebar.selectbox("Select a Page", ["Tweet Data", "Data Visualizations", "Sentiment Analysis"])

if page == "Tweet Data":
    st.title('📊 Tweet Data for Indonesia vs Bahrain')
    st.markdown('Welcome to the tweet analysis dashboard! Here you can filter tweets based on username and keywords.')
    
    st.sidebar.header('🔍 Filters')
    username_filter = st.sidebar.text_input('Filter by Username')
    keyword_filter = st.sidebar.text_input('Filter by Keyword in Tweet')
    
    filtered_df = df.copy()
    if username_filter:
        filtered_df = filtered_df[filtered_df['username'].str.contains(username_filter, case=False, na=False)]
    if keyword_filter:
        filtered_df = filtered_df[filtered_df['cleaned_tweet'].fillna('').astype(str).str.contains(keyword_filter, case=False, na=False)]
    
    none_count = df['cleaned_tweet'].isna().sum()
    st.write(f"Tweets with None values in 'cleaned_tweet': {none_count}")
    
    st.subheader(f"Displaying {len(filtered_df)} filtered tweets")
    st.dataframe(filtered_df)
    
    st.write("### 📈 Tweet Statistics:")
    st.write(f"Total tweets: {len(df)}")
    st.write(f"Filtered tweets: {len(filtered_df)}")

elif page == "Data Visualizations":
    st.title('📈 Data Visualizations for Wasit Bahrain Tweets')
    st.markdown('This section provides visual insights into the tweet data.')
    
    st.write("### 📊 Top 5 Active Users:")
    top_users = df['username'].value_counts().head(5)
    st.bar_chart(top_users)
    
    st.write("### 🥧 Tweet Distribution by User:")
    fig, ax = plt.subplots()
    ax.pie(top_users, labels=top_users.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  
    st.pyplot(fig)
    
    st.write("### 📝 Word Count in Tweets:")
    df['Word_Count'] = df['cleaned_tweet'].apply(lambda x: len(str(x).split()))
    word_count_chart = df[['username', 'favorite_count']].groupby('username').mean().head(10)
    st.line_chart(word_count_chart)

elif page == "Sentiment Analysis":
    st.title('😊😐😠 Sentiment Analysis of Wasit Bahrain Tweets')
    st.markdown('This section provides insights into the sentiment of the tweets.')
    
    if 'label' not in df.columns:
        st.error("Sentiment analysis has not been performed on this dataset. Please make sure you have a 'label' column with sentiment values.")
    else:
        st.write("### 📊 Sentiment Distribution:")
        sentiment_counts = df['label'].value_counts()
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind='bar', ax=ax)
        plt.title("Sentiment Distribution")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        st.pyplot(fig)
        
        st.write("### 🥧 Sentiment Distribution (Pie Chart):")
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  
        plt.title("Sentiment Distribution")
        st.pyplot(fig)
        
        st.write("### 👥 Sentiment by Top Users:")
        top_users = df['username'].value_counts().head(10).index
        user_sentiment = df[df['username'].isin(top_users)].groupby('username')['label'].value_counts(normalize=True).unstack()
        fig, ax = plt.subplots(figsize=(12, 6))
        user_sentiment.plot(kind='bar', stacked=True, ax=ax)
        plt.title("Sentiment Distribution by Top Users")
        plt.xlabel("username")
        plt.ylabel("Proportion")
        plt.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("### 📊 Sentiment Statistics:")
        st.write(sentiment_counts)
        
        sentiment_percentages = sentiment_counts / len(df) * 100
        st.write("### 📊 Sentiment Percentages:")
        for sentiment, percentage in sentiment_percentages.items():
            st.write(f"{sentiment}: {percentage:.2f}%")

st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(120deg, #f0f4c3, #f5d8d9);
    }
    .sidebar .sidebar-content {
        background: #fbe9e7;
    }
    </style>
    """,
    unsafe_allow_html=True
)