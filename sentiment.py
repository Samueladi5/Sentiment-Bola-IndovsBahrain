import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
import hydralit_components as hc

st.set_page_config(page_title='Indonesia vs Bahrain Tweets Analysis', layout='wide', initial_sidebar_state='expanded')

# Define the menu data
menu_data = [
    {'icon': "🏠", 'label': "Home"},
    {'icon': "📊", 'label': "Tweet Data"},
    {'icon': "📈", 'label': "Data Visualizations"},
    {'icon': "😊", 'label': "Sentiment Analysis"},
    {'icon': "ℹ️", 'label': "About"}
]

# Configure the navigation bar
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme={
        'txc_inactive': '#FFFFFF',
        'menu_background': '#0178e4',
        'txc_active': '#FFFFFF',
        'option_active': '#000000'
    },
    sticky_nav=True,
    hide_streamlit_markers=False,
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        background: linear-gradient(120deg, #f8f9fa, #e9ecef);
    }
    .css-1d391kg {
        padding: 1rem;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with hc.HyLoader("Loading model...", loader_name=hc.Loaders.standard_loaders):
        time.sleep(3)
    return "Model loaded"

model = load_model()

@st.cache_data
def load_data():
    df = pd.read_csv('wasit_bola_bersih.csv')
    return df

df = load_data()

if menu_id == "Home":
    # Create a nice card-like container for personal info
    with st.container():
        st.markdown("""
            <div style='padding: 20px; border-radius: 10px; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h1 style='text-align: center; color: #0178e4;'>Welcome to Tweet Analysis Dashboard</h1>
                <hr>
                <h2 style='color: #333;'>Student Information</h2>
                <p style='font-size: 24px, font-color: #000000, font-family: Monospace;'>
                    <strong>Name:</strong> Samuel Adi Saut Puryanto<br>
                    <strong>Student ID:</strong> 21537141018<br>
                    <strong>University:</strong> Univeristas Negeri Yogyakarta<br>
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Add some statistics cards
    col1, col2, col3 = st.columns(3)
    with col1:
        hc.info_card(title='Total Tweets', content=str(len(df)), theme_override={'bgcolor': '#0178e4', 'title_color': 'white'})
    with col2:
        hc.info_card(title='Unique Users', content=str(df['username'].nunique()), theme_override={'bgcolor': '#00b4d8', 'title_color': 'white'})
    with col3:
        hc.info_card(title='Total Interactions', content=str(df['favorite_count'].sum()), theme_override={'bgcolor': '#03045e', 'title_color': 'white'})

elif menu_id == "Tweet Data":
    st.title('📊 Tweet Data for Indonesia vs Bahrain')
    st.markdown('Welcome to the tweet analysis dashboard! Here you can filter tweets based on username and keywords.')
    
    username_filter = st.text_input('Filter by Username')
    keyword_filter = st.text_input('Filter by Keyword in Tweet')
    
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

elif menu_id == "Data Visualizations":
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

elif menu_id == "Sentiment Analysis":
    st.title('😊 Sentiment Analysis')
    
    with hc.HyLoader('Loading Sentiment Analysis...', loader_name=hc.Loaders.pulse_bars):
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
            plt.xlabel("Username")
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

elif menu_id == "About":
    st.title('**Indonesia vs Bahrain Tweet Analysis Dashboard**')
    st.markdown('''
    ### Project Title: **Indonesia vs Bahrain Tweet Analysis Dashboard**
    
    #### Description:
    This project focuses on analyzing tweet data related to the Indonesia vs Bahrain football match, specifically looking at tweets that mention the performance of the referee ("wasit"). The project aims to provide insights through:
    
    1. **Tweet Data Exploration**: View and filter tweets based on usernames and keywords.
    2. **Data Visualizations**: Display visual insights like the most active users and tweet distributions.
    3. **Sentiment Analysis**: Analyze the sentiment of tweets (positive, neutral, negative) and show the distribution of sentiments across the dataset.
    
    #### Project Objectives:
    - To explore and filter tweets to gain meaningful insights.
    - To visualize key statistics and user interactions through charts and graphs.
    - To perform sentiment analysis on tweets to understand the public’s reaction to the referee’s performance.
    
    #### Tools Used:
    - **Python** for data processing.
    - **Streamlit** for building the interactive web application.
    - **Pandas** for data manipulation.
    - **Matplotlib** for data visualization.
    - **Hydralit Components** for enhancing the UI and navigation.
    
    #### Dataset:
    The tweet dataset (`wasit_bola_bersih.csv`) contains cleaned tweet data related to the Indonesia vs Bahrain match, including tweet text, usernames, and sentiment labels (if available).
    
    #### Future Enhancements:
    - Improving the sentiment analysis by integrating more robust models.
    - Expanding the data visualizations to show more detailed trends over time.
    ''')
    
    st.markdown("""
        <div style='padding: 20px; border-radius: 10px; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <h2 style='color: #0178e4;'>Project Overview</h2>
            <p style='font-size: 16px; line-height: 1.6;'>
                This project focuses on analyzing tweet data related to the Indonesia vs Bahrain football match, 
                specifically looking at tweets that mention the performance of the referee ("wasit"). 
                The project provides insights through:
                <ul>
                    <li>Tweet Data Exploration</li>
                    <li>Data Visualizations</li>
                    <li>Sentiment Analysis</li>
                    <li>User Engagement Patterns</li>
                </ul>
            </p>
            
            <h2 style='color: #0178e4; margin-top: 20px;'>Project Objectives</h2>
            <p style='font-size: 16px; line-height: 1.6;'>
                <ul>
                    <li>To explore and filter tweets to gain meaningful insights</li>
                    <li>To visualize key statistics and user interactions</li>
                    <li>To perform sentiment analysis on tweets</li>
                </ul>
            </p>
            
            <h2 style='color: #0178e4; margin-top: 20px;'>Technologies Used</h2>
            <p style='font-size: 16px; line-height: 1.6;'>
                <ul>
                    <li>Python</li>
                    <li>Streamlit</li>
                    <li>Pandas</li>
                    <li>Matplotlib</li>
                    <li>Hydralit Components</li>
                </ul>
            </p>
            
            <h2 style='color: #0178e4; margin-top: 20px;'>Future Enhancements</h2>
            <p style='font-size: 16px; line-height: 1.6;'>
                <ul>
                    <li>Improving the sentiment analysis with more robust models</li>
                    <li>Expanding the data visualizations to show detailed trends</li>
                    <li>Adding more interactive features and filters</li>
                </ul>
            </p>
        </div>
    """, unsafe_allow_html=True)

# Add a footer
st.markdown("""
    <div style='position: fixed; bottom: 0; width: 100%; background-color: #f8f9fa; padding: 10px; text-align: center;'>
        <p style='margin: 0;'>© 2024 Tweet Analysis Dashboard | Created with ❤️ using Streamlit</p>
    </div>
""", unsafe_allow_html=True)
