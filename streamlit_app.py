import streamlit as st
from streamlit_chat import message
from streamlit_lottie import st_lottie
import ast
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from ScreenNavigationGraph import ScreenNavigationGraph
from Logger import Logger
from datetime import datetime
import json
import requests
from config import Config
import os
import random
import re
import sqlite3
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

@dataclass
class ChatMessage:
    content: str
    timestamp: str
    is_user: bool = False

class NavigationState:
    def __init__(self):
        self.messages: List[ChatMessage] = []
        self.initialized_navigation: bool = False
        self.initialized_program: bool = False

    def add_message(self, content: str, is_user: bool = False):
        self.messages.append(
            ChatMessage(
                content=content,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                is_user=is_user
            )
        )

    def clear_history(self):
        self.messages.clear()
        self.initialized_navigation = False
        self.initialized_program = False


class UI:
    @staticmethod
    def set_page_config():
        st.set_page_config(
            page_title="Navigation Assistant",
            page_icon="🧭",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    @staticmethod
    def apply_custom_css():
        st.markdown("""
            <style>
            .main { padding: 0rem 1rem; }
            .stTextInput>div>div>input { border-radius: 10px; }
            .stSidebar .sidebar-content { background-color: #f0f2f6; }
            .css-1d391kg { padding: 1rem 1rem; }
            .stButton>button {
                border-radius: 10px;
                width: 100%;
            }
            .user-message, .assistant-message {
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
            }
            .user-message { background-color: #e6f3ff; }
            .assistant-message { background-color: #f0f2f6; }
            </style>
        """, unsafe_allow_html=True)

    @staticmethod
    @st.cache_resource
    def load_lottie_animation(url: str) -> Optional[Dict]:
        try:
            if os.path.exists(Config.ANIMATION_FILE):
                with open(Config.ANIMATION_FILE, "r") as f:
                    return json.load(f)
            
            response = requests.get(url)
            if response.status_code != 200:
                return None
                
            with open(Config.ANIMATION_FILE, "wb") as f:
                f.write(response.content)
            with open(Config.ANIMATION_FILE, "r") as f:
                return json.load(f)
                
        except Exception as e:
            Logger.write_log(f"Error loading Lottie animation: {str(e)}")
            return None

class NavigationAssistant:
    def __init__(self):
        self.df = pd.read_excel(Config.EXCEL_PATH)
        self.client = QdrantClient(Config.HOST)
        self.embeddings = self._get_embeddings()
        self.vector_store = self._initialize_vector_store()
        self.nav = self._initialize_navigation_graph()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = self._initialize_stop_words()
        self.greeting_patterns = self._load_greeting_patterns()

    @staticmethod
    @st.cache_resource
    def _get_embeddings():
        return HuggingFaceEmbeddings(
            model_name=Config.MODEL_NAME,
            cache_folder=Config.MODEL_CACHE,
            model_kwargs={"trust_remote_code": True}
        )

    def _initialize_vector_store(self) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self.client,
            collection_name=Config.COLLECTION_NAME,
            embedding=self.embeddings
        )

    def _initialize_navigation_graph(self) -> ScreenNavigationGraph:
        nav = ScreenNavigationGraph()
        screens = self.df['Screen Name'].tolist()
        for screen in screens:
            nav.add_screen(screen)
        
        for _, row in self.df.iterrows():
            nav.add_navigation(row['ParentPage'], row['Screen Name'])
        return nav

    def _load_greeting_patterns(self) -> Dict[str, List[str]]:
        return {
            'hello': ['Hi', 'Hello', 'Hey'],
            'greet': ['Hi there', 'Hello there', 'Hey there'],
            'welcome_back': ['Welcome back', 'Nice to see you again', 'Great to have you back'],
            'help': [
                "I'm here to help you navigate through the application. Where would you like to go?",
                "How can I assist you with navigation today?",
                "I can help you find any screen in the application. Just let me know where you want to go!"
            ]
        }

    def _initialize_stop_words(self) -> set:
        stop_words = set(stopwords.words('english'))
        custom_stop_words = {
            'open', 'take', 'go', 'want', 'need', 'help', 'show', 'display',
            'give', 'tell', 'provide', 'u', 'navigate', 'Please', 'can', 'you',
            'please', 'would', 'could', 'i', 'need', 'like', 'if', 'that',
            'works', 'for', 'me', 'your', 'way', 'over', 'on', 'make', 'find',
            'reach', 'direct', 'bring', 'guide', 'transport', 'shift', 'focus'
        }
        stop_words.update(custom_stop_words)
        stop_words.remove('down')
        stop_words.remove('am')
        return stop_words

    def get_random_greeting(self, greeting_type: str) -> str:
        return random.choice(self.greeting_patterns.get(greeting_type, ['Hello']))
        # return None

    def process_navigation_request(self, user_input: str) -> str:
        try:
            if self._is_greeting(user_input):
                return self._handle_greeting(user_input)
                
            lemmatized_input = self._lemmatize_text(user_input)
            Logger.write_log(f"Lemmatized input: {lemmatized_input}")
            
            results = self.vector_store.similarity_search(lemmatized_input, k=1)
            Logger.write_log(f"Search results: {results}")
            if not results:
                return "Sorry, I couldn't find the screen you're looking for."
                
            target_info = ast.literal_eval(results[0].page_content)
            target_screen = target_info['Screen Name']
            
            # Try paths from both home and admin screens
            path = self._get_navigation_path("AMGIOTHomeScreen", target_screen)
            if path:
                return self._format_navigation_response(path, "Home Screen")
                
            admin_path = self._get_navigation_path("AMGIOTSetup", target_screen)
            if admin_path:
                return self._format_navigation_response(
                    admin_path, 
                    "Admin Settings",
                    admin_required=True
                )
            
            return "Sorry, I couldn't find a path to the requested screen."
            
        except Exception as e:
            Logger.write_log(f"Error in navigation: {str(e)}")
            return f"An error occurred while processing your request: {str(e)}"

    def process_program_request(self, user_input: str) -> str:
        try:
            number_match = re.search(r'\d+', user_input)
            if not number_match:
                return "Please provide a program number."
            
            number = str(int(number_match.group()))
            
            with sqlite3.connect(Config.DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM Programs WHERE [Program Name] LIKE ?",
                    ('%' + number + '%',)
                )
                results = cursor.fetchall()
                
            return (f"Program {results[0][2]}: \n {results[0][1]}" if results 
                   else f"No program found with number {number}.")
                
        except Exception as e:
            Logger.write_log(f"Error in program request: {str(e)}")
            return f"An error occurred while processing your request: {str(e)}"

    def _is_greeting(self, text: str) -> bool:
        common_greetings = {'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening'}
        text_lower = text.lower().strip()
        return (text_lower in common_greetings or 
                any(greeting in text_lower for greeting in common_greetings))

    def _handle_greeting(self, text: str) -> str:
        greeting = self.get_random_greeting('greet')
        help_message = self.get_random_greeting('help')
        return f"{greeting}! 👋 {help_message}"

    def _get_navigation_path(self, start_screen: str, target_screen: str) -> Optional[List[str]]:
        path = self.nav.get_navigation_path(start_screen, target_screen)
        if path:
            return [self._get_user_screen_name(screen) for screen in path]
        return None

    def _get_user_screen_name(self, screen: str) -> str:
        screen_info = self.df[self.df['Screen Name'] == screen]
        return screen_info['User Screen Name'].iloc[0] if not screen_info.empty else screen
    def _lemmatize_text(self, text: str) -> str:
        words = text.split()
        lemmatized_words = [
            self.lemmatizer.lemmatize(word) 
            for word in words 
            if word.lower() not in self.stop_words
        ]
        lemmatized_text = ' '.join(lemmatized_words)
        return ''.join(c for c in lemmatized_text if c.isalnum() or c.isspace())

    def _format_navigation_response(self, path: List[str], start_point: str, admin_required: bool = False) -> str:
        response_parts = [
            "🎯 Found the screen you're looking for!",
            "\n⚠️ Note: This requires Admin Settings access." if admin_required else "",
            f"\n🧭 Navigation path from {start_point}:",
            " ➡️ ".join(path)
        ]
        return "\n".join(filter(None, response_parts))
def main():
    UI.set_page_config()
    UI.apply_custom_css()

    nav_assistant = NavigationAssistant()
    nav_state = NavigationState()

    # Initialize session state if needed
    if 'state' not in st.session_state:
        st.session_state['state'] = nav_state

    with st.sidebar:
        st.title("🧭 Navigation Assistant")

        lottie_json = UI.load_lottie_animation(Config.LOTTIE_URL)
        if lottie_json:
            st_lottie(lottie_json, height=200)

        st.markdown("---")
        page = st.radio("📱 Select Mode", ["Navigation Menu", "Program Recall"])

        if st.button("Clear Chat History"):
            st.session_state['state'].clear_history()
            st.rerun()

    # Main content
    if page == "Navigation Menu":
        st.title("🗺️ Navigation Assistant")
        if not st.session_state['state'].initialized_navigation:
            st.session_state['state'].clear_history()
            initial_greeting = (
                f"{nav_assistant.get_random_greeting('hello')}! 👋\n\n"
                "Welcome to the Navigation Assistant! I can help you find "
                "the best path to reach any screen in the application. "
                "Simply tell me where you want to go!"
            )
            st.session_state['state'].add_message(initial_greeting)
            st.session_state['state'].initialized_navigation = True
    else:  # Program Recall
        st.title("📝 Program Recall")
        if not st.session_state['state'].initialized_program:
            st.session_state['state'].clear_history()
            initial_greeting = (
                f"{nav_assistant.get_random_greeting('hello')}! 👋\n\n"
                "Welcome to the Program Recall! I can help you find "
                "the Program to load to Fanuc CNC Machine. "
                "Simply tell me which program you want to load!"
            )
            st.session_state['state'].add_message(initial_greeting)
            st.session_state['state'].initialized_program = True

    # Display chat messages
    for idx, msg in enumerate(st.session_state['state'].messages):
        message(
            msg.content,
            is_user=msg.is_user,
            key=f"msg_{idx}",
            avatar_style="adventurer" if msg.is_user else "bottts"
        )

    # User input handling
    prompt = ("Where would you like to go?" if page == "Navigation Menu"
             else "Which program would you like to load?")
    user_input = st.chat_input(prompt)

    if user_input:
        Logger.write_log(f"User input: {user_input}")
        st.session_state['state'].add_message(user_input, is_user=True)

        response = (nav_assistant.process_navigation_request(user_input)
                   if page == "Navigation Menu"
                   else nav_assistant.process_program_request(user_input))

        Logger.write_log(f"Assistant response: {response}")
        st.session_state['state'].add_message(response)
        st.rerun()

if __name__ == "__main__":
    main()
