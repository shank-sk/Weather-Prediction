# 🌦️ Seattle Weather Explorer – Predicting Weather Types Using Gaussian Naive Bayes

> Seattle Weather Explorer is a **Streamlit-based machine learning application** that predicts general weather types such as *Rainy*, *Sunny*, and *Foggy* using historical meteorological data from Seattle.  
It combines **data analysis**, **visualization**, and **interactive prediction** into a single, intuitive platform.

---

## 🚀 Key Features

- 📊 **Data Visualization** – Explore historical Seattle weather data with interactive graphs and trend analysis.  
- 🤖 **Weather Prediction** – Predict weather types using a trained **Gaussian Naive Bayes** classifier.  
- 🧠 **Model Insights** – View model metrics like Accuracy, Precision, Recall, and F1-score.  
- 💡 **Interactive Interface** – Make real-time predictions directly through the **Streamlit** web interface.  

---

## 🧱 System Architecture

Seattle Weather Explorer follows a modular machine learning workflow:

1. **Data Preprocessing** – Cleans, encodes, and normalizes weather data for model stability.  
2. **Exploratory Data Analysis (EDA)** – Visualizes distributions and correlations among features.  
3. **Model Training** – Implements and trains a **Gaussian Naive Bayes** classifier using `scikit-learn`.  
4. **Evaluation** – Assesses model performance using metrics such as Accuracy, Precision, Recall, and F1-score.  
5. **Streamlit Integration** – Provides an intuitive, web-based interface for interactive predictions and visualizations.  

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit  
- **Model**: Gaussian Naive Bayes (Scikit-learn)  
- **Data Handling**: Pandas, NumPy  
- **Visualization**: Matplotlib, Seaborn  
- **Environment**: Python 3.8+  

---

## ⚙️ Installation

# 1. Clone the repo
```bash
git clone https://github.com/your-username/seattle-weather-explorer.git
cd seattle-weather-explorer
```
# 2. (Optional) Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # on Windows
# or
source venv/bin/activate  # on macOS/Linux
```
# 3. Install dependencies
```bash
pip install -r requirements.txt
```
# 4. Run the Streamlit app
```bash
streamlit run app.py
```
## ▶️ Usage
1. Launch the app using streamlit run app.py.
2. Explore weather datasets through interactive plots and summary statistics.
3. Enter or upload new weather parameter values to predict the weather type.
4. View model accuracy and feature impact through evaluation visualizations.

## 📊 Results
- The Gaussian Naive Bayes model achieved strong classification accuracy across weather types.
- Visualization and performance metrics confirm consistent results and reliable predictions.
- Demonstrates that even a simple probabilistic model can yield meaningful environmental insights.

## 🚀 Future Enhancements
- Add additional features (humidity, air pressure, visibility).
- Implement advanced ML models like Random Forest, SVM, or LSTM for time-series forecasting.
- Integrate real-time weather data APIs (e.g., OpenWeatherMap).
- Deploy the app on Streamlit Cloud or Render for global access.

## 🙌 Credits 
Built with ❤️ by [Shashank Arya](https://github.com/shank-sk) 

Thanks to OpenAI, Streamlit, and the open-source community.
