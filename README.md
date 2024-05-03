# airbnb_price_prediction

![Airbnb_platform](Airbnb_platform.png)

## Overview

This project focuses on predicting the prices of Airbnb listings in the Lake Como region of Italy. The ultimate goal is to provide insights and predictions to potential hosts and travelers in the area, aiding them in decision-making processes related to property listing and rental.

This comprehensive project begins from data collection to the final prediction phase : 
## Objectives

1. **Web Scraping:** Gather data from the Airbnb website using web-scraping, specifically targeting listings in the Lake Como region. This involves collecting information such as listing details, amenities, prices, and location data, using Beautiful Soup and Selenium libraries. 

2. **Data Cleaning:** Process the scraped data to handle missing values, outliers, and inconsistencies. 

3. **Feature Engineering:** Create more advanced new features, like "accesibility to the city center" or transform existing ones to enhance the predictive power of the model.

4. **Machine Learning Modeling:** Develop predictive models using machine learning algorithms to estimate the prices of Airbnb listings, using scikit-learn.

5. **Streamlit Web Application:** Build an interactive web application using Streamlit framework to provide a user-friendly interface for exploring the project insights and predictions. Users can input their listing details and receive estimated prices, facilitating a seamless experience for hosts and travelers.

---

## Project Structure

- **Classification.ipynb:** Jupyter notebook containing classification analysis.
- **Data_Cleaning_Feature_Engineering.ipynb:** Jupyter notebook demonstrating data cleaning and feature engineering techniques.
- **Modeling.ipynb:** Jupyter notebook showcasing machine learning modeling techniques.
- **Webscraping_Airbnb_Final.ipynb:** Jupyter notebook detailing the web scraping process.
- **streamlit:** Directory containing files for the Streamlit web application.
  - **streamlit_final.py:** Main Python script for the Streamlit web application.

---

## Running the Streamlit App

To run the Streamlit web application, execute the following command:

```bash
streamlit run /streamlit/streamlit_final.py
```

The application will start, allowing users to explore the project insights and make predictions for Airbnb listings in the Lake Como region.

---

