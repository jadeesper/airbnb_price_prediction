#!/usr/bin/env python
# coding: utf-8



import streamlit as st
import pandas as pd
from sklearn import datasets
import plotly.express as px

from streamlit_option_menu import option_menu

df = pd.read_csv('Dataframefinal_clean.csv') #importing the dataframe

selected = option_menu(menu_title = "Menu", options = ["Building the model", "Price simulator"], default_index = 0, orientation = "horizontal", menu_icon = "house")

if selected == "Building the model":
    st.title(f"You have selected {selected}")
     #   st.set_page_config("Predicting Airbnb's housing prices in the city of Como")
    st.title("Predicting Airbnb Housing Prices in the city of Como")

    st.subheader("Webscraping the Airbnb Website")

    st.markdown('We directly collect data from the Airbnb Website.')

    st.markdown('''
      The objective is to collect as much data as possible for our final model: the first problem we encounter is Airbnb's restriction of amount of results to 300. 
      We therefore decide to carry out several searches: one per month, for four summer months. This will allow us to:\n
      - Collect a larger amount of observations (1200), giving us an exploitable dataframe for further steps\n
      - Capture price differences related to peak season (most likely july and august) versus low season (most likely june and september)\n''')

    st.markdown('''
        In our general result pages, we observe a certain amount of features per listing. However, precise information isn't available. 
    \n''')

    st.image("./general_page_airbnb_new.jpg")

    st.markdown('''
       We therefore need to collect all 1200 urls (corresponding to our listings' list) and scrape each one individually.\n
       Here is an example of how a typcial listing's url looks like:\n''')

    st.image("./first_page_new.jpg")

    st.image("./second_page_new.jpg")

    st.markdown('''
    Several features, notably the amenities, the price (without tax) or the location's rating will be particularly useful for the rest of our modelling process\n''')



    # -*- coding: utf-8 -*-

#!/usr/bin/env python
# coding: utf-8

# In[1]:


    import streamlit as st

    st.title("Data Cleaning")

    st.markdown("We now have raw data coming from the Airbnb website, and we therefore need to clean it")
    st.markdown("We had to : ")
    st.write("- Get rid of missing values (replacing them by 0 or median value) ")
    st.write("- Handle outliers")

    fig = px.histogram(df, x="Prix par nuit", title="Distribution of prices per night")
    st.plotly_chart(fig)

    st.write("- Get rid of duplicates")
    st.write("- Transform Strings into numerical values ")

    st.write(" For example : Handling the **_amenities_** column")

    import pandas as pd

    df = pd.read_csv("june_emp.csv")
    dfA = pd.read_csv("july_emp.csv")
    dfB = pd.read_csv("august_emp.csv")
    dfC = pd.read_csv("sept_emp.csv")


    #j'arrive pas à prendre que amenities
    df1 = df['amenities'].head()

    df_entier = pd.concat([df,dfA,dfB,dfC], ignore_index=True)

    #' An example of the data cleaning we had to is in the **_amenities_** column : we had to separate each amenities of each house in a different column based on what the column contained while being careful because there is a part of a column that also contains what is **_NOT_** in the house'

    df["Vue sur le jardin"] = df["amenities"].str.contains("Vue sur le jardin")
    df["Vue sur la vallée"] = df["amenities"].str.contains("Vue sur la vall")
    df["Vue sur la montagne"] = df["amenities"].str.contains("Vue sur la montagne")
    df2 = df[["Vue sur le jardin", "Vue sur la vallée", "Vue sur la montagne"]].head(5)


    st.dataframe(df1, 2000, 1000)
    st.dataframe(df2, 2000, 1000)  # Same as st.write(df)
    st.caption("Using a command, we transform each amenity into a boolean, creating an 'amenities' dataframe ")

    st.title("Feature Engineering")

    data = pd.get_dummies(df2, columns = ['Vue sur le jardin',
                                         'Vue sur la vallée',
                                         'Vue sur la montagne'], drop_first = True)
    st.markdown("**_One Hot Encoding_**")
    st.dataframe(data)
    st.caption("There is an example on how we handled categorical values : once they were in a boolean shape, we gave them **_binary values_** ")

    st.markdown("**_Creation of the season feature_**")

    df_entier.loc[df_entier['date'].str.contains('juin') | df_entier['date'].str.contains('sept'), 'Saison'] = 0
    df_entier.loc[df_entier['date'].str.contains('juil'), 'Saison'] = 1
    df_entier.loc[df_entier['date'].str.contains('août'), 'Saison'] = 2

    from sklearn.utils import shuffle
    df_entier2 = shuffle(df_entier)

    df_entier1 = df_entier2[["date", "Saison"]].head(5)
    st.dataframe(df_entier1)
    st.caption("Once it's done, each house take a specific value between 0 and 2, corresponding to the amount of activiy associated with the rental's period")

    st.markdown("**_Creation of a housing type feature_**")

    dfE = pd.read_csv("Dataframefinal_clean.csv")
    dfE1 = dfE['Logement codé'].value_counts()
    dfE2 = dfE[['name', 'Logement codé']].head(5)
    st.dataframe(dfE2)
    st.caption("In this case, we have created a housing scoring system based on the type of housing column that sets each observation in a class going from 0 to 6. If the type of housing is expensive, it goes into a higher class.")

    st.markdown("**_Creation of a location score_**")

    dfF = pd.read_csv("location_feature.csv")

    df40 = dfF[['Localisation2','Emplacement', 'Prix par nuit', 'Distance in km', 'Score' ]].head()
    st.dataframe(df40, 2000, 1000)

    st.write("We want to create a score based on location (each city surrounding Como gets a score). In order to build it, we use three characteristics:")
    st.write("- the average rating per location")
    st.write("- the average price per location")
    st.write("- the location's distance to the epicenter of activity")
    st.write("After MinMax Scaling these variables, we assemble them to deduct our final feature: the location's score")

    st.title('Data visualisation') #title of the subpage
    
    st.dataframe(df.head())
    
    st.write('Our final DataFrame looks like this:')
    
    
    dfE['Logement codé'] = dfE['Logement codé'].astype(str)
    dfE.loc[dfE['Logement codé'].str.contains('0'), 'Housing type'] = 'Hostel'
    dfE.loc[dfE['Logement codé'].str.contains('1'), 'Housing type'] = 'Private bedroom'
    dfE.loc[dfE['Logement codé'].str.contains('2'), 'Housing type'] = 'House'
    dfE.loc[dfE['Logement codé'].str.contains('3'), 'Housing type'] = 'High-end shared housing'
    dfE.loc[dfE['Logement codé'].str.contains('4'), 'Housing type'] = 'Appartment'
    dfE.loc[dfE['Logement codé'].str.contains('5'), 'Housing type'] = 'Entry-level housing'
    dfE.loc[dfE['Logement codé'].str.contains('6'), 'Housing type'] = 'High-end housing'

    fig2 = px.histogram(dfE, x="Prix par nuit", color="Saison", marginal="rug",title="Distributions of prices per night, per season",
                           hover_data=dfE.columns)
    st.plotly_chart(fig2)

    fig3 = px.histogram(dfE, x="Housing type", title="Distribution of housing types")
    st.plotly_chart(fig3)



    import seaborn as sns
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import streamlit as st

    df = pd.read_csv('Dataframefinal_clean.csv')
    st.title('Modeling')
    st.header('Regression')

    st.subheader('Feature selection')
    st.write('Heatmap method: allows us to give an intuition')

    potential_features = df.iloc[:,np.r_[7,9:12,13:48]]
    fig, ax = plt.subplots(figsize=(25,25))
    sns.heatmap(potential_features.corr(), annot=True, ax=ax);
    st.pyplot(fig)

    st.subheader('Models')
    st.write('- Linear Regression')
    st.write('- Random Forest')
    st.write('- XGBoost')

    st.subheader('XGBoost')
    st.write('- Allows us to retain a model of 15 features')
    st.write('- We retain a model with a 0.41 R², which is a good result')
    st.write('- It is the model that maximizes the R² while minimizing the number of features')

    st.header('Classification')

    st.subheader('Feature selection')
    st.write('Random Forest Classifier method: allows us to select the most relevant features')
    df = df.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Localisation', 'name', 'date', 'Emplacement', 'Localisation2'], axis=1)
        # We define X as the features set and y as the target set
    X = df.drop(['Logement codé'],axis=1)
    y = df['Logement codé']

        # We display each variable's accuracy
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier()
    rf.fit(X, y)
    imp_var = pd.Series(rf.feature_importances_, index = X.columns).sort_values(ascending = False)
    st.write(imp_var)

    st.subheader('Models')
    st.write('- Naive bees')
    st.write('- SVC')
    st.write('- Random Forest')

    st.subheader('Random Forest')
    st.write('- We retained a model with 4 features to avoid overfitting')
    st.write('- We obtain a 97 % F1-score')
    st.write('- F1-score since it allows to mix the precision and the recall scores')


# -*- coding: utf-8 -*-

if selected == "Price simulator":
    st.title(f"You have selected {selected}")

    import streamlit as st
    import pandas as pd
    import xgboost as xgb
    import numpy as np
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_predict
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import KFold
    from sklearn.model_selection import GridSearchCV

    from sklearn.preprocessing import FunctionTransformer

    #Feature selection
    from sklearn.feature_selection import RFECV
    from sklearn.feature_selection import SelectFromModel
    from sklearn.model_selection import cross_val_predict

    st.title("Welcome to the AIRBNB price simulator - Como, Italy edition")
    st.caption('For those who plan their holidays while thinking about their bank account.')

    st.image('./lake-como-weather-varenna-summer.jpg', use_column_width= True)

    st.caption('Please enter the housing features to predict the prices.')

    #Start with numerical values

    placeholder_rooms = st.empty()
    placeholder_travelers = st.empty()
    placeholder_beds = st.empty()
    placeholder_baths = st.empty()


    nrooms = placeholder_rooms.number_input('Number of rooms in the listing: ', min_value= 0)
    ntrav = placeholder_travelers.number_input('Number of travelers: ', min_value= 0)
    nbeds = placeholder_beds.number_input('Number of beds: ', min_value= 0)
    nbaths = placeholder_baths.number_input('Number of bathrooms: ', min_value= 0)


    #rooms = st.text_input('Number of rooms in the listing')
    #st.write('You have entered', rooms,'room(s)')

    #travelers = st.text_input('Number of travelers included')
    #st.write('There are', travelers,'traveler(s)')

    #beds = st.text_input('Number of beds in the listing')
    #st.write('You have entered', beds,'bed(s)')

    #bathroom = st.text_input('Number of bathrooms in the listing')
    #st.write('You have entered', bathroom,'bathroom(s)')


    #Many different ways to treat the binary values

    #options = st.multiselect('What are the features included in the listing?',['Hairdryer', 'Bidet', 'Iron dryer', 'Oven', 'Air conditionner', 'Pool'])

    st.caption('What types of amenities does your housing have?')

    Pool = st.checkbox('Pool')
    AC = st.checkbox('Air Conditionner (AC)')
    oven = st.checkbox('Oven')
    hairdryer = st.checkbox('Hairdryer')
    bidet = st.checkbox('Bidet')
    ID = st.checkbox('Iron dryer')




    if hairdryer:
        HD = 1
    else: HD = 0
    if bidet:
        BD = 1
    else : BD = 0
    if ID:
        irdr = 1
    else: irdr = 0
    if oven:
        ov = 1
    else: ov = 0
    if AC:
        air = 1
    else: air = 0
    if Pool:
        pisc = 1
    else: pisc = 0

    view = st.selectbox(
         'Does your listing have a view on Lake Como?',
         ('Yes', 'No'))

    if view=='Yes':
        vue = 1
    else: vue = 0

    typehs = st.selectbox(
         'Please select the type of housing you plan on staying in',
         ('Shared room', 'Private room in a condominium', 'Private room in individual housing',
          'Private room in a loft', 'Private room in a suite', 'Private room in a chalet', 
          'Private room in a villa', 'Entire rental in condomium', 'Entire rental in individual housing',
          'Entire rental in loft', 'Entire rental in suite', 'Entire rental in chalet', 
          'Entire rental in suite', 'Houseboat'))

    if typehs == 'Shared room': hs = 0
    elif typehs == 'Private room in a condominium': hs = 1
    elif typehs == 'Private room in individual housing': hs = 2
    elif typehs == 'Private room in a loft': hs = 3
    elif typehs == 'Private room in a suite': hs = 3
    elif typehs == 'Private room in a chalet': hs = 3
    elif typehs == 'Private room in a villa': hs = 3
    elif typehs == 'Entire rental in condomium': hs = 4
    elif typehs == 'Entire rental in individual housing': hs = 5
    elif typehs == 'Entire rental in loft': hs = 6
    elif typehs == 'Entire rental in suite': hs = 6
    elif typehs == 'Entire rental in chalet': hs = 6
    elif typehs == 'Entire rental in suite': hs = 6
    else: hs = 6

    #Import the list of the cities + the scores used
    dtc = pd.read_csv("total_location.csv")
    city_list = dtc["Localisation2"].values.tolist()

    city = st.selectbox('Please choose the city you are staying in',(city_list))

    score = dtc.loc[dtc['Localisation2'] == city, 'Score'].iloc[0]


    #create the new dataframe with all the values 
    test = [nrooms,ntrav,nbeds,nbaths,HD,BD,irdr,ov,air,pisc,vue,score,hs]
    t1 = pd.DataFrame([test], columns=['Nombre de chambres',
     'Nombre de voyageurs',
     'Nombre de lits',
     'Nombre de salles de bains',
     'Sèche-cheveux_True',
     'Bidet_True',
     'Fer à repasser_True',
     'Four_True',
     'Climatisation_True',
     'Piscine_True','Vue sur le lac_True','Score','Logement codé'])







    #hairdry = st.selectbox('Does your listing contain a hairdryer?',('Yes', 'No'))

    #bidet = st.selectbox('Does your listing contain a bidet?',('Yes', 'No'))

    #fer = st.selectbox('Does your listing contain a iron dryer?',('Yes', 'No'))

    #oven = st.selectbox('Does your listing contain a oven?',('Yes', 'No'))

    #aircond = st.selectbox('Does your listing contain a air conditionner?',('Yes', 'No'))

    #pool = st.selectbox('Does your listing contain a pool?',('Yes', 'No'))


    #Import the list of the cities + the scores used
    #dtc = pd.read_csv("total_location.csv")
    #city_list = dtc["Localisation2"].values.tolist()

    #city = st.selectbox('Please choose the city you are staying in',(city_list))


    #Let's train our model once again 
    #Importing the CSV on which we are going to work on 
    df = pd.read_csv('Dataframefinal_clean.csv') #total CSV with floats
    model = df.iloc[:,np.r_[7:12,13:48]] #only keeping the numerical values
    y = model["Prix par nuit"]
    X = model.iloc[:,np.r_[0,2:40]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=893717398)

    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                    max_depth = 5, alpha = 10, n_estimators = 1000) #xgboost
    feature_col_x_bis = ['Nombre de chambres',
     'Nombre de voyageurs',
     'Nombre de lits',
     'Nombre de salles de bains',
     'Sèche-cheveux_True',
     'Bidet_True',
     'Fer à repasser_True',
     'Four_True',
     'Climatisation_True',
     'Piscine_True',
     'Vue sur le lac_True',
     'Score',
     'Logement codé']
    #Xbis = df[feature_col_x_bis]
    #feature_y = ['Prix par nuit']
    #y = df[feature_y]

    #create the function transformer object with Logarithm transformation 
    #logarithm_transfer = FunctionTransformer(np.log, validate = True)

    #Apply the transformation
    #data_new = logarithm_transfer.transform(y)

    #R2 = cross_val_score(xg_reg, Xbis, data_new, cv=10, scoring='r2').mean()
    #R2
    #y_pred = cross_val_predict(xg_reg, Xbis, data_new, cv=10)

    data_new = np.log(y_train) #transform into logarithm

    xg_reg.fit(X_train[feature_col_x_bis],data_new) #fit the xgboost with the variables chosen with RFECV + log(y)
    prediction = xg_reg.predict(t1[feature_col_x_bis])
    pred = np.exp(prediction) #retransform it into an exponential value

    #R2 = cross_val_score(xg_reg, X_train[feature_col_x_bis], data_new, cv=10, scoring='r2').mean()
    #R2 #almost 0.4153

    result = st.button("Find my price") #if clicking on the button, displays the price
    if result: 
        st.text("Our calculator predicts the following price : ")
        pred[0]















