import pandas as pd
from sklearn.cluster import DBSCAN
import plotly.express as px
from sklearn.preprocessing import StandardScaler


def new_clean_data():

    df = pd.read_csv('cleaned_data.csv')

    temp = []
    deaths = []
    cases = []

    for i in range(len(df)):    #Find the last insert of every country
        try:
            if df.loc[i, 'Entity'] != df.loc[i + 1, 'Entity']:
                deaths.append(df.loc[i, 'Deaths'])  #Append the overall deaths of every country
                cases.append(df.loc[i, 'Cases'])    #Append the overall cases of every country
                temp.append(df.loc[i, :])   #Append all elements of every country
        except:
            deaths.append(df.loc[i, 'Deaths'])  #Append the overall deaths of the last country
            cases.append(df.loc[i, 'Cases'])    #Append the overall cases of the last country
            temp.append(df.loc[i, :])   #Append all elements of the last country

    sum_tests = []
    for j in df['Entity'].unique():     #For every country find the tolal number of tests
        filt = (df['Entity'] == j)
        sum_tests.append((df.loc[filt, 'Daily tests']).sum())

    death_persentage = []
    Positivity_persentage = []
    population = df['Population'].unique()

    for k in range(len(deaths)):
        death_persentage.append(round(((deaths[k] / population[k]) * 100), 2))  #Find the death persentage of every country
        Positivity_persentage.append(round(((cases[k] / sum_tests[k]) * 100), 2))   #Find the positivity persentage of every country

    df = pd.DataFrame(data=temp)    #Create a new dataframe that contains the last insertion of every country
    df = df.drop(['Date', 'Daily tests'], axis=1)   #Drop Date, Daily tests
    df['Sum Tests'] = sum_tests #Insert the Sum tests of every country to the dataframe
    df['Deaths_Persentage'] = death_persentage  #Insert the death persentage of every country to the dataframe
    df['Positivity_Persentage'] = Positivity_persentage #Insert the positivity persentage of every country to the dataframe
    df.to_csv('new_cleaned_data.csv', index=False, encoding="utf-8-sig")    #save dataframe


new_clean_data()


def clust(par1, par2):

    df = pd.read_csv('new_cleaned_data.csv')

    data = df.loc[:, [par1, par2]]  #Locate clustering data

    scaller = StandardScaler()
    data = scaller.fit_transform(data)  #Scale data for better results

    x = []
    y = []
    for i in range(len(df)):    # Split data to x and y
        x.append(data[i][1])
        y.append(data[i][0])

    dbscan_cluster_model = DBSCAN(eps=0.5, min_samples=1).fit(data) #Initialize DBSCAN and fit data
    df['cluster'] = dbscan_cluster_model.labels_    #Save the clustering model labels
    fig = px.scatter(df, x=df[par2], y=df[par1], color=df['cluster'], text=df["Entity"])    #Plot clustering data x = par2, y = par2
    fig.update_traces(textposition='top center')
    fig.show()  #Show plot


clust('Positivity_Persentage', 'Average temperature per year')  #Plot the clustering data of x = 'Average temperature per year' and y = 'Positivity_Persentage'
#clust('Positivity_Persentage', 'Hospital beds per 1000 people')
#clust('Positivity_Persentage', 'Medical doctors per 1000 people')
#clust('Positivity_Persentage', 'GDP/Capita')
#clust('Positivity_Persentage', 'Median age')
#clust('Positivity_Persentage', 'Population aged 65 and over (%)')



#clust('Deaths_Persentage', 'Average temperature per year')
#clust('Deaths_Persentage', 'Hospital beds per 1000 people')
#clust('Deaths_Persentage', 'Medical doctors per 1000 people')
#clust('Deaths_Persentage', 'GDP/Capita')
#clust('Deaths_Persentage', 'Median age')
#clust('Deaths_Persentage', 'Population aged 65 and over (%)')



