import requests
import urllib
import codecs
import urllib.request
import csv
import json
import xmltodict
import pandas as pd
from matplotlib import pyplot as plt
import datetime
import numpy as np
import pandas
from matplotlib.patches import Patch
from datetime import datetime


def csv_converter(url):
    csvList = []

    listCount = 0
    # Start Date of the monkey pox data set
    startDate = datetime(2022, 4, 22).date()

    try:

        rData = requests.get(url)

        # Creating a temporary file and saving the requested data in it
        # The file will be deleted after the check.

        toExtractFilePath = "to_extract_dataset_1.csv"

        with open(toExtractFilePath, 'wb') as fw:
            fw.write(rData.content)
        fw.close()

        fr = open(toExtractFilePath, 'r')
        # fr = open("./data/to_extract_dataset_1.csv")
        values = csv.reader(fr)
        listValues = list(values)

        for i in listValues:
            try:
                converted_value = datetime.strptime(i[0], "%d/%m/%Y").date()
                if converted_value >= startDate:
                    csvList.append(i)
                else:
                    pass

            except ValueError:
                continue

        fr.close()

        csvListSort = csvList
        for i in csvList:
            convertedDate = datetime.strptime(i[0], "%d/%m/%Y").date()
            csvListSort[listCount][0] = convertedDate
            listCount += 1

        csvListSort = sorted(csvListSort)

        # clearing the csv list and list count
        csvList = []
        listCount = 0

        csvList.extend(csvListSort)

        for i in csvListSort:
            csvList[listCount][0] = i[0].strftime('%Y-%m-%d')
            listCount += 1

        # Writing csv files
        # https://www.geeksforgeeks.org/writing-csv-files-in-python/ , 07.11.2022
        '''  #the following lines create the extracted covid data set. we comment them out as to not create a new dataset inside the enviroment

        with open("extracted_covid_data.csv", 'w', newline='') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            # writing the fields
            csvwriter.writerow(listValues[0])

            # writing the data rows
            csvwriter.writerows(csvList)
            '''
    except:
        pass


if __name__ == '__main__':
    url = "https://opendata.ecdc.europa.eu/covid19/nationalcasedeath_eueea_daily_ei/csv"
    csv_converter(url)

############################################################################################################


dataset1= {
    "creator" : "European Centre for Disease Prevention and Control (ECDC)",
    "catalogName" : "European Data Portal" ,
    "catalogURL" : "https://data.europa.eu/" ,
    "datasetID" : "https://www.ecdc.europa.eu/en/publications-data/data-daily-new-cases-covid-19-eueea-country" ,
    "resourceURL" : "https://github.com/SeeOneeee/data_processing_1_shared_repo/raw/main/extracted_covid_data.csv"  ,
    "pubYear" : "2022"  ,
    "lastAccessed" : "2022-11-08T22:46:59.988120"
}

dataset2= {
    "creator" : "European Centre for Disease Prevention and Control (ECDC)" ,
    "catalogName" : "European Data Portal" ,
    "catalogURL" : "https://data.europa.eu/" ,
    "datasetID" : "http://data.europa.eu/88u/dataset/monkeypox" ,
    "resourceURL" : "https://opendata.ecdc.europa.eu/monkeypox/casedistribution/json/data.json"  ,
    "pubYear" : "2022"  ,
    "lastAccessed" : "2022-11-08T22:47:00.265477"  ,
}


def accessData(datadict):
    file_format_url = 'unknown'
    # getting the file format from the url
    url = datadict['resourceURL']
    find_data_type = url.split('.')
    find_data_type.extend(url.split('/'))
    find_data_type = set(find_data_type)
    data_formats = ["csv", "json", "xml", "tsv"]
    for i in data_formats:
        if i in find_data_type:
            file_format_url = i.upper()
            break

    file_format_header = 'unknown'
    # getting the file format from the headers
    rData = requests.get(datadict['resourceURL'])
    headerDict = rData.headers
    try:
        if 'Content-Type' in headerDict:
            temp = (str(headerDict['Content-Type']).split("/"))[1]
            if temp in data_formats:
                file_format_header = temp.upper()
    except:
        pass

    # getting the file size
    file_size = int(headerDict['Content-Length'])
    try:
        if headerDict['Accept-Ranges'].lower() == 'bytes':  # if it is in bytes we convert to KB
            file_size = file_size / 1000
    except:
        pass

    if file_format_url == file_format_header:
        datadict.update({
                            "detectedFormat": f"{file_format_url}"})  # https://www.w3schools.com/python/python_dictionaries_add.asp 07.11. 2022
    elif file_format_url == 'unknown':  # update function adapted from above source
        datadict.update({"detectedFormat": f"{file_format_header}"})
    elif file_format_header == 'unknown':
        datadict.update({"detectedFormat": f"{file_format_url}"})

    if datadict['detectedFormat'] == 'unknown':
        datadict.update({"filesizeKB": 0})
    else:
        datadict.update({"filesizeKB": file_size})

    return datadict



def parseFile(datadict, format):

    responseValidate = False
    fileFormat = datadict["detectedFormat"]
    filePath = datadict["resourceURL"]

    if format == fileFormat:

        if fileFormat == "CSV":
            responseValidate = validate_csv(filePath)

        elif fileFormat == "JSON":
            responseValidate = validate_json(filePath)

        elif fileFormat == "XML":
            responseValidate = validate_xml(filePath)

    else:
        responseValidate = False
        print("Wrong file-format")
        pass

    print("File is valid: " + str(responseValidate))

    return responseValidate

def validate_csv(filePath):

    correctCSV = False

    try:
        resp = urllib.request.urlopen(filePath)
        listData = []
        countIterate = 0
        correctDelimiter = False
        correctRows = True

        if not correctDelimiter:
            csvfile = csv.reader(codecs.iterdecode(resp, 'utf-8'), delimiter=',')
            if len(csvfile.__next__()) > 1:
                correctDelimiter = True

        if not correctDelimiter:
            csvfile = csv.reader(codecs.iterdecode(resp, 'utf-8'), delimiter=';')
            if len(csvfile.__next__()) > 1:
                correctDelimiter = True

        if not correctDelimiter:
            csvfile = csv.reader(codecs.iterdecode(resp, 'utf-8'), delimiter='\t')
            if len(csvfile.__next__()) > 1:
                correctDelimiter = True

        for i in csvfile:
            listData.append(i)

        countColumns = len(listData[0])

        while countIterate < len(listData):
            if len(listData[countIterate]) < countColumns:
                print("Error not full")
                correctRows = False
                break
            countIterate += 1

        if correctRows and correctDelimiter:
            correctCSV = True
        else:
            correctCSV = False
    except:

        pass

    return correctCSV


def validate_json(filePath):
    correctJSON = False
    iterationMax = 2

    try:
        resp = urllib.request.urlopen(filePath)
        jsonfile = json.loads(resp.read().decode('utf-8'))
        # Just to test the data - possible parsing

        for i in jsonfile:
            iterationMax -= 1
            if iterationMax == 0:
                break

        correctJSON = True

    except:
        print("Invalid JSON File")
        pass

    return correctJSON

def validate_xml(file):
    correctXML = False
    try:
        with urllib.request.urlopen(file) as f:
            xml_file = (xmltodict.parse(f.read()))
            # Try to parse the file
            correctXML = True
    except:
        pass

    f.close()
    return correctXML


def describeFile(datadict):
    if datadict['detectedFormat'] == 'CSV':

        if parseFile(datadict, 'CSV') == False:  # if parsing is false return empty dictionary
            return {}

        resp = urllib.request.urlopen(datadict['resourceURL'])
        csvfile = csv.reader(codecs.iterdecode(resp, 'utf-8'))

        csvfile_header = csvfile.__next__()  # loading the header (column names) so it does not get counted as a row

        maximum = 0
        row_num = 0
        for row in csvfile:
            row_num += 1
            for n in range(len(row)):
                if len(row[n]) > maximum:  # finding the longest text
                    maximum = len(row[n])
                    max_col = n
        col_num = len(csvfile_header)  # the number of elements in the header correspond to number of columns

        result = {
            "numberOfColumns": col_num,
            "numberOfRows": row_num,
            "longestColumn": max_col
        }
        return result

    elif datadict['detectedFormat'] == 'JSON':

        if parseFile(datadict, 'JSON') == False:  # if parsing is false return empty dictionary
            return {}

        path = urllib.request.urlopen(datadict['resourceURL'])

        jsonfile = json.loads(path.read().decode('utf-8'))

        attr_num = 0
        nest_dpth = 0
        for i in str(jsonfile[1]):  # https://www.tutorialspoint.com/find-depth-of-a-dictionary-in-python 08.11. 2022
            if i == '{':  # counting of brackets to count the nesting from source above
                nest_dpth += 1
            elif i == ':':
                attr_num += 1

        max_list = 0
        for i in jsonfile:
            values = i.values()
            for n in values:
                if type(n) == list and len(n) > max_list:
                    max_list = len(n)

        result = {
            "numberOfAttributes": attr_num,
            "nestingDepth": nest_dpth,
            "longestListLength": max_list
        }
        return result

    elif datadict['detectedFormat'] == 'XML':

        if parseFile(datadict, 'XML') == False:  # if parsing is false return empty dictionary
            return {}

        with urllib.request.urlopen(datadict['resourceURL']) as f:
            xml_file = (xmltodict.parse(f.read()))
        f.close()

        attr_num = 0
        nest_dpth = 0
        for i in str(xml_file[1]):  # https://www.tutorialspoint.com/find-depth-of-a-dictionary-in-python 08.11. 2022
            if i == '{':  # counting of brackets to count the nesting from source above
                nest_dpth += 1
            elif i == ':':
                attr_num += 1

        max_child = 0
        root = xml_file.values()

        for i in root:
            values = i.values()
            for n in values:
                if type(n) == list and len(
                        n) > max_child:
                    max_child = len(n)

        result = {
            "numberOfElementsAttributes": attr_num,
            "nestingDepth": nest_dpth,
            "maxChildren": max_child
        }
        return result



data = pd.read_csv('data_notebook-1_extcovid.csv')

# DATA PREPARATION

# we sort the countries by population so we can show them in the plot later
sorted_countries = data.groupby('countryterritoryCode')['popData2020'].mean().sort_values(
    ascending=False).index.tolist()
# we aggregate the cases for each country by month
temp_country_cases = data.groupby(['countryterritoryCode', 'month'])['cases'].sum()

# we store all of the values into a dictionary, keeping only the 4 largest cities and adding the rest to 'other' while also calculating the total
country_aggregate = {'total': 0, 'other': 0}
for i in range(len(sorted_countries)):
    if i <= 3:
        country_aggregate[sorted_countries[i]] = (temp_country_cases[sorted_countries[i]])
    else:
        country_aggregate['other'] += (temp_country_cases[sorted_countries[i]])
    country_aggregate['total'] += (temp_country_cases[sorted_countries[i]])

# we create a new dictionary where we divide the values by 100 000 so the values are easier to understand
data_aggregate = {'month': country_aggregate['total'].index.tolist()}
for i in country_aggregate:
    data_aggregate[i] = []
    for month in country_aggregate[i].index.tolist():
        data_aggregate[i].append(country_aggregate[i][month] / 100000)
# transforming the dictionary into a dataframe
data_aggregate = pd.DataFrame(data_aggregate)
# changing the months from numbers into abbreviations
month_mapping = {
    1: 'Jan',
    2: 'Feb',
    3: 'Mar',
    4: 'Apr',
    5: 'May',
    6: 'Jun',
    7: 'Jul',
    8: 'Aug',
    9: 'Sep',
    10: 'Oct',
    11: 'Nov',
    12: 'Dec'
}

# Replace numerical months with abbreviations
data_aggregate['month'] = data_aggregate['month'].map(month_mapping)
# PLOTTING

# setting the size of the figure
fig, ax = plt.subplots(figsize=(12, 6))
# changing the amount of ticks for the x-axis
ax.set_xticks(np.arange(0, max(data_aggregate['total']),
                        10))  # https://www.folkstalk.com/tech/changing-the-number-of-ticks-on-a-matplotlib-plot-axis-with-code-examples/  29.11. 2022
# removing the spines of the figure
for s in ['top', 'bottom', 'left', 'right']: ax.spines[s].set_visible(False)
# removing the tick markers on both axes
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
# inverting the y-axis so the first value is the earliest entry
ax.invert_yaxis()
# setting the grid to be behind the bar charts and creating the grid
ax.set_axisbelow(
    True)  # https://stackoverflow.com/questions/1726391/matplotlib-draw-grid-lines-behind-other-graph-elements 29.11. 2022
ax.grid(color='gray', alpha=0.6, linestyle='--', linewidth=1)

months = data_aggregate['month'].tolist()
# plotting the bar charts, we start with the total even though it will not be visible, because we will use it to write the annotaion
ax.barh(months, data_aggregate['total'], height=0.5)
ax.barh(months, data_aggregate['DEU'], height=0.5, label='DEU')
ax.barh(months, data_aggregate['FRA'], height=0.5, left=data_aggregate['DEU'],
        label='FRA')  # https://www.tutorialspoint.com/horizontal-stacked-bar-chart-in-matplotlib 29.11. 2022
ax.barh(months, data_aggregate['ITA'], height=0.5, left=data_aggregate['FRA'] + data_aggregate['DEU'], label='ITA')
ax.barh(months, data_aggregate['ESP'], height=0.5,
        left=data_aggregate['FRA'] + data_aggregate['DEU'] + data_aggregate['ITA'], label='ESP')
ax.barh(months, data_aggregate['other'], height=0.5,
        left=data_aggregate['FRA'] + data_aggregate['DEU'] + data_aggregate['ITA'] + data_aggregate['ESP'],
        label='Other')

# writing out the annotations
n = -1
x = [0 for i in range(len(data_aggregate['month']))]
first = 0
for i in ax.patches:
    # we first write out the totals
    if first < 7:
        plt.text(i.get_width() + 0.3, i.get_y() + 0.45, str(round((i.get_width()), 2)), fontsize=10,
                 fontweight='bold')  # https://www.geeksforgeeks.org/bar-plot-in-matplotlib/  29.11. 2022
        first += 1
    # and after that the individual countries
    else:
        n += 1
        if x[n] + i.get_width() < x[n] + 3.5 and x[n] != 0:
            x[n] += i.get_width()
        else:
            x[n] += i.get_width()
            plt.text(x[n] - 3.5, i.get_y() + 0.3, str(round((i.get_width()), 1)), fontsize=8,
                     fontweight='bold')  # https://www.geeksforgeeks.org/bar-plot-in-matplotlib/  29.11. 2022
        if n == 6:
            n = -1

# adding the scale for the x-axis, title and legend for the countries
ax.set_xlabel('1 : 100 000')
ax.set_title('COVID-19 cases in the EU between April and November 2022', loc='left', )
plt.legend()
# printing the figure
plt.show()

####### PLOT NUMBER 2 #########
def vis_data_2():
    mpDf = pandas.read_json(r'data_notebook-1_monkeypox.json')
    mpDf["DateRep"] = pandas.to_datetime(mpDf['DateRep'], format='%Y-%m-%d')

    # For Monthly aggregatated data
    temp2 = mpDf.groupby(mpDf.DateRep.dt.month)['ConfCases'].sum()
    data_aggregate2 = pandas.DataFrame({'month': temp2.index.tolist(), 'cases': temp2.values.tolist()})
    month_names = ['April', 'May', 'June', 'July', 'August', 'September', 'October']
    data_aggregate2['month_names'] = month_names

    print(data_aggregate2['cases'].mean())

    cases_color = [{c <= 2954: 'lightblue', c > 2954: 'orange'}[True] for c in data_aggregate2['cases']]
    color_legend = {"under_mean": "lightblue", "over_mean": "orange"}

    plotData = data_aggregate2.reindex(columns=['month', 'month_names', 'cases'])

    a1 = data_aggregate2.plot(x='month_names', y='cases', kind="bar", color=cases_color)
    a1.legend([Patch(facecolor=color_legend['under_mean']), Patch(facecolor=color_legend['over_mean'])],
              ["under_mean", "over_mean"])
    a1.bar_label(a1.containers[0])
    plt.xticks(rotation=30, horizontalalignment="center")
    plt.title("EU Monkey pox cases")
    plt.xlabel("Year: 2022")
    plt.ylabel("Number of cases")


if __name__ == '__main__':
    vis_data_2()


##### PLOT NUMBER 3 #####
data1 = pd.read_csv("data_notebook-1_extcovid.csv")
data2 = pd.read_json("data_notebook-1_monkeypox.json")

data2['DateRep'] = pd.to_datetime(data2['DateRep'], errors='coerce')
data1['dateRep'] = pd.to_datetime(data1['dateRep'], errors='coerce')

data1['cases'] = data1['cases'].astype('Int64') # make covid cases 'int64' instead of 'float64'

data1['dateRep'] = data1['dateRep'].dt.month   # we are interested in monthly data
data2['DateRep'] = data2['DateRep'].dt.month

grouped_data1 = data1.groupby('dateRep')['cases'].sum()    # grouping by month and covid cases
grouped_data2 = data2.groupby('DateRep')['ConfCases'].sum()   # grouping by month and monkeypox cases

grouped_data1 = grouped_data1.reset_index()
grouped_data2 = grouped_data2.reset_index()

grouped_data1.columns = ['Date', 'Cases_Covid']   # name the columns
grouped_data2.columns = ['Date', 'Cases_MonkeyPox']  # name the columns

merged_data = pd.merge(grouped_data1, grouped_data2, on=['Date'])  # merging two dataframes on months
merged_data['Date'] = merged_data['Date'].replace([4, 5, 6, 7, 8, 9, 10, 11], ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov"])

normalized_data = merged_data.copy() # create a copy of dataframe to normalize it
#function to normalize the data
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

normalized_data['Cases_MonkeyPox'] = normalize(normalized_data['Cases_MonkeyPox'])
normalized_data['Cases_Covid'] = normalize(normalized_data['Cases_Covid'])
y_axis1 = normalized_data['Cases_MonkeyPox'].tolist()
y_axis2 = normalized_data['Cases_Covid'].tolist()
x_axis = normalized_data['Date'].tolist()

plt.plot(x_axis, y_axis2, linewidth = 2.5)
plt.plot(x_axis, y_axis1, linewidth = 2.5)
plt.title("Normalized growth rate of COVID-19 and Monkeypox cases \nbetween April and November 2022")
plt.gca().legend(('COVID-19','Monkeypox'))
plt.show()

# persist merged dataset as a file
merged_data.to_csv("merged_dataset.csv")
