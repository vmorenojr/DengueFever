import pandas as pd
import googlemaps as gmp

#Key to use googlemaps API
gmaps = gmp.Client(key = '---YOUR API KEY HERE---')

#Loading the DataFrame
df = pd.read_csv('Dados/dengue.csv.gz')

#Creating a list with the full address of each state in the dataframe

states = []
for x in range(0,len(df)):
    t = df.loc[x]['UF']+', '+ df.loc[x]['municipio']+', '+df.loc[x]['estado']+', Brazil'
    if t in states:
        continue
    else:
        states.append(t)
        
        
#Making a list with states names
mun = [a for a in set(df.municipio)]

distance_mun = pd.DataFrame(index=mun,columns=mun)

#For each state, get the distance from every state in Brazil

for i in range(0,27):
    matrix = gmaps.distance_matrix(states[i],states)
    for j in range(0,27):
        try:
            distance_mun.iloc[i,j] = round(matrix['rows'][0]['elements'][j]['distance']['value']/1000000,2)
        except:
            continue
            
#Googlemaps couldn't get the distances for 2 states: 
#Macapá and Rio Branco, therefore we will get the great circle distance between these states and every other state in Brazil

def haversine(lon1, lat1, lon2, lat2): 
    from math import radians, cos, sin, asin, sqrt
    #Calculate the great circle distance between two points 
    #on the earth (specified in decimal degrees)
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers.
    return(c * r)

missing_states = [[states[12],12],[states[20],20]]

#Get the latitude and longitude of these states and insert into the distance_matrix
for x in missing_states:
    e1 = gmaps.geocode(x[0])
    lon1 = e1[0]['geometry']['location']['lng']
    lat1 = e1[0]['geometry']['location']['lat']   
    for i in range(0,27):
        e2 = gmaps.geocode(states[i])
        lon2 = e2[0]['geometry']['location']['lng']
        lat2 = e2[0]['geometry']['location']['lat']
        distance_mun.iloc[x[1],i] = round(haversine(lon1,lat1,lon2,lat2)/1000,2)
        distance_mun.iloc[i,x[1]] = round(haversine(lon1,lat1,lon2,lat2)/1000,2)
        
#Creating another version of the distance_matrix with different column names

dic = {X:x for X,x in zip(df.municipio,df.UF)}
distance_uf = distance_mun.rename(dic,columns=dic)

#save the dataframe
distance_uf.to_csv("distance_matrix_uf.csv")
distance_mun.to_csv("distance_matrix_mun.csv")

