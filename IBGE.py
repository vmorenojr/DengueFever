import pandas as pd
import googlemaps
import pickle

gmaps = googlemaps.Client(key = 'AIzaSyBP1wd8YMxm9r5PTuI1kC7VbO8aMgmbbUk')

regioes = {'Norte':[['Acre', 'Rio Branco'],['Aamapá','Macapá'],['Amazonas','Manaus'],['Pará','Belém'],['Rondônia','Porto Velho'],['Roraima','Boa Vista'],['Tocantins','Palmas']], 'Nordeste': [['Alagoas', 'Maceió'], \
['Bahia','Salvador'],['Ceará','Fortaleza'],['Maranhão','São Luís'],['Paraíba','João Pessoa'],['Pernambuco','Recife'],['Piauí','Teresina'],['Rio Grande do Norte','Natal'],['Sergipe','Aracaju']], 'Centro-Oeste': \
[['Goiás','Goiânia'],['Mato Grosso','Cuiabá'],['Mato Grosso do Sul','Campo Grande']],'Sudeste': [['Espírito Santo','Vitória'],['Minas Gerais','Belo Horizonte'],['Rio de Janeiro','Rio de Janeiro'],['São Paulo','São Paulo']], \
'Sul':[['Paraná,Curitiba'],['Santa Catarina','Florianópolis'],['Rio Grande do Sul','Porto Alegre']]}


def cod_to_mun(cod, completo=True):
    #dado um código, retorna o município referente
    #cod = código do município
    #se completo = True, retorna o código completo
    df = pd.read_csv('Dados/codigos_ibge.csv')
    for x in df['Código Município Completo']:
        if int(x/10) == cod:
            cod = x
            break
    if completo:
        return(df.loc[df[df['Código Município Completo'] == cod].index.item(),'Nome_Município'])
    return(df.loc[df[df['Município'] == cod].index.item(),'Nome_Município'])


def cod_to_uf(cod):
    #dado um código, retorna a UF referente
    #cod = código do município
    df = pd.read_csv('Dados/codigos_ibge.csv')
    for x in df['Código Município Completo']:
        if int(x/10) == cod:
            cod = x
            break    
    return(df.loc[df[df['Código Município Completo'] == cod].index.item(),'Nome_UF'])

def return_loc(cod, municipio = True):
    #se municipio = True, dado um código, retorna a latitude e longitude do município referente ao código
    #se municipio = False, retorna a latitude referente a UF do código
    #cod = codigo do município
    #retorna: list com [lat,long]
    if municipio:
        latlng = gmaps.geocode(str(cod_to_mun(cod)) + ' ' + str(cod_to_uf(cod)) + ' Brazil')[0]['geometry']['location']
        return(latlng[0],latlng[1])
    latlng = gmaps.geocode(str(code_to_uf(cod)) + ' Brazil')[0]['geometry']['location']
    return(latlng[0],latlng[1])

def save_obj(obj, name):
    #salva um objeto em python (lista, dict, df etc) em um arquivo no tipo .pkl
    #entrada: obj = objeto a ser salvo , name = endereço/nome do arquivo a ser criado (sem o .pkl)
    #saida: objeto na pasta
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    #carrega um objeto em formato .pkl
    #name: endedeço do arquivo .pkl (sem o .pkl)
    #saida: o objeto guardado no arquivo .pkl
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

def group_by(df, per, reg):
    df.dt_sintoma = pd.to_datetime(df.dt_sintoma)
    df = df[df.dt_sintoma >= '2006-1-1']
    df.set_index('dt_sintoma', drop=True, inplace=True)
    df = df[df['regiao'] == reg]
    df.drop(columns=['co_municipio'],inplace = True)
    return (df.groupby(
        ['regiao', pd.Grouper(freq=per)]).sum())