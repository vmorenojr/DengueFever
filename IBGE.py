import pandas as pd

def cod_to_mun(cod, completo=True):
    #dado um código, retorna o município referente
    #cod = código do município
    #se completo = True, retorna o código completo
    df = pd.read_csv('codigos_municipios_brasil_ibge.csv')
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
    df = pd.read_csv('codigos_municipios_brasil_ibge.csv')
    for x in df['Código Município Completo']:
        if int(x/10) == cod:
            cod = x
            break    
    return(df.loc[df[df['Código Município Completo'] == cod].index.item(),'Nome_UF'])