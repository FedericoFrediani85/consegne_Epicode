#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas


# In[2]:


import pandas as pd


# In[3]:


file_path = r'C:\Users\Lenovo\Desktop\covid.csv'


# In[4]:


df = pd.read_csv (file_path)


# In[5]:


print (df)


# In[6]:


# 1. Dimensioni del dataset (righe e colonne)
print(f"Dimensioni del dataset: {df.shape}")

# 2. Metadati del dataset
print("\nInformazioni sui metadati del dataset:")
df.info()

# 3. Descrizione statistica
print("\nDescrizione statistica del dataset:")
print(df.describe())

# 4. Visualizzazione del tipo di dato contenuto nelle colonne
print("\nTipo di dati per ogni colonna:")
print(df.dtypes)


# In[7]:


import pandas as pd

# rimoazione valori NA
df_clean = df.dropna(subset=['total_cases'])

# Raggruppamento per continente
cases_by_continent = df_clean.groupby('continent')['total_cases'].sum().reset_index()

# Calcolo del totale dei casi
total_cases_worldwide = df_clean['total_cases'].sum()

# Aggiunta della colonna % rispetto al totale mondiale
cases_by_continent['percentage_of_world'] = (cases_by_continent['total_cases'] / total_cases_worldwide) * 100

# Visualizzazione dei isultati
print(cases_by_continent)


# In[8]:


import matplotlib.pyplot as plt

# Convertire la colonna 'date' in formato datetime per una corretta gestione temporale
df['date'] = pd.to_datetime(df['date'])

# Filtro dei dati per paese Italia e per anno 2022
df_italy_2022 = df[(df['location'] == 'Italy') & (df['date'].dt.year == 2022)]

# Rimozione delle righe NA nei casi totali o nei nuovi casi
df_italy_2022_clean = df_italy_2022.dropna(subset=['total_cases', 'new_cases'])

# Grafico 1: Evoluzione dei casi totali in Italia nel 2022
plt.figure(figsize=(10, 6))
plt.plot(df_italy_2022_clean['date'], df_italy_2022_clean['total_cases'], label='Totale Casi', color='b')
plt.title("Evoluzione dei Casi Totali in Italia (2022)")
plt.xlabel("Data")
plt.ylabel("Casi Totali")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

# Grafico 2: Numero di Nuovi Casi in Italia nel 2022
plt.figure(figsize=(10, 6))
plt.plot(df_italy_2022_clean['date'], df_italy_2022_clean['new_cases'], label='Nuovi Casi', color='r')
plt.title("Numero di Nuovi Casi in Italia (2022)")
plt.xlabel("Data")
plt.ylabel("Nuovi Casi")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()


# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Converti la colonna 'date' in formato datetime
df['date'] = pd.to_datetime(df['date'])

# Filtriamo i dati per il periodo tra maggio 2022 e aprile 2023 e per Italia, Germania e Francia
df_selected_countries_icu = df[(df['location'].isin(['Italy', 'Germany', 'France'])) &
                               (df['date'] >= '2022-05-01') & (df['date'] <= '2023-04-30')]

# Rimozione delle righe NA nei pazienti in terapia intensiva
df_selected_countries_icu_clean = df_selected_countries_icu.dropna(subset=['icu_patients'])

# Creiamo un boxplot per visualizzare la distribuzione dei pazienti in terapia intensiva
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_selected_countries_icu_clean, x='location', y='icu_patients', palette='Set2')

# Aggiungiamo il titolo e le etichette
plt.title('Distribuzione dei Pazienti in Terapia Intensiva (ICU) - Italia, Germania, Francia (Maggio 2022 - Aprile 2023)', fontsize=14)
plt.xlabel('Nazione', fontsize=12)
plt.ylabel('Numero di Pazienti in Terapia Intensiva (ICU)', fontsize=12)
plt.tight_layout()

# Mostriamo il grafico
plt.show()


# In[17]:


import matplotlib.pyplot as plt

# Carica il dataset (assicurati che sia il tuo file CSV o il dataframe già caricato)
# df = pd.read_csv('your_dataset.csv')

# Converti la colonna 'date' in formato datetime
df['date'] = pd.to_datetime(df['date'])

# Filtriamo i dati per Italia, Germania, Francia e Spagna nel 2021
df_selected_countries_2021 = df[(df['location'].isin(['Italy', 'Germany', 'France', 'Spain'])) & 
                                (df['date'].dt.year == 2021)]

# Sommiamo i pazienti ospitalizzati per ciascun paese
hospitalized_sum = df_selected_countries_2021.groupby('location')['hosp_patients'].sum().reset_index()

# Mostriamo i risultati numericamente
print("Somma dei pazienti ospitalizzati nel 2021 per ciascun paese:")
print(hospitalized_sum)

# Grafico della somma dei pazienti ospitalizzati
plt.figure(figsize=(8, 5))
plt.bar(hospitalized_sum['location'], hospitalized_sum['hosp_patients'], color='skyblue')
plt.title('Somma dei Pazienti Ospitalizzati (2021) per Italia, Germania, Francia e Spagna')
plt.xlabel('Paese')
plt.ylabel('Somma dei Pazienti Ospitalizzati')
plt.tight_layout()
plt.show()

# Check dati nulli nella colonna 'hosp_patients'
missing_data = df_selected_countries_2021['hosp_patients'].isnull().sum()
print(f"\nNumero di dati mancanti nella colonna 'hosp_patients': {missing_data}")

#  dati NA
if missing_data > 0:
    print("\nEsistono valori mancanti per i pazienti ospitalizzati.")
else:
    print("\nNon ci sono valori mancanti nella colonna 'hosp_patients'.")


# In[ ]:




