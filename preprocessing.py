from google.colab import files
uploaded = files.upload()
import pandas as pd

df = pd.read_excel("H1B.csv.xlsx")
print("Original dataset size:", len(df))
df.head()

import pandas as pd

df = pd.read_excel("H1B.csv.xlsx")
# Convert dates
df['CAcase_submitted'] = pd.to_datetime(df['case_submitted'], errors='coerce')
df['decision_date'] = pd.to_datetime(df['decision_date'], errors='coerce')

# Remove rows with missing dates
df = df.dropna(subset=['case_submitted', 'decision_date'])

# Create processing time
df['Processing_Days'] = (df['decision_date'] - df['case_submitted']).dt.days

# Remove invalid rows
df = df[df['Processing_Days'] >= 0]

print("Cleaned dataset size:", len(df))
df_20k['Processing_Months'] = (df_20k['Processing_Days'] / 30).round().astype(int)
df_20k = df_20k.rename(columns={
    'case_submitted': 'Application_Date',
    'decision_date': 'Decision_Date',
    'case_status': 'Visa_Status',
    'employer_state': 'Processing_Office'
})

# Add Visa_Type manually (since it's H1B dataset)
df_20k['Visa_Type'] = 'H1B'

df_20k.head()
state_map = {
    'AL': 'Alabama',
    'AK': 'Alaska',
    'AZ': 'Arizona',
    'AR': 'Arkansas',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DE': 'Delaware',
    'DC': 'District of Columbia',
    'FL': 'Florida',
    'GA': 'Georgia',
    'HI': 'Hawaii',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'IA': 'Iowa',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'ME': 'Maine',
    'MD': 'Maryland',
    'MA': 'Massachusetts',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MS': 'Mississippi',
    'MO': 'Missouri',
    'MT': 'Montana',
    'NE': 'Nebraska',
    'NV': 'Nevada',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NY': 'New York',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VT': 'Vermont',
    'VA': 'Virginia',
    'WA': 'Washington',
    'WV': 'West Virginia',
    'WI': 'Wisconsin',
    'WY': 'Wyoming'
}

df_20k['Processing_Office'] = df_20k['Processing_Office'].replace(state_map)
df_20k.to_csv("Final_Dataset.csv", index=False)
print("File saved successfully!")
from google.colab import files
files.download("Final_Dataset.csv")




