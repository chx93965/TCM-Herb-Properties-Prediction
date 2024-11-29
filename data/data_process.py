import pandas as pd
import ast

# 1. Assign titles to each column for chemicals.csv
chemicals_df = pd.read_csv("chemicals.csv", encoding="latin1", header=None)
column_names = chemicals_df.iloc[0, 0].split(',')
column_names.insert(0, "Unnamed_0")
if len(column_names) < chemicals_df.shape[1]:
    extra_cols = [f"Unnamed_{i}" for i in range(len(column_names), chemicals_df.shape[1])]
    column_names.extend(extra_cols)

chemicals_df.columns = column_names
chemicals_df = chemicals_df[1:].reset_index(drop=True)
chemicals_df = chemicals_df.apply(pd.to_numeric, errors="ignore")
chemicals_df.to_csv("chemicals_new.csv", index=False)

# 2. Convert categories/organs/flavors to boolean, will use them as targets for herb_component.csv
df = pd.read_csv('herb_components.csv', encoding='latin1')

categories = [
    ("Great cold", "isGreatCold"),
    ("Cold", "isCold"),
    ("Mildly cold", "isMildlyCold"),
    ("Cool", "isCool"),
    ("Even", "isEven"),
    ("Warm", "isWarm"),
    ("Mildly warm", "isMildlyWarm"),
    ("Hot", "isHot"),
    ("Great hot", "isGreatHot")
]

organs = [
    ("Liver meridian", "isLiverMeridian"),
    ("Spleen meridian", "isSpleenMeridian"),
    ("Kidney meridian", "isKidneyMeridian"),
    ("Heart meridian", "isHeartMeridian"),
    ("Large intestine meridian", "isLargeIntestineMeridian"),
    ("Stomach meridian", "isStomachMeridian"),
    ("Lung meridian", "isLungMeridian"),
    ("Gallbladder meridian", "isGallbladderMeridian"),
    ("Triple burner meridian", "isTripleBurnerMeridian"),
    ("Bladder meridian", "isBladderMeridian"),
    ("Others meridian", "isOthersMeridian"),
    ("Small intestine meridian", "isSmallIntestineMeridian"),
    ("Pericardium meridian", "isPericardiumMeridian")
]

flavors = [
    ("Bitter", "isBitter"),
    ("Pungent", "isPungent"),
    ("Sweet", "isSweet"),
    ("Astringent", "isAstringent"),
    ("Mildly bitter", "isMildlyBitter"),
    ("Salty", "isSalty"),
    ("Mildly sweet", "isMildlySweet"),
    ("Sour", "isSour")
]
for value, new_col in categories:
    df[new_col] = (df['Category'] == value).astype(int)
    
for value, new_col in organs:
    df[new_col] = df['Organs'].fillna("").apply(
        lambda x: int(any(value in item.strip() for item in x.split(',')))
    )

for value, new_col in flavors:
    df[new_col] = df['Flavors'].fillna("").apply(lambda x: int(value in x.split(',')))
    

df.to_csv('herbs.csv', index=False)

# 3. Reverse map the chemicals to new added targets
herbs_df = pd.read_csv("herbs.csv", encoding="latin1", header=0, index_col=False)
chemicals_df = pd.read_csv("chemicals_new.csv", encoding="latin1", header=0, index_col=False)

boolean_columns = [col for col in herbs_df.columns if col.startswith("is")]

for col in boolean_columns:
    if col not in chemicals_df.columns:
        chemicals_df[col] = 0

for _, herb_row in herbs_df.iterrows():
    try:
        chemical_indices = [int(idx) for idx in ast.literal_eval(herb_row["Chemicals"])]
    except (ValueError, SyntaxError):
        chemical_indices = []

    for idx in chemical_indices:
        if idx < len(chemicals_df):
            for col in boolean_columns:
                if herb_row[col] == 1:
                    chemicals_df.at[idx, col] = 1

chemicals_df.fillna(chemicals_df.mean(), inplace=True)  # Replace NaNs with column mean
chemicals_df.to_csv("updated_chemicals.csv", index=False)

