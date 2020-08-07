import pandas as pd
from sklearn.preprocessing import Imputer, MinMaxScaler, PolynomialFeatures, LabelEncoder
df = pd.read_csv("data/train.csv").fillna('0')
test_df = pd.read_csv("data/test.csv").fillna('0')
drop_columns_list= ["Field_{}".format(i) for i in(1,2,5,6,7,8,9,11,15,18,25,32,33,34,35,40,43,44,45,68,46,48,49)] + ["currentLocationLocationId","F_startDate","F_endDate","E_endDate","E_startDate","C_startDate","C_endDate","G_startDate","G_endDate","A_startDate","A_endDate"] + ['partner0_B', 'partner0_K', 'partner0_L', 'partner1_B', 'partner1_D', 'partner1_E', 'partner1_F', 'partner1_K', 'partner1_L', 'partner2_B', 'partner2_G', 'partner2_K', 'partner2_L', 'partner3_B', 'partner3_C', 'partner3_F', 'partner3_G', 'partner3_H', 'partner3_K', 'partner3_L', 'partner4_A', 'partner4_B', 'partner4_C', 'partner4_D', 'partner4_E', 'partner4_F', 'partner4_G', 'partner4_H', 'partner4_K', 'partner5_B', 'partner5_C', 'partner5_H', 'partner5_K', 'partner5_L','ngaySinh','diaChi',"currentLocationLatitude","currentLocationLongitude","homeTownLocationId","homeTownLatitude","homeTownLongitude","data.basic_info.locale","currentLocationCity","currentLocationCountry","currentLocationName","currentLocationState","homeTownCity","homeTownCountry","homeTownName"]
df = df.drop(drop_columns_list,axis=1)
test_df= test_df.drop(drop_columns_list,axis=1)
list_drop = []

le = LabelEncoder()
le_count = 0

# Iterate through the columns
print(df.dtypes)
for col in df:
    if df[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(df[col].unique())) <= 10:
            # Train on the training data
            le.fit(df[col].values.tolist())
            df[col] = df[col].cat.codes
            # Keep track of how many columns were label encoded
            le_count += 1
print(df)

# for col in df.columns:
#     if(col!="label" and col!="id"):
#         train_list = list(set(df[col].values.tolist()))
#         test_list = list(set(test_df[col].values.tolist()))
#         if(len(train_list)!=len(test_list)):
#             print(col)
#             if(col=="F_numQuery"):
#                 print(train_list)
#                 print(test_list)
#             if(len(train_list)<10):
#                 print(train_list)
#                 print(test_list)
