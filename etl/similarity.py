import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import os
from utils import Constants


class similarity():
    def user_similarity(self, user_profile="resources/user_profile.csv", user_interests="resources/user_interests.csv"):
        user_profile_df = pd.read_csv(user_profile)
        user_profile_df.drop(['bio', 'description', 'first_name', 'last_name', 'email', 'phone_number'], axis=1,
                             inplace=True)

        user_profile_df["dob"] = pd.to_datetime(
            user_profile_df["dob"])  # Convert "dob" column to datetime if not already in that format
        current_date = pd.to_datetime('today').normalize()
        user_profile_df['dob'] = pd.to_datetime(user_profile_df['dob'])
        user_profile_df['age'] = (current_date - user_profile_df['dob']).astype('int64')

        mode_age = user_profile_df["age"].mode().values[0]
        mode_children = user_profile_df["children"].mode().values[0]
        fill_values = {"exercise": "exercise_unknown", "global": False, "maximum_distance": 1000,
                       "personality": "personality_unknown", "pets": "pets_unknown",
                       "relation_goal": "relation_goal_unknown", "sleeping": "sleeping_unknown",
                       "smoke": "smoke_unknown", "star_sign": "star_sign_unknown", "status": "Single",
                       "vaccinated": "vaccinated_unknown", "work_at": "work_at_uknonwn", "age": mode_age\
            , "education": "Unknown_education", "drink": "Unknown_drink", "diet": "Unknown_diet",
                       "children": mode_children}
        user_profile_df.fillna(fill_values, inplace=True)

        drink_encoder = preprocessing.LabelEncoder()
        exercise_encoder = preprocessing.LabelEncoder()
        education_encoder = preprocessing.LabelEncoder()
        smoke_encoder = preprocessing.LabelEncoder()

        user_profile_df['smoke_encoded'] = smoke_encoder.fit_transform(user_profile_df['smoke'])
        user_profile_df['smoke_encoded'].replace({'No': 0, 'Occasionally': 1, 'Yes': 3, 'smoke_unknown': 2},
                                                 inplace=True)

        user_profile_df['exercise_encoded'] = exercise_encoder.fit_transform(user_profile_df['exercise'])
        user_profile_df['exercise_encoded'].replace(
            {'Sedentary': 0, 'Light': 1, 'Moderate': 3, 'Active': 4, 'exercise_unknown': 2}, inplace=True)

        user_profile_df['education_encoded'] = education_encoder.fit_transform(user_profile_df['education'])
        user_profile_df['education_encoded'].replace(
            {'Unknown_education': 0, 'HighSchool': 1, 'Diploma': 2, 'AtUni': 3, 'Bachelors': 4, 'Masters': 5},
            inplace=True)

        user_profile_df['drink_encoded'] = drink_encoder.fit_transform(user_profile_df['drink'])
        user_profile_df['drink_encoded'].replace({'No': 0, 'Occasionally': 1, 'Yes': 2, 'drink_unknown': 3},
                                                 inplace=True)

        diet_encoder = preprocessing.OneHotEncoder()
        gender_encoder = preprocessing.OneHotEncoder()
        global_encoder = preprocessing.OneHotEncoder()
        pets_encoder = preprocessing.OneHotEncoder()
        preferance_encoder = preprocessing.OneHotEncoder()
        sleeping_encoder = preprocessing.OneHotEncoder()
        star_sign_encoder = preprocessing.OneHotEncoder()
        status_encoder = preprocessing.OneHotEncoder()
        vaccinated_encoder = preprocessing.OneHotEncoder()
        relation_goal_encoder = preprocessing.OneHotEncoder()
        children_encoder = preprocessing.OneHotEncoder()

        user_profile_df["age_diff"] = user_profile_df["age"] - 18

        diet_encoded = diet_encoder.fit_transform(user_profile_df[['diet']])
        diet_encoded_df = pd.DataFrame(diet_encoded.toarray(), columns=diet_encoder.\
                                       get_feature_names_out(['diet']))

        gender_encoded = gender_encoder.fit_transform(user_profile_df[['gender']])
        gender_encoded_df = pd.DataFrame(gender_encoded.toarray(), columns=gender_encoder.\
                                         get_feature_names_out(['gender']))

        global_encoded = global_encoder.fit_transform(user_profile_df[['global']])
        global_encoded_df = pd.DataFrame(global_encoded.toarray(), columns=global_encoder.\
                                         get_feature_names_out(['global']))

        pets_encoded = pets_encoder.fit_transform(user_profile_df[['pets']])
        pets_encoded_df = pd.DataFrame(pets_encoded.toarray(), columns=pets_encoder.\
                                       get_feature_names_out(['pets']))

        preferance_encoded = preferance_encoder.fit_transform(user_profile_df[['preferance']])
        preferance_encoder_df = pd.DataFrame(preferance_encoded.toarray(), columns=preferance_encoder.\
                                             get_feature_names_out(['preferance']))

        sleeping_encoded = sleeping_encoder.fit_transform(user_profile_df[['sleeping']])
        sleeping_encoded_df = pd.DataFrame(sleeping_encoded.toarray(), columns=sleeping_encoder.\
                                           get_feature_names_out(['sleeping']))

        relation_goal_encoded = relation_goal_encoder.fit_transform(user_profile_df[['relation_goal']])
        relation_goal_encoded_df = pd.DataFrame(relation_goal_encoded.toarray(), columns=relation_goal_encoder.\
                                                get_feature_names_out(['relation_goal']))

        star_sign_encoded = star_sign_encoder.fit_transform(user_profile_df[['star_sign']])
        star_sign_encoded_df = pd.DataFrame(star_sign_encoded.toarray(), columns=star_sign_encoder.\
                                            get_feature_names_out(['star_sign']))

        status_encoded = status_encoder.fit_transform(user_profile_df[['status']])
        status_encoded_df = pd.DataFrame(status_encoded.toarray(), columns=status_encoder.\
                                         get_feature_names_out(['status']))

        vaccinated_encoded = vaccinated_encoder.fit_transform(user_profile_df[['vaccinated']])
        vaccinated_encoded_df = pd.DataFrame(vaccinated_encoded.toarray(), columns=vaccinated_encoder.\
                                             get_feature_names_out(['vaccinated']))

        children_encoded = children_encoder.fit_transform(user_profile_df[['children']])
        children_encoded_df = pd.DataFrame(children_encoded.toarray(), columns=children_encoder.\
                                           get_feature_names_out(['children']))

        user_profile_df['latitude'] = user_profile_df['location'].apply(lambda x: x.split(" ")[0].replace("lat:", ""))
        user_profile_df['longitude'] = user_profile_df['location'].apply(lambda x: x.split(" ")[1].replace("long:", ""))

        user_profile_df['latitude'] = pd.to_numeric(user_profile_df['latitude'])
        user_profile_df['longitude'] = pd.to_numeric(user_profile_df['longitude'])

        path = 'resources/'+"user_profile_normalised"+".csv"
        user_profile_df.to_csv(path, index=False)

        user_profile_df = pd.concat(
            [user_profile_df, gender_encoded_df, global_encoded_df, pets_encoded_df, preferance_encoder_df,\
             sleeping_encoded_df, star_sign_encoded_df, status_encoded_df, vaccinated_encoded_df,
             relation_goal_encoded_df\
                , diet_encoded_df, children_encoded_df], axis=1)

        latitude_bins = Constants.latitude_bins
        longitude_bins = Constants.longitude_bins

        # Create labels based on the ranges
        user_profile_df['latitude_range'] = pd.cut(user_profile_df['latitude'], bins=latitude_bins)
        user_profile_df['longitude_range'] = pd.cut(user_profile_df['longitude'], bins=longitude_bins)

        # Group the DataFrame based on the labels and split into multiple DataFrames
        grouped_df = user_profile_df.groupby(['latitude_range', 'longitude_range'])

        # Create a dictionary to store the resulting DataFrames
        df_dict = {}
        for group_name, group_data in grouped_df:
            df_name = f"latitude_{group_name[0]}_longitude_{group_name[1]}"
            Constants.df_names.append(df_name)
            df_dict[df_name] = group_data.drop(columns=['latitude_range', 'longitude_range'])

        # Access individual DataFrames using keys in df_dict
        print(df_dict.keys())

        user_interests_df = pd.read_csv(user_interests)

        bool_columns = ['ayurveda', 'bollywood', 'classical_music', 'cooking', 'crafts', 'cricket', \
                        'cycling', 'dancing', 'fashion', 'food', 'gaming', 'gardening', 'gym', \
                        'hiking', 'history_and_culture', 'indian_cuisine', 'meditation', 'movies', \
                        'music', 'painting', 'pets', 'photography', 'reading', 'regional_dance', 'spirituality', \
                        'sports', 'swimming', 'technology', 'traveling', 'volunteering', 'writing', 'yoga']
        user_interests_df[bool_columns] = user_interests_df[bool_columns].replace({True: 1, False: 0})

        cosine_sim_list = []
        for df_name, df in df_dict.items():
            print(f"\nDataFrame: {df_name}")
            # print(df)
            path = 'resources/'+df_name+".csv"
            df.to_csv(path, index=False)

            df.drop(
                ['location', 'latitude', 'longitude',  'work_at', 'dob', 'children', 'diet', 'drink', 'education', 'exercise', 'gender',
                 'global', 'personality', \
                 'pets', 'preferance', 'relation_goal', 'sleeping', 'smoke', 'star_sign', 'status', 'vaccinated', 'age'],
                axis=1, inplace=True)

            merged_df = df.merge(user_interests_df, left_on='id', right_on='userid', how='left')
            merged_df.drop(columns='userid', inplace=True)


            scaler = MinMaxScaler(feature_range=(0, 1))
            normalized_data = scaler.fit_transform(merged_df)
            cosine_sim = cosine_similarity(normalized_data)
            cosine_sim_list.append(cosine_sim)
            print("cosine_sim"+str(len(cosine_sim)))
        return cosine_sim_list
