def fill_nan(df):
    # fill NaN in df sets and categorize columns datatype
    # adding missing column for the missing imputed values
    for column in df.columns:
        if column in numerical_feature_names:
            df[column+'_missing'] = df[column].isnull()
            mean = np.nanmean(df[column].values)
            df.loc[df[column].isnull(), column] = mean
        else:
            df.loc[df[column].isnull(), column] = "unknown"
    return df

# Because otherwise the feature will be perfectly
# linear and the regression will not be able to understand the different parameters
def ensure_at_least_two_values_in_columns(df):
    number_un = df.nunique() >= 2

    columns_with_at_least_two_entries = df.columns[number_un == True]

    df = df[columns_with_at_least_two_entries]

    return df

### remove similar features by only considering the most signaficant feature for similiar features
def _features_explaining_output_without_similiar_features(dataframe_cleaned_of_nan, lower_threshold=0.65):
    """
    removes the features too similiar too eachother but keeps one of these features to represent the output.
    This is choosen by explaintion of the output
    """
    pairwise_explanation = pairwise_r2(dataframe_cleaned_of_nan)
    upper_threshold = 0.9999
    ## now we are not interested in the y_column, in the pairwise matrix, this is why we take it from both axis
    pairwise_explanation.drop(self.y_output, axis=1, inplace=True)
    pairwise_explanation.drop(self.y_output, axis=0, inplace=True)

    features_that_are_similiar = pairwise_explanation.applymap(lambda x: x if (x > lower_threshold and x < upper_threshold) else None)

    feature_columns = []
    columns_remove_due_to_similiarity = []
    cols = features_that_are_similiar.columns

    cols_similiarity = defaultdict()
    for col in cols:
        similiar_features_to_column = features_that_are_similiar.unstack().dropna()[col].index.tolist()
        cols_similiar_to_eachother = [col] + similiar_features_to_column


        cols_similiar = defaultdict()
        ## need more than feature to be seperated
        if len(cols_similiar_to_eachother) > 1:
            explanation_of_output_from_similiar_features = r2_to_output(dataframe_cleaned_of_nan[self.y_output], dataframe_cleaned_of_nan[cols_similiar_to_eachother])

            explanation_of_output = max(explanation_of_output_from_similiar_features)
            ## TODO: hardcoded value
            if explanation_of_output < 0.95:
                index_of_feature_to_keep = explanation_of_output_from_similiar_features.index(explanation_of_output)
                feature_to_keep = cols_similiar_to_eachother.pop(index_of_feature_to_keep)
                feature_columns.append(feature_to_keep)
                columns_remove_due_to_similiarity.append(cols_similiar_to_eachother)

                cols_similiarity[feature_to_keep] = cols_similiar_to_eachother

        if len(cols_similiar_to_eachother) == 1:
            if not dataframe_cleaned_of_nan[cols_similiar_to_eachother].empty:
                explanation_of_output = r2_to_output(dataframe_cleaned_of_nan[self.y_output], dataframe_cleaned_of_nan[cols_similiar_to_eachother])
            else:
                explanation_of_output = [0]
            if explanation_of_output[0] > 0.95:
                columns_remove_due_to_similiarity.append(cols_similiar_to_eachother)
                cols_similiarity[cols_similiar_to_eachother[0]] = cols_similiar_to_eachother

    self.save_processed_dataset(self.dataset_data, cols_similiarity=cols_similiarity)

    # same feature can be explaining similiar attributes
    columns_remove_due_to_similiarity = [item for sublist in columns_remove_due_to_similiarity for item in sublist]
    cols_similiar_to_eachother = list(set(cols_similiar_to_eachother))
    feature_columns = list(set(feature_columns))
    columns_explaining_output = dataframe_cleaned_of_nan.drop(columns_remove_due_to_similiarity, axis=1).columns.tolist()
    columns_explaining_output.remove(self.y_output[0])

    if self.mandatory_features:
        for col in self.mandatory_continous_feature_columns:
            if col not in columns_explaining_output:
                columns_explaining_output.append(col)
    self.save_processed_dataset(self.dataset_data, continuous_feature_columns=columns_explaining_output)
    return dataframe_cleaned_of_nan[columns_explaining_output]



