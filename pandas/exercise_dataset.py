import pandas as pd

DATASET = "lib_scratchpad/pandas/final_exercise_dataset.csv"

# Read the data
exercise_dataset = pd.read_csv(DATASET)

# Inspect the data
# This shows by rows, not columns
print(exercise_dataset.head(10))
print(exercise_dataset.tail(10))

# See all the columns
print(exercise_dataset.columns)

# Filtering
# List all te unique values in bodyPart column
print(exercise_dataset["bodyPart"].unique())
# Filter only lower legs and upper legs from bodyPart
target_bodypart = ("upper legs", "lower legs")
legs_exercises = exercise_dataset[(exercise_dataset["bodyPart"].isin(target_bodypart)) & (exercise_dataset["difficulty"]=="Advanced")]
print(legs_exercises.head())
# Temporarily adjust the option to print all the rows
with pd.option_context("display.max_columns", None):
    print(legs_exercises.head())
# Print only with certain column names
column_names_to_show = ["name", "equipment", "difficulty", "secondaryMuscles"]
print("============================")
print(legs_exercises[column_names_to_show])

# Selecting
# Print first row
print("============================")
first_row = legs_exercises.iloc[0]
print(first_row)
# Print first 3 rows, first 3 columns
specific_row = legs_exercises.iloc[0:2,0:3]
print("============================")
print(specific_row)