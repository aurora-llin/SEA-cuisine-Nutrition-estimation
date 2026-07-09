
# database.py

# Simple Google Sheets database reader for your VS Code project.
# Install pandas:
#    pip install pandas

import pandas as pd

SHEET_ID = "1B0xJ4iryY_va2HvGJqkn9OhfMRhugaareigbXowkN0c"

DISH_NUTRITION_GID = "0"
DISH_INGREDIENT_MANIFEST_GID = "2040394004"
INGREDIENTS_NUTRITION_GID = "479037523"


# Create the CSV export link for one Google Sheet tab.
def get_csv_url(gid):
    return f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={gid}"


# Read the Dish nutrition tab into a pandas DataFrame.
def read_dish_nutrition():
    url = get_csv_url(DISH_NUTRITION_GID)
    return pd.read_csv(url)


# Read the Dish ingredient manifest tab into a pandas DataFrame.
def read_dish_ingredient_manifest():
    url = get_csv_url(DISH_INGREDIENT_MANIFEST_GID)
    return pd.read_csv(url)


# Read the Ingredients nutrition tab into a pandas DataFrame.
def read_ingredients_nutrition():
    url = get_csv_url(INGREDIENTS_NUTRITION_GID)
    return pd.read_csv(url)


# Print one DataFrame in a simple CSV format.
def print_as_csv(df):
    print(df.to_csv(index=False))


# Print the Dish nutrition tab.
def print_dish_nutrition():
    df = read_dish_nutrition()
    print_as_csv(df)


# Print the Dish ingredient manifest tab.
def print_dish_ingredient_manifest():
    df = read_dish_ingredient_manifest()
    print_as_csv(df)


# Print the Ingredients nutrition tab.
def print_ingredients_nutrition():
    df = read_ingredients_nutrition()
    print_as_csv(df)


# Print all three tabs one by one.
def print_all_database():
    print("Dish nutrition")
    print_dish_nutrition()

    print("Dish ingredient manifest")
    print_dish_ingredient_manifest()

    print("Ingredients nutrition")
    print_ingredients_nutrition()


# Run this file directly to test if the database can be read.
if __name__ == "__main__":
    print_all_database()
