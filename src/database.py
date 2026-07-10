
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

def get_dish_nutrition_row(dish_id):
    df = read_dish_nutrition()
    row = df[df["dish_id"] == dish_id]
    return row.iloc[0] if not row.empty else None

def get_dish_manifest(dish_id):
    df = read_dish_ingredient_manifest()
    return df[df["dish_id"] == dish_id].reset_index(drop=True)

def get_ingredient_row(ingredient_id):
    df = read_ingredients_nutrition()
    row = df[df["ingredient_id"] == ingredient_id]
    return row.iloc[0] if not row.empty else None

def compute_hidden_extras(dish_id):
    """The untracked oil/salt/sauce residual: dish total minus itemized manifest total."""
    dish = get_dish_nutrition_row(dish_id)
    manifest = get_dish_manifest(dish_id)
    if dish is None or manifest.empty:
        return None
    return {
        "calories_kcal": dish["calories_kcal"] - manifest["cal_contrib"].sum(),
        "protein_g": dish["protein_g"] - manifest["protein_g_contrib"].sum(),
        "carbs_g": dish["carbs_g"] - manifest["carbs_g_contrib"].sum(),
        "fat_g": dish["fat_g"] - manifest["fat_g_contrib"].sum(),
        "fiber_g": dish["fiber_g"] - manifest["fiber_g_contrib"].sum(),
    }

def compute_ingredient_contrib(ingredient_id, grams):
    """Nutrition for a given gram amount of one ingredient, from the per-100g table."""
    row = get_ingredient_row(ingredient_id)
    if row is None:
        return None
    factor = grams / 100.0
    return {
        "calories_kcal": row["calories_kcal"] * factor,
        "protein_g": row["protein_g"] * factor,
        "carbs_g": row["carbs_g"] * factor,
        "fat_g": row["fat_g"] * factor,
        "fiber_g": row["fiber_g"] * factor,
    }

# Run this file directly to test if the database can be read.
if __name__ == "__main__":
    print_all_database()
