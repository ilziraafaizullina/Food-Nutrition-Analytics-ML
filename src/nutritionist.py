#!/usr/bin/env python3

import sys

from recipes import NutritionistApp


def parse_arguments():
    """
        Метод парсит аргументы командной строки
    """

    if len(sys.argv) < 2:
        return []

    raw = " ".join(sys.argv[1:]).strip()

    if not raw:
        return []

    return [item.strip() for item in raw.split(",") if item.strip()]


def format_forecast_output(prediction):
    """
        Метод преобразует предсказанный класс блюда в текст для вывода
    """

    if prediction == "bad":
        return (
            "You might find it tasty, but in our opinion, "
            "it is a bad idea to have a "
            "dish with that list of ingredients."
        )

    if prediction == "great":
        return "In our opinion, it is a great set of ingredients for a dish."

    return "In our opinion, this set of ingredients is so-so for a dish."


def format_nutrition_output(nutrition):
    """
        Метод форматирует данные о питательной ценности ингредиентов
    """

    if not nutrition:
        return "No nutrition facts found."

    lines = []

    for ingredient, nutrients in nutrition.items():
        lines.append(ingredient.capitalize())

        for nutrient_name, nutrient_value in nutrients.items():
            lines.append(
                f"{nutrient_name} - {nutrient_value:.0f}% of Daily Value"
            )

        lines.append("")

    return "\n".join(lines).strip()


def format_similar_recipes_output(similar_recipes):
    """
        Метод форматирует список похожих рецептов для вывода
    """

    if not similar_recipes:
        return "There are no similar recipes."

    lines = []

    for recipe in similar_recipes:
        lines.append(
            f"- {recipe['title']}, rating: {recipe['rating']}, URL:\n"
            f"{recipe['url']}"
        )

    return "\n".join(lines)


def format_menu_recipe(recipe, nutrient_limit=10):
    """
        Метод форматирует один рецепт дневного меню для вывода
    """

    lines = []

    lines.append(f"{recipe['title']} (rating: {recipe['rating']:.3f})")
    lines.append("Ingredients:")

    for ingredient in recipe["ingredients"]:
        lines.append(f"- {ingredient}")

    lines.append("Nutrients:")

    shown = 0
    sorted_nutrients = recipe["nutrients"].sort_values(ascending=False)

    for nutrient_name, nutrient_value in sorted_nutrients.items():
        if nutrient_value <= 0:
            continue

        lines.append(f"- {nutrient_name.lower()}: {nutrient_value:.0f}%")
        shown += 1

        if shown >= nutrient_limit:
            break

    lines.append(f"URL: {recipe['url']}")

    return "\n".join(lines)


def format_daily_menu(menu):
    """
        Метод форматирует дневное меню для вывода в консоль
    """

    if menu is None:
        return "No valid daily menu found."

    parts = []

    parts.append("BREAKFAST")
    parts.append("---------------------")
    parts.append(format_menu_recipe(menu["breakfast"]))
    parts.append("")

    parts.append("LUNCH")
    parts.append("---------------------")
    parts.append(format_menu_recipe(menu["lunch"]))
    parts.append("")

    parts.append("DINNER")
    parts.append("---------------------")
    parts.append(format_menu_recipe(menu["dinner"]))

    return "\n".join(parts)


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--daily-menu":
        app = NutritionistApp()
        menu = app.create_daily_menu(n_trials=10000)
        print(format_daily_menu(menu))
        return

    ingredients = parse_arguments()

    if not ingredients:
        print("Usage:")
        print("./nutritionist.py milk, honey, jam")
        print("./nutritionist.py --daily-menu")
        return

    app = NutritionistApp()
    result = app.analyze(ingredients)

    invalid_ingredients = result["invalid_ingredients"]
    valid_ingredients = result["valid_ingredients"]

    if invalid_ingredients:
        missing = ", ".join(invalid_ingredients)
        print(
            f"The following ingredients are missing in our database: {missing}"
        )

        if not valid_ingredients:
            return

        print("We will continue with the recognized ingredients only.\n")

    print("I. OUR FORECAST")
    if result["prediction"] is None:
        print("Forecast is unavailable.")
    else:
        print(format_forecast_output(result["prediction"]))

    print("\nII. NUTRITION FACTS")
    print(format_nutrition_output(result["nutrition"]))

    print("\nIII. TOP-3 SIMILAR RECIPES:")
    print(format_similar_recipes_output(result["similar_recipes"]))


if __name__ == "__main__":
    main()
