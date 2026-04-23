import os
import joblib
import numpy as np
import pandas as pd


class DataRepository:
    """
        Данный класс выполняет препроцессинг и подготовку данных для работы
    """

    META_COLS = {"title", "url", "rating", "breakfast", "lunch", "dinner"}

    def __init__(
        self,
        model_path="./data/best_model.pkl",
        nutrition_path="./data/nutrition_data_by_fdc_id.csv",
        recipes_path="./data/recipes_with_urls.csv"
    ):
        self.model_path = model_path
        self.nutrition_path = nutrition_path
        self.recipes_path = recipes_path

        self.model = None
        self.recipes_df = None
        self.nutrition_df = None
        self.ingredient_columns = None
        self.nutrient_columns = None

        self._load_all()

    def _load_all(self):
        """
            Метод просто запускает по очереди тре метода ниже
        """

        self._load_model()
        self._load_recipes()
        self._load_nutrition()

    def _load_model(self):
        """
            Метод загружает модель
        """

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        self.model = joblib.load(self.model_path)

    def _load_recipes(self):
        """
            Метод загружает данные о рецептах
            И формирует датафрейм с ингредиентами
            Для предсказания
        """

        if not os.path.exists(self.recipes_path):
            raise FileNotFoundError(
                f"Recipes file not found: {self.recipes_path}"
            )

        self.recipes_df = pd.read_csv(self.recipes_path)

        self.recipes_df["title"] = (
            self.recipes_df["title"]
                .astype(str)
                .str.strip()
        )

        self.recipes_df["url"] = (
            self.recipes_df["url"]
                .astype(str)
                .str.strip()
        )

        self.ingredient_columns = [
            col for col in self.recipes_df.columns
            if col not in self.META_COLS
        ]

    def _load_nutrition(self):
        """
            Метод загружает данные о нутриентах
            И формирует датафрейм с питательными веществами
            Которые отображены в виде процента суточной нормы
        """

        if not os.path.exists(self.nutrition_path):
            raise FileNotFoundError(
                f"Nutrition file not found: {self.nutrition_path}"
            )

        self.nutrition_df = pd.read_csv(self.nutrition_path)

        self.nutrition_df["ingredient"] = (
            self.nutrition_df["ingredient"]
                .astype(str)
                .str.strip()
                .str.lower()
        )

        excluded = {"ingredient", "fdc_id", "description"}

        self.nutrient_columns = [
            col for col in self.nutrition_df.columns
            if col not in excluded
        ]


class IngredientProcessor:
    """
        Данный класс подготавливает признаки для модели
    """

    def __init__(self, ingredient_columns):
        self.ingredient_columns = ingredient_columns
        self.known_ingredients = {
            str(col).strip().lower()
            for col in ingredient_columns
        }

    def normalize_ingredient(self, ingredient):
        return str(ingredient).strip().lower()

    def parse_ingredients(self, raw_ingredients):
        """
            Метод парсит ингредиенты и возвращает их список
        """

        if isinstance(raw_ingredients, str):
            ingredients = raw_ingredients.split(",")
        else:
            ingredients = raw_ingredients

        return [
            self.normalize_ingredient(ingredient)
            for ingredient in ingredients
            if str(ingredient).strip()
        ]

    def validate_ingredients(self, ingredients):
        """
            Метод проверяет какие ингредиенты есть в базе
        """

        normalized = self.parse_ingredients(ingredients)

        valid = []
        invalid = []

        for ingredient in normalized:
            if ingredient in self.known_ingredients:
                valid.append(ingredient)
            else:
                invalid.append(ingredient)

        return valid, invalid

    def build_feature_row(self, ingredients, feature_columns):
        """
            Метод возвращает датафрейм с признаками для модели
        """

        ingredient_set = set(self.parse_ingredients(ingredients))

        row = {
            col: 1.0 if str(col).strip().lower() in ingredient_set else 0.0
            for col in feature_columns
        }

        return pd.DataFrame([row], columns=feature_columns)


class RatingPredictor:
    """
        Данный класс предсказывает рейтинг рецепта
    """

    CLASS_MAP = {
        0: "bad",
        1: "so-so",
        2: "great"
    }

    def __init__(self, model):
        self.model = model

    def get_feature_columns(self, fallback_columns):
        """
            Метод возвращает список колонок которые ждет модель
        """

        if hasattr(self.model, "feature_names_in_"):
            return list(self.model.feature_names_in_)

        return fallback_columns

    def predict(self, X):
        """
            Метод принимает подвыборку Х и делает предсказание
        """

        prediction = self.model.predict(X)[0]

        if isinstance(prediction, (int, np.integer)):
            prediction = int(prediction)

            if prediction not in self.CLASS_MAP:
                raise ValueError(
                    f"Unknown prediction label: {prediction}"
                )

            return self.CLASS_MAP[prediction]

        if prediction in {"bad", "so-so", "great"}:
            return str(prediction)

        raise ValueError(f"Unknown prediction label: {prediction}")


class NutritionService:
    """
        Класс возвращает информацию о питательных веществах
    """

    def __init__(self, nutrition_df, nutrient_columns):
        self.nutrition_df = nutrition_df
        self.nutrient_columns = nutrient_columns

        self.nutrition_lookup = self.nutrition_df.set_index("ingredient")

    def get_nutrition(self, ingredients):
        """
            Метод принимает список ингредиентов и возвращает словарь
            с питательными веществами
        """

        result = {}

        for ingredient in ingredients:
            if ingredient not in self.nutrition_lookup.index:
                continue

            row = self.nutrition_lookup.loc[ingredient, self.nutrient_columns]

            nutrients = {
                col: float(row[col])
                for col in self.nutrient_columns
                if pd.notna(row[col]) and float(row[col]) > 0
            }

            result[ingredient] = nutrients

        return result


class SimilarRecipesFinder:
    """
        Класс находит похожие рецепты по ингредиентам
    """

    def __init__(self, recipes_df, ingredient_columns):
        self.recipes_df = recipes_df
        self.ingredient_columns = ingredient_columns
        self.recipe_matrix = (
            self.recipes_df[self.ingredient_columns]
                .astype(float)
        )

    def find_similar(self, ingredients, top_n=3):
        """
            Метод сравнивает ингредиенты пользователя с каждым рецептом из 
            датасета, вычисляет степень похожести по коэффициенту Жаккара и 
            отбирает лучшие подходящие рецепты
        """

        ingredient_set = set(ingredients)

        if not ingredient_set:
            return []

        query_size = len(ingredient_set)

        query_vector = np.array(
            [float(col in ingredient_set) for col in self.ingredient_columns]
        )

        recipe_matrix = self.recipe_matrix.to_numpy()

        intersection = np.minimum(recipe_matrix, query_vector).sum(axis=1)
        union = np.maximum(recipe_matrix, query_vector).sum(axis=1)

        similarity = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=float),
            where=union > 0
        )

        if query_size <= 4:
            candidate_mask = (intersection >= 1) & (similarity >= 0.2)
        else:
            candidate_mask = (intersection >= 2) & (similarity >= 0.3)

        if not candidate_mask.any():
            return []

        candidate_indices = np.where(candidate_mask)[0]
        candidate_scores = similarity[candidate_indices]

        sorted_order = np.argsort(candidate_scores)[::-1][:top_n]
        best_indices = candidate_indices[sorted_order]

        results = []

        for idx in best_indices:
            row = self.recipes_df.iloc[idx]
            results.append({
                "title": row["title"],
                "rating": row["rating"],
                "url": row["url"],
                "similarity": float(similarity[idx])
            })

        return results


class MenuCreator:
    """
        Этот класс отвечает за случайную генерацию дневного меню
    """

    META_COLS = {"title", "url", "rating", "breakfast", "lunch", "dinner"}

    def __init__(
        self,
        recipes_df,
        nutrition_df,
        nutrient_columns,
        ingredient_columns
    ):
        self.recipes_df = recipes_df.copy()
        self.nutrition_df = nutrition_df.copy()
        self.nutrient_columns = nutrient_columns
        self.ingredient_columns = ingredient_columns

        self.nutrition_lookup = self.nutrition_df.set_index("ingredient")

    def _get_recipe_ingredients(self, recipe_row):
        """
            Метод возвращает список ингредиентов, входящих в рецепт
        """

        ingredients = []

        for ingredient in self.ingredient_columns:
            if ingredient in recipe_row and float(recipe_row[ingredient]) > 0:
                ingredients.append(ingredient)

        return ingredients

    def _sum_recipe_nutrients(self, ingredients):
        """
            Метод считает суммарную питательную ценность рецепта
        """

        total = pd.Series(
            data=np.zeros(len(self.nutrient_columns)),
            index=self.nutrient_columns,
            dtype=float
        )

        for ingredient in ingredients:
            if ingredient not in self.nutrition_lookup.index:
                continue

            row = self.nutrition_lookup.loc[ingredient, self.nutrient_columns]
            row = pd.to_numeric(row, errors="coerce").fillna(0.0)
            total = total.add(row, fill_value=0)

        return total

    def _prepare_category(self, category_name):
        """
            Метод подготавливает список рецептов для заданной категории
        """

        if category_name not in self.recipes_df.columns:
            raise ValueError(f"Column '{category_name}' is missing in recipes data.")

        category_df = (
            self.recipes_df[self.recipes_df[category_name] == 1]
                .copy()
        )

        prepared_rows = []

        for _, row in category_df.iterrows():
            ingredients = self._get_recipe_ingredients(row)
            nutrients = self._sum_recipe_nutrients(ingredients)

            prepared_rows.append({
                "title": row["title"],
                "url": row["url"],
                "rating": float(row["rating"]),
                "ingredients": ingredients,
                "nutrients": nutrients
            })

        return prepared_rows

    def _is_valid_menu(self, breakfast, lunch, dinner):
        """
            Метод проверяет, является ли меню допустимым по нутриентам
        """

        total_nutrients = (
            breakfast["nutrients"]
            + lunch["nutrients"]
            + dinner["nutrients"]
        )

        return bool((total_nutrients <= 100).all())

    def _menu_score(self, breakfast, lunch, dinner):
        """
            Метод вычисляет качество меню
        """
        
        return (
            breakfast["rating"]
            + lunch["rating"]
            + dinner["rating"]
        )

    def create_daily_menu(self, n_trials=10000):
        """
            Метод генерирует случайное дневное меню
        """

        breakfasts = self._prepare_category("breakfast")
        lunches = self._prepare_category("lunch")
        dinners = self._prepare_category("dinner")

        if not breakfasts or not lunches or not dinners:
            return None

        best_menu = None
        best_score = -1

        for _ in range(n_trials):
            breakfast = breakfasts[np.random.randint(len(breakfasts))]
            lunch = lunches[np.random.randint(len(lunches))]
            dinner = dinners[np.random.randint(len(dinners))]

            if not self._is_valid_menu(breakfast, lunch, dinner):
                continue

            score = self._menu_score(breakfast, lunch, dinner)

            if score > best_score:
                best_score = score
                best_menu = {
                    "breakfast": breakfast,
                    "lunch": lunch,
                    "dinner": dinner
                }

        return best_menu


class NutritionistApp:
    """
        Класс объединяет все основные части приложения
    """

    def __init__(
        self,
        model_path="./data/best_model.pkl",
        nutrition_path="./data/nutrition_data_by_fdc_id.csv",
        recipes_path="./data/recipes_with_urls.csv"
    ):
        self.repository = DataRepository(
            model_path=model_path,
            nutrition_path=nutrition_path,
            recipes_path=recipes_path
        )

        self.ingredient_processor = IngredientProcessor(
            self.repository.ingredient_columns
        )

        self.predictor = RatingPredictor(self.repository.model)

        self.nutrition_service = NutritionService(
            self.repository.nutrition_df,
            self.repository.nutrient_columns
        )

        self.similar_finder = SimilarRecipesFinder(
            self.repository.recipes_df,
            self.repository.ingredient_columns
        )

        self.menu_creator = MenuCreator(
            recipes_df=self.repository.recipes_df,
            nutrition_df=self.repository.nutrition_df,
            nutrient_columns=self.repository.nutrient_columns,
            ingredient_columns=self.repository.ingredient_columns,
        )

    def analyze(self, raw_ingredients):
        """
            Метод выполняет полный анализ списка ингредиентов пользователя
        """

        valid_ingredients, invalid_ingredients = (
            self.ingredient_processor.validate_ingredients(raw_ingredients)
        )

        prediction = None

        if valid_ingredients:
            feature_columns = self.predictor.get_feature_columns(
                self.repository.ingredient_columns
            )

            X = self.ingredient_processor.build_feature_row(
                valid_ingredients,
                feature_columns
            )

            prediction = self.predictor.predict(X)

        nutrition = self.nutrition_service.get_nutrition(valid_ingredients)

        similar_recipes = self.similar_finder.find_similar(
            valid_ingredients,
            top_n=3
        )

        return {
            "valid_ingredients": valid_ingredients,
            "invalid_ingredients": invalid_ingredients,
            "prediction": prediction,
            "nutrition": nutrition,
            "similar_recipes": similar_recipes
        }

    def create_daily_menu(self, n_trials=10000):
        return self.menu_creator.create_daily_menu(n_trials=n_trials)
