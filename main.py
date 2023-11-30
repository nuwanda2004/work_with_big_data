# work number 1
#A)
"""x = 2
y = ((1-x)/(1+x)) - 1.6*x**3 * ((x+7)**(1/3))
print(y)
"""
#B)
"""
number = input("Write the number:")
if int(number) >= 14000:
    print("IB")
if 6000 <= int(number) <= 14000:
    print("II")
if 2000 <= int(number) <= 6000:
    print("III")
if 200 <= int(number) <= 2000:
    print("IV")
if int(number) < 200:
    print("V")
"""

# work number 2
#A)
"""
def is_prime(number):
    if number <= 1:
        return False

    for i in range(2, int(number ** 0.5) + 1):
        if number % i == 0:
            return False

    return True

number = int(input("Введите число: "))
if is_prime(number):
    print("Prime number")
else:
    print("Not prime number")
"""
#B)
"""
count = 0
previous_number = float('inf')  # Инициализируем предыдущее число как бесконечность

while True:
    number = float(input("Введите число (0 для завершения): "))

    if number == 0:
        break

    if number > previous_number:
        count += 1

    previous_number = number

print(f"Количество чисел, больших предыдущего: {count}")
"""
# work number 3
"""
text = "Maximov Alexey Maximovich ICTMS 2-4"
words = text.split()
finish = []
for i in words:
    a = len(i)
    finish.append(a)
print(finish)
"""
#work number 4
#A)
"""
dict = {"Building1": 20, "Building2": 120, "Building3": 80}
for name, high in dict.items():
    if high < 30:
        print("здание повышенной этажности")
    if 30 < high < 50:
        print("1 категория многоэтажных зданий")
    if 50 < high < 75:
        print("2 категория")
    if 75 < high < 100:
        print("3 категория")
    if high > 100:
        print("высотное здание")
"""

#B)
"""
import json

clients_count = 0
names = []
k = 0
# Открываем JSON файл для чтения (режим 'r' - read)
with open('data.json', 'r') as json_file:
    # Загружаем данные из JSON файла
    data = json.load(json_file)

    # Получаем список событий из поля "events_data"
    events_data = data.get("events_data", [])

    # Перебираем каждое событие
    for entry in events_data:
        # Проверяем наличие ключа "category" и его значение
        if "category" in entry and entry["category"] in ["page", "report"]:
            names.append(entry["client_id"])
    for name in names:
        if names.count(name) == 1:
            k += 1
print("Количество клиентов с действиями в категориях 'page' или 'report':", k)

"""
#work number 5
"""
import numpy as np
import random

n = 4
m = 4

A = []
for i in range(n):
    row = [random.randint(0,100) for j in range(m)]
    A.append(row)

print("Исходная матрица:")
for row in A:
    print(row)

max_abs_value = max(abs(element) for row in A for element in row)

B = [[element / max_abs_value for element in row] for row in A]
B_rounded = [[round(element, 2) for element in row] for row in B]

print("Новая матрица (деление на", max_abs_value, "):")
for row in B_rounded:
    print(row)

variance_B = np.var(B_rounded)
print("Дисперсия элементов новой матрицы", round(variance_B, 2))

mean_B = sum(sum(row) for row in B_rounded) / (n*m)
sum_squares_diff = sum((element - mean_B) ** 2 for row in B_rounded for element in row)
variance_B2 = sum_squares_diff / (n*m)
print("Дисперсия элементов матрицы через программирование формулы", round(variance_B2, 2))
"""
#work number 6

import pandas as pd
"""
# Загрузка данных из файла
df = pd.read_csv('football.csv')

# Находим средний уровень хладнокровия
average_composure = df['Composure'].mean()

# Фильтруем игроков с уровнем хладнокровия выше среднего
composed_players = df[df['Composure'] > average_composure]

# Находим количество забитых пенальти
penalties_scored = composed_players['Penalties'].sum()

# Выводим результаты
print(f"Игроки из страны с уровнем хладнокровия выше среднего: {composed_players['Nationality'].unique()}")
print(f"Общее количество забитых пенальти: {penalties_scored}")
"""
#work number 7
#A)
"""
import pandas as pd

# Загрузим данные из файла
df = pd.read_csv('StudentsPerformance.csv')

# Подсчитаем уникальные значения в столбце "parental level of education"
unique_education_levels = df['parental level of education'].nunique()

# Выведем результат
print(f"Количество различных вариантов значений в столбце 'parental level of education': {unique_education_levels}")
"""
#B)
"""
import pandas as pd

# Загрузим данные из файла
df = pd.read_csv('StudentsPerformance.csv')

# Создаем функцию
def custom_function(row):
    education_prefix = row['parental level of education'][:5]
    lunch_suffix = row['lunch'][-5:]
    return education_prefix + lunch_suffix

# Применяем функцию к датафрейму и записываем результат в новый столбец 'new_column'
df['new_column'] = df.apply(lambda row: custom_function(row), axis=1)

# Выведем обновленный датафрейм
print(df)
"""
#work number 8
"""
import pandas as pd

# Загрузим данные из файла
df = pd.read_csv('films.csv')

# Создаем столбец 'profit_loss', который представляет собой разность между сборами и бюджетом фильма
df['profit_loss'] = df['revenue'] - df['budget']

# Фильтруем фильмы за период с 2012 по 2014 год
filtered_df = df[(df['release_year'] >= 2012) & (df['release_year'] <= 2014)]

# Находим самый убыточный фильм
most_unprofitable_film = filtered_df.loc[filtered_df['profit_loss'].idxmin()]

# Выводим информацию о самом убыточном фильме
print("Самый убыточный фильм:")
print(most_unprofitable_film[['original_title', 'profit_loss']])
"""
#work number 9
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка данных из файла tips.csv
tips = pd.read_csv('tips.csv')

# Создание графика
plt.figure(figsize=(12, 8))

# Используем sns.barplot для столбчатой диаграммы с разделением по курящим и некурящим
ax = sns.barplot(x='day', y='tip', hue='smoker', data=tips, palette={'No': 'skyblue', 'Yes': 'salmon'}, ci=None)

# Настройка графика
plt.title('Average Tips by Day and Smoking Status', fontsize=16)
plt.xlabel('Day of the Week', fontsize=14)
plt.ylabel('Average Tip Amount', fontsize=14)
plt.legend(title='Smoker', loc='upper right', labels=['Non-Smoker', 'Smoker'], fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Добавление аннотации для пояснения графика
plt.annotate('Weekend', xy=(4.5, 3), xytext=(5.5, 4), arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=12)

# Добавление дополнительных элементов для лучшего восприятия
sns.despine(trim=True, left=True)

# Исправление наложения табличек
ax.legend(title='Smoker', loc='upper right', labels=['Non-Smoker', 'Smoker'])

# Вывод графика
plt.show()

"""
#work number 10
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Загрузка данных
file_path = 'mycar_lin.csv'  # Замени путь_к_файлу на реальный путь к файлу
data = pd.read_csv(file_path)

# Посмотрим на первые несколько строк данных
print(data.head())

# Выбор признаков и целевой переменной
X = data[['Speed']]
y = data['Stopping_dist']

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

# Создание и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Прогнозирование на тестовом наборе
y_pred = model.predict(X_test)

# Оценка качества модели
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Визуализация результатов
plt.scatter(X_test, y_test, color='black', label='Actual')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Predicted')
plt.xlabel('Speed')
plt.ylabel('Stopping Distance')
plt.title('Linear Regression Model')
plt.legend()
plt.show()

# Вывод характеристик модели
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R2): {r2}')

# Вывод коэффициентов модели
print('Intercept:', model.intercept_)
print('Coefficient:', model.coef_[0])


#work number 11 (last)
#task 1.3
import pandas as pd

df = pd.read_csv('cacao_flavors.csv')
df.columns = ['bar_id', 'company', 'specific_origin', 'ref', 'review_date', 'cocoa_percent', 'company_location', 'rating', 'bean_type', 'broad_origin']
df.to_csv('cacao_flavors.csv', index=False)
"""
#task 1.4.1
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('cacao_flavors.csv')
unique_bar_ids = df['bar_id'].nunique()
print(f"Количество уникальных значений в 'bar_id': {unique_bar_ids}")
plt.figure(figsize=(10, 6))
plt.bar(df['bar_id'], df['rating'], color='black')
plt.xlabel('Bar ID')
plt.ylabel('Rating')
plt.title('Rating by Bar ID')
plt.grid(True)
plt.show()
"""
#task 1.4.2
"""
company_counts = df['company'].value_counts()
selected_companies = company_counts[company_counts > 10]
filtered_df = df[df['company'].isin(selected_companies.index)]
print(filtered_df['company'].unique())
all_companies_count = len(df['company'].unique())
print(f"Всего компаний: {all_companies_count}")
filtered_companies_count = len(filtered_df['company'].unique())
print(f"Количество отфильтрованных компаний: {filtered_companies_count}")
"""
#task 1.4.3
"""
unique_regions_count = df['specific_origin'].nunique()
print(f"Количество оригинальных регионов: {unique_regions_count}")

specific_origin_counts = df['specific_origin'].value_counts()
more_than_10_count = (specific_origin_counts > 10).sum()
print(f"Количество значений specific_origin, встретившихся более 10 раз: {more_than_10_count}")
"""
#task 1.4.4
"""
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.histplot(df['ref'], bins=30, kde=True, color='skyblue')
plt.xlabel('REF')
plt.ylabel('Частота')
plt.title('Распределение данных REF')
plt.grid(True)
plt.show()

print(f"Максимальное значение REF: {df['ref'].max()}")
print(f"Минимальное значение REF: {df['ref'].min()}")
print(f"Количество пропусков REF: {df['ref'].isnull().sum()}")
"""
#task 1.4.5
"""
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(df['review_date'], bins=30, kde=True, color='salmon')
plt.xlabel('Review Date')
plt.ylabel('Частота')
plt.title('Распределение данных Review Date')
plt.grid(True)
plt.show()

print(f"Макс Review Date: {df['review_date'].max()}")
print(f"Мин Review Date: {df['review_date'].min()}")
print(f"Кол-во пропусков Review Date: {df['review_date'].isnull().sum()}")
"""
#task 1.4.6
"""
import matplotlib.pyplot as plt
import seaborn as sns
df['cocoa_percent'] = df['cocoa_percent'].str.rstrip('%').astype('float') / 100.0
plt.figure(figsize=(10, 6))
sns.histplot(df['cocoa_percent'], bins=30, kde=True, color='green')
plt.xlabel('Cocoa Percent')
plt.ylabel('Частота')
plt.title('Распределение данных Cocoa Percent')
plt.grid(True)
plt.show()
print(f"Макс Cocoa Percent: {df['cocoa_percent'].max()}")
print(f"МинCocoa Percent: {df['cocoa_percent'].min()}")
print(f"Кол-во пропусков Cocoa Percent: {df['cocoa_percent'].isnull().sum()}")
"""
#task 1.4.7
"""
total_countries = df['company_location'].nunique()
print(f"Всего стран в столбце company_location: {total_countries}")
more_10_reviews = df['company_location'].value_counts()
more_10_reviews = more_10_reviews[more_10_reviews > 10]
print(f"Количество стран с более чем 10 ревью: {len(more_10_reviews)}")
"""
#task 1.4.8
"""
import matplotlib.pyplot as plt
import seaborn as sns
# Построим ящик с усами (boxplot) для столбца Rating
# Ящик с усами - график, показывающий основные статистические характеристики данных, такие как медиана, квартили и выбросы.
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['rating'], color='orange')
plt.xlabel('rating')
plt.title('Ящик с усами для столбца Rating')
plt.show()
"""
#task 1.4.9
"""
missing_values = df['bean_type'].isnull().sum()
print(f"Кол-во пропусков в Bean Type: {missing_values}")
bean_type_counts = df['bean_type'].value_counts()
more_than_10 = bean_type_counts[bean_type_counts > 10]
print(f"Кол-во значений Bean Type, встретившихся > 10 раз: {len(more_than_10)}")

# Отфильтруем данные
filtered_df = df[df['bean_type'].isin(more_than_10.index)]
"""
#task 1.4.10.1
"""
missing_values = df['broad_origin'].isnull().sum()
print(f"Количество пропусков в столбце broad_origin: {missing_values}")
broad_origin_counts = df['broad_origin'].value_counts()
more_than_10 = broad_origin_counts[broad_origin_counts > 10]
print(f"Количество значений в столбце broad_origin, встретившихся более 10 раз: {len(more_than_10)}")
unique_values = df['broad_origin'].nunique()
print(f"Количество уникальных значений в столбце broad_origin: {unique_values}")

# Отфильтруем данные
filtered_df = df[df['broad_origin'].isin(more_than_10.index)]
"""
#1.4.10.2
"""
import seaborn as sns
import matplotlib.pyplot as plt

numeric_data = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Матрица корреляций числовых данных')
plt.show()
"""

#task 1.4.10.3
"""
import seaborn as sns
import matplotlib.pyplot as plt

categorical_columns = ['company_location', 'bean_type', 'broad_origin']

for column in categorical_columns:
    plt.figure(figsize=(12, 6))
    sns.countplot(x=column, data=df, palette='viridis')
    plt.title(f'Распределение переменной {column}')
    plt.xticks(rotation=45)
    plt.show()
"""
#main task number 11
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

file_path = 'cacao_flavors.csv'
data = pd.read_csv(file_path)

features = data[['ref']]
target = data['rating']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.9, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
plt.scatter(X_test, y_test, color='black', label='Actual')
plt.scatter(X_test, y_pred, color='blue', linewidth=3, label='Predicted')
plt.xlabel('Ref')
plt.ylabel('Rating')
plt.title('Linear Regression Model')
plt.legend()
plt.show()
"""
#liza variant 15
#work number 5
'''
import numpy as np
n = 5  
A = np.random.randint(1, 100, size=(n, n))
print("Исходная матрица:")
print(A)
for i in range(n):
    max_element_index = np.argmax(A[i, :])
    A[i, max_element_index], A[i, i] = A[i, i], A[i, max_element_index]
print("\nМатрица после обмена:")
print(A)
diagonal_elements = np.diagonal(A)
median_standard = np.median(diagonal_elements)
sorted_diagonal = np.sort(diagonal_elements)
length = len(sorted_diagonal)
if length % 2 == 0:
    median_formula = (sorted_diagonal[length // 2 - 1] + sorted_diagonal[length // 2]) / 2
else:
    median_formula = sorted_diagonal[length // 2]
print("\nМедиана (стандартная функция):", median_standard)
print("Медиана (программирование формулы):", median_formula)
'''
#work number 6
'''
import pandas as pd

# Загрузка данных
data = pd.read_csv("football.csv")

# Расчет среднего возраста игроков с реакцией выше средней и максимальным числом пенальти
above_avg_reactions = data[data['Reactions'] > data['Reactions'].mean()]
max_penalties = data[data['Penalties'] == data['Penalties'].max()]
avg_age_reactions_penalties = above_avg_reactions['Age'].mean()

# Расчет среднего возраста игроков с уровнем агрессии выше среднего и максимальной скоростью
above_avg_aggression = data[data['Aggression'] > data['Aggression'].mean()]
max_sprint_speed = data[data['SprintSpeed'] == data['SprintSpeed'].max()]
avg_age_aggression_speed = above_avg_aggression['Age'].mean()

# Расчет разницы между средними возрастами
age_difference = avg_age_reactions_penalties - avg_age_aggression_speed

print("Разница между средним возрастом игроков с реакцией выше средней и максимальным числом пенальти, и средним возрастом игроков с уровнем агрессии выше среднего и максимальной скоростью:", age_difference)
'''
#work number 7
#A)
'''
import pandas as pd

# Загрузка данных
data_students = pd.read_csv("StudentsPerformance.csv")

# Подсчет уникальных этнических групп
unique_ethnic_groups = data_students['race/ethnicity'].unique()

# Вывод результатов
print("Количество разных этнических групп:", len(unique_ethnic_groups))
print("Эти группы:", unique_ethnic_groups)

#B)
def process_string(input_string):
    # Подсчет количества пробелов
    spaces_count = input_string.count(' ')

    # Поиск индекса символа "―"
    dash_index = input_string.find('―')

    # Умножение индекса на количество пробелов и выделение соответствующей подстроки
    result = input_string[:spaces_count * dash_index]

    return result

# Применение функции к столбцу "parental level of education" через лямбда-функцию
data_students['processed_parental_level'] = data_students['parental level of education'].apply(lambda x: process_string(x))

# Вывод результата
print(data_students[['parental level of education', 'processed_parental_level']])
'''
#work number 8
'''
import pandas as pd

# Загрузка данных
data_films = pd.read_csv("films.csv")

# Подсчет количества фильмов для каждого режиссера
directors_counts = data_films['director'].value_counts()

# Получение режиссера с наибольшим количеством фильмов
most_films_director = directors_counts.idxmax()
num_films = directors_counts.max()

print("Режиссер снял больше всего фильмов:")
print(f"{most_films_director} - {num_films} фильмов")
'''
#karina v1n2
N = int(input("ВВедите объем цистерны: "))
m = int(input("Введите количество литров, доставляемые роботом: "))
k = 0
l = 0
while l <= N:
    l += m + k*m
    k+=1
print(k)
