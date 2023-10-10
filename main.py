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
        if names.count(name) > 1:
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
"""
import pandas as pd

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

