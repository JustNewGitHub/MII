import csv
import numpy as nup
from numpy.random import choice
from numpy import genfromtxt
import datetime as dati
import os
import pandas as pads
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

def graphics():

    NewDataFrame = pads.read_csv('new_data.csv')
    numbers = NewDataFrame["number"]
    num_min = numbers.min()
    num_max = numbers.max()
    num_ave = numbers.mean()
    num_disp = numbers.var()
    num_std = numbers.std()
    num_median = numbers.median()
    num_mode = mode_val(numbers)
    salary = NewDataFrame["salary"]
    sal_min = salary.min()
    sal_max = salary.max()
    sal_ave = salary.mean()
    sal_disp = salary.var()
    sal_std = salary.std()
    sal_median = salary.median()
    sal_mode = mode_val(salary)
    projects = NewDataFrame["completed_projects"]
    pjt_min = projects.min()
    pjt_max = projects.max()
    pjt_ave = projects.mean()
    pjt_disp = projects.var()
    pjt_std = projects.std()
    pjt_median = projects.median()
    pjt_mode = mode_val(projects)

    plt.figure(figsize=(14, 10), dpi=80)
    plt.hlines(y=projects, xmin=0, xmax=salary, color='C0', alpha=0.4, linewidth=5)
    plt.gca().set(ylabel='Колличество проектов', xlabel='Заработная плата')
    plt.title('Зависимость числа проектов от заработной платы', fontdict={'size': 20})
    plt.show()

    data=[NewDataFrame["gender"].value_counts()["Муж"],NewDataFrame["gender"].value_counts()["Жен"]]
    plt.pie(data, labels=["Мужчины","Женщины"])
    plt.title("Распределение мужчин и женщин в фирме")
    plt.ylabel("")
    plt.show()

    data=NewDataFrame
    myLocator = mticker.MultipleLocator(4)
    plt.figure(figsize=(16, 10), dpi=80)
    plt.plot_date(data["start_date"],data["salary"])
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.ylabel('Заработная плата')
    plt.xlabel('Даты')
    plt.title("Изменение ЗП в течение всего времени")
    plt.show()

def mode_val(values):

    dict={}
    for elem in values:
        if elem in dict:
            dict[elem]+=1
        else:
            dict[elem]=1
    v = list(dict.values())
    k = list(dict.keys())

    return k[v.index(max(v))]

def generate_data():

    MyData = [["number", "full_name", "gender", "birth_date", "start_date", "division", "position", "salary",
               "completed_projects"]]
    Gender = ["Муж", "Жен"]
    Surnames = ["Антипов", "Бабаев", "Вавилов", "Галкин", "Данилин", "Евсюткин", "Жеглов", "Задорнов", "Ивачев",
                "Кабаков", "Лабутин",
                "Маврин", "Назаров", "Овсеев", "Павлов", "Райкин", "Савочкин", "Табаков", "Уваров", "Фандеев",
                "Хабалов", "Царёв", "Чадов",
                "Шаляпин", "Щукин", "Эвентов", "Юров", "Ягодин"]
    Initials = ["А", "Б", "В", "Г", "Д", "Ж", "З", "И", "К", "Л", "М", "Н", "Р", "С", "Т", "У", "Ф", "Э", "Ю", "Я"]
    Divisions = ["Отдел информационной безопасности", "Отдел разработки ПО", "Отдел контроля качества и процесов",
                 "Отдел развития ИТ", "Отдел поддержки ИТ"]
    Positions = [["Руководитель отдела информационной безопасности", "Специалист", "Начинающий специалист"],
                 ["Руководитель отдела разработки ПО", "Senior разработчик", "Middle разработчик"],
                 ["Руководитель отдела контроля качества и процесов", "Специалист", "Работник"],
                 ["Руководитель отдела развития ИТ", "Специалист", "Работник"],
                 ["Руководитель отдела поддержки ИТ", "Специалист", "Работник"]]

    for i in range(1, 1500):
        nup.random.seed(i)
        num = i
        full_name = ""
        gend = ""
        birth = ""
        start = ""
        div = ""
        pos = ""
        salary = 0
        completed_projects = 0

        gender = nup.random.randint(0, 2)
        gend = Gender[gender]
        if (gender != 0):
            full_name = Surnames[nup.random.randint(0, 27)] + "а " + Initials[nup.random.randint(0, 19)] + "." + Initials[
                nup.random.randint(0, 19)] + "."
        else:
            full_name = Surnames[nup.random.randint(0, 27)] + " " + Initials[nup.random.randint(0, 19)] + "." + Initials[
                nup.random.randint(0, 19)] + "."
        current_date = dati.date.today()
        year = current_date.year - nup.random.randint(0, 11)
        month = nup.random.randint(1, 13)
        day = nup.random.randint(1, 29)
        start = str(day) + "." + str(month) + "." + str(year)
        byear = year - nup.random.randint(18, 31)
        bmonth = nup.random.randint(1, 13)
        bday = nup.random.randint(1, 29)
        birth = str(bday) + "." + str(bmonth) + "." + str(byear)
        divis = nup.random.randint(0, 5)
        div = Divisions[divis]
        posit = choice([0, 1, 2], 1, [0.1, 0.3, 0.6])[0]
        pos = Positions[divis][posit]
        salary = nup.random.randint(10000, 20001) * (5 - divis) * (3 - posit)
        completed_projects = nup.random.randint(1, 21) * (3 - posit)

        MyData.append([num, full_name, gend, birth, start, div, pos, salary, completed_projects])

    for per in MyData:
        for param in per:
            print(param, end=" , ")
        print("")

    with open("new_data.csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
        for per in MyData:
            file_writer.writerow(per)

def nup_statistic():

    MyData = []

    with open('new_data.csv', mode="r", encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            data=row
            MyData.append([data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8]])

    for per in MyData:
        for param in per:
            print(param, end=" , ")
        print("")

    MyData=nup.array(MyData)
    numbers=nup.array(MyData[:,0])
    numbers=nup.delete(numbers, 0)
    numbers=[int(item) for item in numbers]
    num_min=nup.min(numbers)
    num_max=nup.max(numbers)
    num_ave= nup.average(numbers)
    num_disp=nup.var(numbers)
    num_std=nup.std(numbers)
    num_median=nup.median(numbers)
    num_mode=mode_val(numbers)
    salary=nup.array(MyData[:,7])
    salary=nup.delete(salary,0)
    salary = [int(item) for item in salary]
    sal_min=nup.min(salary)
    sal_max=nup.max(salary)
    sal_ave=nup.average(salary)
    sal_disp=nup.var(salary)
    sal_std=nup.std(salary)
    sal_median = nup.median(salary)
    sal_mode=mode_val(salary)
    projects=nup.array(MyData[:,8])
    projects = nup.delete(projects, 0)
    projects = [int(item) for item in projects]
    pjt_min = nup.min(projects)
    pjt_max = nup.max(projects)
    pjt_ave = nup.average(projects)
    pjt_disp = nup.var(projects)
    pjt_std = nup.std(projects)
    pjt_median = nup.median(projects)
    pjt_mode=mode_val(projects)
    print(numbers)

    print("")
    print("Статистические данные")
    print("")
    print("Для столбца номер: min="+str(num_min)+" ; max="+str(num_max)+" ; ave="+str(num_ave)+" ; disp="+str(num_disp)+" ; std="+str(num_std)+" ; median="+str(num_median)+" ; mode="+str(num_mode))
    print("Для столбца зарплата: min=" + str(sal_min) + " ; max=" + str(sal_max) + " ; ave=" + str(sal_ave) + " ; disp=" + str(sal_disp) + " ; std=" + str(sal_std) + " ; median=" + str(sal_median) + " ; mode=" + str(sal_mode))
    print("Для столбца проекты: min=" + str(pjt_min) + " ; max=" + str(pjt_max) + " ; ave=" + str(pjt_ave) + " ; disp=" + str(pjt_disp) + " ; std=" + str(pjt_std) + " ; median=" + str(pjt_median) + " ; mode=" + str(pjt_mode))

def pads_statistic():

    MyDataFrame = pads.read_csv('new_data.csv')
    numbers=MyDataFrame["number"]
    num_min = numbers.min()
    num_max = numbers.max()
    num_ave = numbers.mean()
    num_disp = numbers.var()
    num_std = numbers.std()
    num_median = numbers.median()
    num_mode = mode_val(numbers)
    salary = MyDataFrame["salary"]
    sal_min = salary.min()
    sal_max = salary.max()
    sal_ave = salary.mean()
    sal_disp = salary.var()
    sal_std = salary.std()
    sal_median = salary.median()
    sal_mode = mode_val(salary)
    projects = MyDataFrame["completed_projects"]
    pjt_min = projects.min()
    pjt_max = projects.max()
    pjt_ave = projects.mean()
    pjt_disp = projects.var()
    pjt_std = projects.std()
    pjt_median = projects.median()
    pjt_mode = mode_val(projects)
    print(MyDataFrame.to_string())

    print("")
    print("Статистические данные")
    print("")
    print("Для столбца номер: min=" + str(num_min) + " ; max=" + str(num_max) + " ; ave=" + str(num_ave) + " ; disp=" + str(num_disp) + " ; std=" + str(num_std) + " ; median=" + str(num_median) + " ; mode=" + str(num_mode))
    print("Для столбца зарплата: min=" + str(sal_min) + " ; max=" + str(sal_max) + " ; ave=" + str(sal_ave) + " ; disp=" + str(sal_disp) + " ; std=" + str(sal_std) + " ; median=" + str(sal_median) + " ; mode=" + str(sal_mode))
    print("Для столбца проекты: min=" + str(pjt_min) + " ; max=" + str(pjt_max) + " ; ave=" + str(pjt_ave) + " ; disp=" + str(pjt_disp) + " ; std=" + str(pjt_std) + " ; median=" + str(pjt_median) + " ; mode=" + str(pjt_mode))

if __name__ == '__main__':

    print("Процесс генерации")
    print("")

    generate_data()

    print("")
    print("CSV файл создан (Нажмите Enter)")
    input()

    print("Numpy статистика")
    print("")

    nup_statistic()

    print("")
    print("CSV файл проанализирован (Нажмите Enter)")
    input()

    print("Pandas статистика")
    print("")

    pads_statistic()

    print("")
    print("CSV файл проанализирован (Нажмите Enter)")
    input()

    print("Графики")
    print("")

    graphics()

    print("")
    print("CSV файл удалён")
    os.remove("new_data.csv")