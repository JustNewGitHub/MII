from random import randint as ri
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print("Вариант 6")

    try:
        print("Введите размер матрицы")
        n = int(input())
        if (n % 2) | (n < 1):
            raise Exception("Размерность должна быть чётной")
        print("Введите множитель")
        k = int(input())

        a = np.array([[ri(-10, 10) for j in range(n)] for i in range(n)])
        print("Матрица A")
        print(a)

        b = a[0: n // 2, 0: n // 2]
        c = a[0: n // 2, n // 2:]
        d = a[n // 2:, 0:n // 2]
        e = a[n // 2:, n // 2:]

        print("Подматрица b")
        print(b)
        print("Подматрица c")
        print(c)
        print("Подматрица d")
        print(d)
        print("Подматрица e")
        print(e)

        f = a.copy()

        firstpart = np.tril(a, k=0)
        secondpart = np.tril(a.transpose(), k=0)
        test = (firstpart==secondpart).all()
        print('Результат симметричности матрицы по побочной диагонали '+str(test))

        if (test):
            b = np.flipud(a[n // 2:, 0:n // 2])
            d = np.flipud(a[0: n // 2, 0: n // 2])
        else:
            e = a[n // 2:, 0:n // 2]
            d = a[n // 2:, n // 2:]

        print("Новые значения")
        print("Подматрица b")
        print(b)
        print("Подматрица c")
        print(c)
        print("Подматрица d")
        print(d)
        print("Подматрица e")
        print(e)

        f = np.concatenate((np.concatenate((b, c), axis=0), np.concatenate((d, e), axis=0)), axis=1);

        print("Матрица F")
        print(f)

        print("Определитель матрицы A")
        det = np.linalg.det(a)
        print(det)

        print("Сумма диагональных элементов F")
        dia = np.trace(f)
        print(dia)

        if det > dia:
            ainv = np.linalg.inv(a)
            at = np.transpose(a)
            finv = np.linalg.inv(f)
            result = np.subtract(np.multiply(ainv, at), np.multiply(k, finv))
        else:
            at = np.transpose(a)
            g = np.tril(a)
            ft = np.transpose(f)
            result = np.multiply(np.subtract(np.add(at, g), ft), k)

        print("Результат")
        print(result)

        graph1 = plt.figure()
        bsize = len(b) * len(b[0])
        x = np.linspace(0, bsize, bsize)
        y = np.asarray(b).reshape(-1)
        plt.plot(x, y)
        plt.title("Подматрица B")
        plt.show()

        graph2 = plt.figure()
        csize = len(c) * len(c[0])
        x = np.linspace(0, csize, csize)
        y = np.asarray(c).reshape(-1)
        plt.plot(x, y)
        plt.title("Подматрица C")
        plt.show()

        graph3 = plt.figure()
        dsize = len(d) * len(d[0])
        x = np.linspace(0, dsize, dsize)
        y = np.asarray(d).reshape(-1)
        plt.plot(x, y)
        plt.title("Подматрица D")
        plt.show()

        graph4 = plt.figure()
        esize = len(e) * len(e[0])
        x = np.linspace(0, esize, esize)
        y = np.asarray(e).reshape(-1)
        plt.plot(x, y)
        plt.title("Подматрица E")
        plt.show()

    except ValueError:
        print("Ваше значение не является числом")
    except Exception as exc:
        print(exc)
        print("Перезапустите прогрпмму")



