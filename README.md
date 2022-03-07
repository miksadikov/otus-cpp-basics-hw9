# OTUS C++ Basic курс

## Задание: "Проектная работа"
### Тема проекта: "Простейшая нейронная сеть".

В данной работе реализована простейшая нейронная сеть, состоящая 
из четырех входных нейронов и одного выходного.

В качестве функции активации используется [сигмоида](https://ru.wikipedia.org/wiki/%D0%A1%D0%B8%D0%B3%D0%BC%D0%BE%D0%B8%D0%B4%D0%B0), выход может принимать значения от 0 до 1.
Начальные веса (весовая матрица) имеют значениe 0,5.
В качестве примера для обучения используется датасет ["Ирисы Фишера"](https://ru.wikipedia.org/wiki/%D0%98%D1%80%D0%B8%D1%81%D1%8B_%D0%A4%D0%B8%D1%88%D0%B5%D1%80%D0%B0).
В данном проекте для простоты используется два вида ирисов из трех, имеющихcя в датасете - только setosa и versicolor.

На вход программы подается CSV-файл с данными для обучения, он имеет вид:
```
4.9,3.1,1.5,0.1,setosa
5.9,3.0,4.2,1.5,versicolor
5.0,3.2,1.2,0.2,setosa
6.0,3.4,4.5,1.6,versicolor
6.8,2.8,4.8,1.4,versicolor
и т.д.
```

Четыре числа - это некие параметры ирисов в сантиметрах, setosa и versicolor - это виды ирисов:
ирис щетинистый (Iris setosa) и ирис разноцветный (Iris versicolor).
Эти четыре числа подаются на наш входной слой, состоящий из четырех нейронов.

После обучения, на вход нейросети подается тестовый набор данных, взятый из того же датасета, но не испoльзовавшийся в обучении.
Для отображения метрик нейросети я воспользовался сторонним проектом: [accuracy-evaluation-cpp](https://github.com/ashokpant/accuracy-evaluation-cpp).

Вывод программы имеет следующий вид:
```
Train...
Epoch 1:
Accuracy Evaluation Results
=======================================
	Average System Accuracy(%): 49.3421
	System Error(%)           : 50.6579
	Precision (Micro)(%)      : 49.3421
	Recall (Micro)(%)         : 49.3421
	Fscore (Micro)(%)         : 49.3421
	Precision (Macro)(%)      : 49.3421
	Recall (Macro)(%)         : 49.342
	Fscore (Macro)(%)         : 49.342
Epoch 2:
Accuracy Evaluation Results
=======================================
	Average System Accuracy(%): 48.75
	System Error(%)           : 51.25
	Precision (Micro)(%)      : 48.75
	Recall (Micro)(%)         : 48.75
	Fscore (Micro)(%)         : 48.75
	Precision (Macro)(%)      : -nan
	Recall (Macro)(%)         : 50
	Fscore (Macro)(%)         : -nan
Epoch 3:
Accuracy Evaluation Results
=======================================
	Average System Accuracy(%): 56.25
	System Error(%)           : 43.75
	Precision (Micro)(%)      : 56.25
	Recall (Micro)(%)         : 56.25
	Fscore (Micro)(%)         : 56.25
	Precision (Macro)(%)      : 56.25
	Recall (Macro)(%)         : 56.3492
	Fscore (Macro)(%)         : 56.2996
Epoch 4:
Accuracy Evaluation Results
=======================================
	Average System Accuracy(%): 87.1626
	System Error(%)           : 12.8374
	Precision (Micro)(%)      : 87.1626
	Recall (Micro)(%)         : 87.1626
	Fscore (Micro)(%)         : 87.1626
	Precision (Macro)(%)      : 87.1626
	Recall (Macro)(%)         : 88.5886
	Fscore (Macro)(%)         : 87.8698
Epoch 5:
Accuracy Evaluation Results
=======================================
	Average System Accuracy(%): 100
	System Error(%)           : 0
	Precision (Micro)(%)      : 100
	Recall (Micro)(%)         : 100
	Fscore (Micro)(%)         : 100
	Precision (Macro)(%)      : 100
	Recall (Macro)(%)         : 100
	Fscore (Macro)(%)         : 100

************* Prediction: *************
Test dataset:
0 0 0 0 1 1 1 1 
Prediction result:
0 0 0 0 1 1 1 1 
```

Как видно, хватило всего 5 эпох обучения.

## Инструкция по сборке

Для сборки проекта необходимo выполнить следующие команды:
```
mkdir build && cd build
cmake ..
cmake --build .
```

