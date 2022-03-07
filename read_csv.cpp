#include "read_csv.h"

int readCsv(std::string csv_file_train, std::string csv_file_test,
            std::vector<float>& x, std::vector<float>& y, std::vector<float>& t,
            std::vector<int>& t_y) {
  std::ifstream infile_train(csv_file_train);
  if (!infile_train.is_open()) {
    std::cerr << "Can't open file " << csv_file_train << std::endl;
    return -1;
  }
  std::ifstream infile_test(csv_file_test);
  if (!infile_test.is_open()) {
    std::cerr << "Can't open file " << csv_file_test << std::endl;
    return -1;
  }
  std::vector<std::string> csv_lines;
  std::vector<std::string> floats;
  std::vector<std::string> classes;
  std::string line, elem;
  int i = 0;

  // Читаем файл с обучающими данными
  while (std::getline(infile_train, line, '\n')) {
    csv_lines.push_back(line);
  }

  // Разделяем набор признаков и название класса на два вектора
  for (const auto& ln : csv_lines) {
    std::stringstream ss(ln);
    while (getline(ss, elem, ',')) {
      if (i < 4) {
        floats.push_back(elem);
        i++;
      } else {
        classes.push_back(elem);
        i = 0;
      }
    }
  }

  // Создаем вектор с признаками
  for (const auto& f : floats) {
    x.push_back(std::stof(f));
  }

  // Создаем вектор с классами: setosa = 0, versicolor = 1
  for (const auto& c : classes) {
    if (c == "setosa") {
      y.push_back(0);
    } else {
      y.push_back(1);
    }
  }

  // Читаем файл с тестовыми данными
  csv_lines.clear();
  classes.clear();
  while (std::getline(infile_test, line, '\n')) {
    csv_lines.push_back(line);
  }

  // Из тестовых данных создаем отдельный вектор с признаками для тестирования
  for (const auto& ln : csv_lines) {
    std::stringstream ss(ln);
    while (getline(ss, elem, ',')) {
      if (i < 4) {
        t.push_back(std::stof(elem));
        i++;
      } else {
        classes.push_back(elem);
        i = 0;
      }
    }
  }

  // Создаем вектор с классами: setosa = 0, versicolor = 1
  for (const auto& c : classes) {
    if (c == "setosa") {
      t_y.push_back(0);
    } else {
      t_y.push_back(1);
    }
  }

  return 0;
}
