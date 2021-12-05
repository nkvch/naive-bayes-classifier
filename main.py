import csv
import math
import numpy as np
import sys

def get_classes_data(filename, class_creating_attribute, train_percent, offset_percent=0):
  classes_with_train_data = {}
  test_data = []
  total_rows_num = 0
  train_rows_num = 0
  offset_num = 0
  attributes = []

  with open(filename, 'r') as wf:
    total_rows_num = sum(1 for row in csv.DictReader(wf, delimiter=';'))
    train_rows_num = round(total_rows_num * train_percent)
    offset_num = round(total_rows_num * offset_percent)

  with open(filename, 'r') as wf:
    reader = csv.DictReader(wf, delimiter=';')

    attributes = reader.fieldnames.copy()
    attributes.remove(class_creating_attribute)

    count = 0

    if offset_num:
      for test_row in reader:
        count += 1

        test_attributes = test_row.copy()
        test_attributes.pop(class_creating_attribute)

        test_data.append({
          'attributes': dict([attr, float(val)] for attr, val in test_attributes.items()),
          class_creating_attribute: test_row[class_creating_attribute],
        })

        if count >= offset_num:
          break


    for row in reader:
      count = count + 1

      class_of_row = row[class_creating_attribute]

      if not class_of_row in classes_with_train_data:
        classes_with_train_data[class_of_row] = {
          'class_size': 1,
        }

        for attr in attributes:
          classes_with_train_data[class_of_row][attr] = [float(row[attr])]

      else:
        classes_with_train_data[class_of_row]['class_size'] += 1

        for attr in attributes:
          classes_with_train_data[class_of_row][attr].append(float(row[attr]))
      
      if count >= offset_num + train_rows_num:
        break

    for test_row in reader:
      test_attributes = test_row.copy()
      test_attributes.pop(class_creating_attribute)

      test_data.append({
        'attributes': dict([attr, float(val)] for attr, val in test_attributes.items()),
        class_creating_attribute: test_row[class_creating_attribute],
      })
  
  return classes_with_train_data, train_rows_num, attributes, test_data

def get_normal_distribution(classes_with_train_data, attributes):
  classes_with_nd_data = {}

  for class_name in classes_with_train_data.keys():
    classes_with_nd_data[class_name] = classes_with_train_data[class_name].copy()

    for attr in attributes:
      data = classes_with_train_data[class_name][attr].copy()

      mean = np.mean(data)
      std_dev = np.std(data)

      classes_with_nd_data[class_name][attr] = {
        'mean': mean,
        'std_dev': std_dev,
      }

  return classes_with_nd_data

def probability_nd(data_point, nd):
  return (1 / (math.sqrt(2 * math.pi) * nd['std_dev'])) \
          * math.exp(-((data_point - nd['mean']) ** 2 / (2 * nd['std_dev'] ** 2)))

def classify(data_point, classes_with_nd_data, train_data_size):
  classes_scores = {}

  for class_name in classes_with_nd_data.keys():
    score = math.log(classes_with_nd_data[class_name]['class_size'] / train_data_size)

    attributes = data_point.keys()

    for attr in attributes:
      probability = probability_nd(data_point[attr], classes_with_nd_data[class_name][attr])

      if not probability:
        probability = sys.float_info.min

      score += math.log(probability)
    
    classes_scores[class_name] = score
  
  most_probable_class = max(classes_scores, key=classes_scores.get)
  
  return most_probable_class, classes_scores

def get_accurancy_data_split(filename, class_creating_attribute, train_percent, offset_percent=0):
  classes_with_train_data, train_data_size, attributes, test_data = get_classes_data(
    filename,
    class_creating_attribute,
    train_percent,
    offset_percent,
  )

  classes_with_nd_data = get_normal_distribution(classes_with_train_data, attributes)

  guessed = 0
  missed = 0

  for test_data_point in test_data:
    estimated_class, score_data = classify(test_data_point['attributes'], classes_with_nd_data, train_data_size)
    real_class = test_data_point[class_creating_attribute]

    if estimated_class == real_class:
      guessed += 1
    else:
      missed +=1

  accurancy = guessed/(guessed + missed)
  
  return accurancy

def get_accurancy_k_cross_validation(filename, class_creating_attribute, k):
  train_percent = 1/k
  accurancy = 0

  for i in range(k):
    ith_accurancy = get_accurancy_data_split(
      filename,
      class_creating_attribute,
      train_percent,
      offset_percent=i*train_percent,
    )

    accurancy += ith_accurancy

  accurancy = accurancy/k

  return accurancy



def main():
  filename = 'winequality-red.csv'
  class_creating_attribute = 'quality'

  data_split_accurancy = get_accurancy_data_split(filename, class_creating_attribute, train_percent=0.6)
  k_cross_accurancy = get_accurancy_k_cross_validation(filename, class_creating_attribute, k=5)

  print(data_split_accurancy)
  print(k_cross_accurancy)

  

if __name__ == '__main__':
  main()
