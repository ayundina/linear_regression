import numpy as np
import csv


def get_data(file_name):
    first_col = []
    second_col = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader, None)
        for row in reader:
            first_col.append(float(row[0]))
            second_col.append(float(row[1]))
    return np.array(first_col), np.array(second_col)


def write_parameters(csv_filename, params):
    with open(csv_filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['intercept', 'coefficient'])
        writer.writerow(params)
