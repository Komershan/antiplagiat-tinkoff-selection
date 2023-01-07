import argparse
import random
import pickle
from primitives import *
from logistic_regression import MyLogisticRegression
from parser import Parser
import os

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("scores")
parser.add_argument("--model")

args = parser.parse_args()

model = pickle.load(open(args.model, 'rb'))

code_parser = Parser()

origin_X = []

with open(args.input) as input_file:
    while True:
        line = input_file.readline()
        if not line:
            break
        print(line.strip().split())
        first_path, second_path = line.strip().split(' ')[0], line.strip().split(' ')[1]
        first_code = open(first_path, 'r').read()
        second_code = open(second_path, 'r').read()
        column = code_parser.get_info_raw(first_code, second_code)
        origin_X.append(column)

X = MyMatrix(origin_X)

predictions = model.predict_proba_external(X)

with open(args.scores, 'w') as scores_file:
    for index in range(predictions.shape[0]):
        scores_file.write(f"{predictions[(index, 0)]}\n")