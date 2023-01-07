import argparse
import random
import pickle
from primitives import *
from logistic_regression import MyLogisticRegression
from parser import Parser
import os

parser = argparse.ArgumentParser()
parser.add_argument("files_dir")
parser.add_argument("plagiat1_dir")
parser.add_argument("plagiat2_dir")
parser.add_argument("--model")
parser.add_argument("--iterations")

args = parser.parse_args()

assert args.model and len(args.model.split('.')) == 2 and args.model.split('.')[1] == 'pkl'

model = MyLogisticRegression()
code_parser = Parser()

files_elements = os.listdir(args.files_dir)
plagiat1_elements = os.listdir(args.plagiat1_dir)
plagiat2_elements = os.listdir(args.plagiat2_dir)

iterations = len(files_elements) * len(plagiat1_elements) + len(files_elements) * len(plagiat2_elements)
if args.iterations:
    iterations = int(args.iterations)

pairs = []

for i in files_elements:
    for j in plagiat1_elements:
        pairs.append((args.files_dir + "/" + i, args.plagiat1_dir + "/" + j, int(i == j)))

for i in files_elements:
    for j in plagiat2_elements:
        pairs.append((args.files_dir + "/" + i, args.plagiat1_dir + "/" + j, int(i == j)))

assert iterations <= len(pairs)

random.shuffle(pairs)

origin_X = []
origin_y = []

for iterations_cnt in range(iterations):
    first_pair_file = open(pairs[iterations_cnt][0], "r").read()
    second_pair_file = open(pairs[iterations_cnt][1], "r").read()
    column = code_parser.get_info_raw(first_pair_file, second_pair_file)
    origin_X.append(column)
    origin_y.append([pairs[iterations_cnt][2]])

X = MyMatrix(origin_X)
y = MyMatrix(origin_y)

model.fit(X, y)
pickle.dump(model, open(args.model, 'wb'))