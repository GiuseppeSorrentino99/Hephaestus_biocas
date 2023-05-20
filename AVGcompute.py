from pandas import read_csv
import argparse

parser = argparse.ArgumentParser(description='Software to compute mean and std from file')
parser.add_argument("-f", "--filename", nargs='?', help='Name of the file to read')
args = parser.parse_args()
df = read_csv(args.filename)
data = df.values
print("mean IoU:", df['IoU'].mean(), "std IoU:", df['IoU'].std())