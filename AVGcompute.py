from pandas import read_csv
import argparse

parser = argparse.ArgumentParser(description='Software to compute mean and std from file')
parser.add_argument("-f", "--filename", nargs='?', help='Name of the file to read')
parser.add_argument("-m", "--mode", nargs='?', choices=['0','1'], default = 0)
args = parser.parse_args()
df = read_csv(args.filename)
data = df.values
if(args.mode == 0):
    print("mean IoU:", df['IoU'].mean(), "std IoU:", df['IoU'].std())
else:
    print("mean Time:", df['time'].mean(), "std time:", df['time'].std())

