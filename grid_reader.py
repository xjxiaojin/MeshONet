import time
import os
from numpy import *
import numpy as np
import matplotlib.pyplot as plt  # Import matplotlib.pyplot module


def extractHeadInfo(f, row):  # Read the first few lines before coordinates start
    x_num_list = []
    for i in range(row):
        t_str = f.readline().strip()
        t_str = t_str.replace(' ', ',')
        t_list = t_str.split(',')
        # print(t_list)
        for index in t_list:
            if index != '':
                x_num_list.append(int(index))
    return x_num_list


def readFile(filepath, flg):
    with open(filepath, 'r') as f:
        t_str = f.readline().strip()  # Read the first line
        # print(t_str)
        t_str = t_str.replace(' ', '')  # Remove spaces in the header line
        row = int(t_str)  # Convert to integer, indicates number of header lines

        # Process header information
        x_num_list = extractHeadInfo(f, row)

        x_list = []
        y_list = []
        z_list = []
        all_list = []

        x_row = 0
        x = x_num_list[0] * x_num_list[1]
        if x % 4 == 0:
            x_row = int(x / 4)
        else:
            x_row = int(x / 4) + 1

        # Read x coordinates
        for i in range(x_row):
            t_str = f.readline().strip()
            if not t_str:
                break
            t_str = t_str.replace('\n', '')
            t_str = t_str.replace(' ', ',')
            templist = t_str.split(',')
            for index in templist:
                if index != '':
                    x_list.append(float(index))

        # Read y coordinates
        for i in range(x_row):
            t_str = f.readline().strip()
            if not t_str:
                break
            t_str = t_str.replace('\n', '')
            t_str = t_str.replace(' ', ',')
            templist = t_str.split(',')
            for index in templist:
                if index != '':
                    y_list.append(float(index))

        # Read z coordinates if flg is True
        if flg:
            for i in range(x_row):
                t_str = f.readline().strip()
                if not t_str:
                    break
                t_str = t_str.replace('\n', '')
                t_str = t_str.replace(' ', ',')
                templist = t_str.split(',')
                for index in templist:
                    if index != '':
                        z_list.append(float(index))

        # Combine coordinates into all_list
        if flg:
            for i in range(len(x_list)):
                temp_list = [(x_list[i]), (y_list[i]), (z_list[i])]
                all_list.append(temp_list)
        else:
            for i in range(len(x_list)):
                temp_list = [(x_list[i]), (y_list[i])]
                all_list.append(temp_list)

        return all_list, x_list, y_list, z_list, x_num_list



