# lab.py


import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# QUESTION 0
# ---------------------------------------------------------------------


def consecutive_ints(ints):
    """
    consecutive_ints tests whether a list contains two 
    adjacent elements that are consecutive integers.
    :param ints: a list of integers
    :returns: a boolean value if ints contains two 
    adjacent elements that are consecutive integers.
    :Example:
    >>> consecutive_ints([5,3,6,4,9,8])
    True
    >>> consecutive_ints([1,3,5,7,9])
    False
    """

    if len(ints) == 0:
        return False

    for k in range(len(ints) - 1):
        diff = abs(ints[k] - ints[k+1])
        if diff == 1:
            return True

    return False


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def median_vs_average(nums):
    '''
    median takes a non-empty list of numbers,
    returning a boolean of whether the median is
    greater or equal than the average
    If the list has even length, it should return
    the mean of the two elements in the middle.
    :param nums: a non-empty list of numbers.
    :returns: bool, whether median is greater or equal than average.
    
    :Example:
    >>> median_vs_average([6, 5, 4, 3, 2])
    True
    >>> median_vs_average([50, 20, 15, 40])
    False
    >>> median_vs_average([1, 2, 3, 4])
    True
    '''
    return np.median(nums) >= np.mean(nums)


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def same_diff_ints(ints):
    """
    same_diff_ints tests whether a list contains
    two list elements i places apart, whose distance 
    as integers is also i.
    :param ints: a list of integers
    :returns: a boolean value if ints contains two 
    elements as described above.
    :Example:
    >>> same_diff_ints([5,3,1,5,9,8])
    True
    >>> same_diff_ints([1,3,5,7,9])
    False
    """
    a = 1
    for i in range(len(ints)):
        while (i + a) < len(ints):
            if (a - i) == (ints[a] - ints[i]):
                return True
            a += 1
    return False



# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def n_prefixes(s, n):
    """
    n_prefixes returns a string of n
    consecutive prefix of the input string.

    :param s: a string.
    :param n: an integer

    :returns: a string of n consecutive prefixes of s backwards.
    :Example:
    >>> n_prefixes('Data!', 3)
    'DatDaD'
    >>> n_prefixes('Marina', 4)
    'MariMarMaM'
    >>> n_prefixes('aaron', 2)
    'aaa'
    """
    new_str = ""
    while n > 0:
        for i in range(0, n):
            new_str += s[i]
        n -= 1
    return new_str


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def exploded_numbers(ints, n):
    """
    exploded_numbers returns a list of strings of numbers from the
    input array each exploded by n.
    Each integer is zero padded.

    :param ints: a list of integers.
    :param n: a non-negative integer.

    :returns: a list of strings of exploded numbers. 
    :Example:
    >>> exploded_numbers([3, 4], 2) 
    ['1 2 3 4 5', '2 3 4 5 6']
    >>> exploded_numbers([3, 8, 15], 2)
    ['01 02 03 04 05', '06 07 08 09 10', '13 14 15 16 17']
    """
    exploded = []
    a = 0
    max_len = len(str(max(ints)))
    while a < len(ints):
        str_explode = ""
        lowest = ints[a] - n
        highest = ints[a] + n
        x = lowest
        while x <= highest:
            str_x = str(x)
            zeroes = ""
            while (len(str_x) + len(zeroes)) < max_len:
                zeroes += "0"
            str_x = zeroes + str_x
            str_explode += str_x + " "
            x += 1
        str_explode = str_explode[:-1]
        exploded.append(str_explode)
        a += 1
    return exploded



# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def last_chars(fh):
    """
    last_chars takes a file object and returns a 
    string consisting of the last character of the line.
    :param fh: a file object to read from.
    :returns: a string of last characters from fh
    :Example:X
    >>> fp = os.path.join('data', 'chars.txt')
    >>> last_chars(open(fp))
    'hrg'
    """
    ret_str = ""
    with open(fh.name, "r") as f:
        for line in f:
            ret_str += line[-2]
    return ret_str


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def arr_1(A):
    """
    arr_1 takes in a numpy array and
    adds to each element the square-root of
    the index of each element.
    :param A: a 1d numpy array.
    :returns: a 1d numpy array.
    :Example:
    >>> A = np.array([2, 4, 6, 7])
    >>> out = arr_1(A)
    >>> isinstance(out, np.ndarray)
    True
    >>> np.all(out >= A)
    True
    """
    B = np.array([])
    for i in range(len(A)):
        B = np.append(B, A[i] + i **(1/2))
    return B
        

def arr_2(A):
    """
    arr_2 takes in a numpy array of integers
    and returns a boolean array (i.e. an array of booleans)
    whose ith element is True if and only if the ith element
    of the input array is a perfect square.
    :param A: a 1d numpy array.
    :returns: a 1d numpy boolean array.
    :Example:
    >>> out = arr_2(np.array([1, 2, 16, 17, 32, 49]))
    >>> isinstance(out, np.ndarray)
    True
    >>> out.dtype == np.dtype('bool')
    True
    """
    ...
    B = np.array([])
    for item in A:
        root = np.sqrt(item)
        if int(root + 0.5) ** 2 == item:
            B = np.append(B, True)
        else:
            B = np.append(B, False)
    return B

def arr_3(A):
    """
    arr_3 takes in a numpy array of stock
    prices per share on successive days in
    USD and returns an array of growth rates.
    :param A: a 1d numpy array.
    :returns: a 1d numpy array.
    :Example:
    >>> fp = os.path.join('data', 'stocks.csv')
    >>> stocks = np.array([float(x) for x in open(fp)])
    >>> out = arr_3(stocks)
    >>> isinstance(out, np.ndarray)
    True
    >>> out.dtype == np.dtype('float')
    True
    >>> out.max() == 0.03
    True
    """
    B = np.array([])
    for i in range(len(A) - 1):
        B = np.append(B, round((A[i+1] - A[i]) / A[i], 2))
    return B

def arr_4(A):
    """
    Create a function arr_4 that takes in A and 
    returns the day on which you can buy at least 
    one share from 'left-over' money. If this never 
    happens, return -1. The first stock purchase occurs on day 0
    :param A: a 1d numpy array of stock prices.
    :returns: the day on which you can buy at least one share from 'left-over' money
    :Example:
    >>> import numbers
    >>> stocks = np.array([3, 3, 3, 3])
    >>> out = arr_4(stocks)
    >>> isinstance(out, numbers.Integral)
    True
    >>> out == 1
    True
    """
    leftover = 0
    for i in range(len(A)):
        stock = A[i]
        if leftover > stock:
            return i
        else:
            leftover += 20 - stock
        


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def salary_stats(salary):
    """
    salary_stats returns a series as specified in the notebook.
    :param salary: a dataframe of NBA salaries as found in `salary.csv`
    :return: a series with index specified in the notebook.
    :Example:
    >>> salary_fp = os.path.join('data', 'salary.csv')
    >>> salary = pd.read_csv(salary_fp)
    >>> out = salary_stats(salary)
    >>> isinstance(out, pd.Series)
    True
    >>> 'total_highest' in out.index
    True
    >>> isinstance(out.loc['duplicates'], bool)

    """
    stats_series = pd.Series()
    stats_series["num_players"] = salary.shape[0]
    stats_series["num_teams"] = salary.get("Team").nunique()
    stats_series["total_salary"] = sum(salary.get("Salary"))
    stats_series["highest_salary"] = salary.get("Salary").max()
    stats_series["avg_bos"] = round(salary[salary.get("Team") == "BOS"].get("Salary").mean(), 2)
    stats_series["third_lowest"] = (salary.sort_values("Salary").get("Player").iloc[2], salary.sort_values("Salary").get("Team").iloc[2] )
    last_names = []
    for name in salary.get("Player"):
        first_last = name.split(" ")
        last_names.append(first_last[1])
    last_counts = {}
    for name in last_names:
        if name in last_counts.keys():
            last_counts[name] += 1
        else:
            last_counts[name] = 1
    for value in last_counts.values():
        if value != 0:
            stats_series["duplicates"] = True
            break
    else:
        stats_series["duplicates"] = False
    highest_paid = salary.sort_values("Salary").get("Team").iloc[-1]
    stats_series["total_highest"] = salary[salary.get("Team") == highest_paid].get("Salary").sum()
    return stats_series

# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def parse_malformed(fp):
    """
    Parses and loads the malformed csv data into a 
    properly formatted dataframe (as described in 
    the question).
    :param fh: file handle for the malformed csv-file.
    :returns: a Pandas DataFrame of the data, 
    as specificed in the question statement.
    :Example:
    >>> fp = os.path.join('data', 'malformed.csv')
    >>> df = parse_malformed(fp)
    >>> cols = ['first', 'last', 'weight', 'height', 'geo']
    >>> list(df.columns) == cols
    True
    >>> df['last'].dtype == np.dtype('O')
    True
    >>> df['height'].dtype == np.dtype('float64')
    True
    >>> df['geo'].str.contains(',').all()
    True
    >>> len(df) == 100
    True
    >>> dg = pd.read_csv(fp, nrows=4, skiprows=10, names=cols)
    >>> dg.index = range(9, 13)
    >>> (dg == df.iloc[9:13]).all().all()
    True
    """
    formed = pd.DataFrame()
    i = 0
    cols = []
    with open(fp, "r") as f:
        for line in f:
            if line[-1] != '"':
                line = line[:-1]
            line = line.replace("\n","").replace(",,",",")
            items = line.split(",")
            if len(items) > 6:
                items = items[0:-1]
            if i == 0:
                for item in items:
                    formed[item] = 0
                cols = items
            else:
                geo = str(items[-2:])[1:-1]
                items = items[0:-2]
                geo = geo.replace('"',"").replace("'","").replace(" ","")
                items2 = []
                for item in items:
                    item = item.replace('"', "").replace("'","")
                    items2.append(item)
                items = items2
                items.append(geo)
                df_add = pd.DataFrame(items, index = cols)
                df_add = df_add.transpose()
                formed = formed.append(df_add)
            i += 1
    formed = formed.reset_index().drop(columns = ["index"])
    formed = formed.astype({"first":str, "last":str, "weight":float, "height":float, "geo":str})
    return formed