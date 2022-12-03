import string

import pandas as pd


def day1():

    cal_per_elf = sorted([sum([int(cal) for cal in elf.split('\n')]) for elf in open('input/day1.txt', 'r').read()[:-1].split('\n\n')])
    
    print('#################')
    print(f'Day 1.1 = {cal_per_elf[-1]}')
    print(f'Day 1.2 = {sum(cal_per_elf[-3:])}')



def day2():

    df = pd.read_csv('input/day2.txt', sep=' ', header=None, names=['a', 'b'])

    sol1 = (
        df
        .assign(
            other = df.a.replace({'A': 'r', 'B': 'p', 'C': 's'}),
            me = df.b.replace({'X': 'r', 'Y': 'p', 'Z': 's'}),
            match = lambda x: x.other + x.me,
            selec_points = lambda x: x.me.replace({'r': 1, 'p': 2, 's': 3}),
            draw_points = lambda x: 3*(x.other == x.me),
            victory_points = lambda x: 6*x.match.isin(['rp', 'ps', 'sr'])
        )
        .select_dtypes('number')
        .sum()
        .sum()
    )
    
    sol2 = (
        df
        .assign(
            other = df.a.replace({'A': 'r', 'B': 'p', 'C': 's'}),
            result = df.b.replace({'X': 'loss', 'Y': 'draw', 'Z': 'victory'}),
            loss = lambda x: x.other.replace({'p': 'r', 'r': 's', 's': 'p'}).where(x.result=='loss'),
            draw = lambda x: x.other.where(x.result=='draw'),
            victory = lambda x: x.other.replace({'p': 's', 'r': 'p', 's': 'r'}).where(x.result=='victory'),
            me = lambda x: x.loss.combine_first(x.draw).combine_first(x.victory),
            match = lambda x: x.other + x.me,
            selec_points = lambda x: x.me.replace({'r': 1, 'p': 2, 's': 3}),
            draw_points = lambda x: 3 * (x.other == x.me),
            victory_points = lambda x: 6 * x.match.isin(['rp', 'ps', 'sr'])
            )
        .select_dtypes('number')
        .sum()
        .sum()
    )

    print('#################')
    print(f'Day 2.1 = {sol1}')
    print(f'Day 2.2 = {sol2}')


def day3():
    
    df = pd.read_csv('input/day3.txt', header=None, names=['rucksack'])

    sol1 = (
        df
        .assign(
            comp_1 = df.rucksack.apply(lambda x: set(x[:len(x)//2])),
            comp_2 = df.rucksack.apply(lambda x: set(x[len(x)//2:])),
            repeated = lambda x: [(a & b).pop() for a, b in zip(x.comp_1, x.comp_2)],
            is_upper = lambda x: x.repeated.str.isupper(),
            priority = lambda x: 26 * x.is_upper + x.repeated.apply(lambda x: 1 + string.ascii_lowercase.index(x.lower()))
        )
        .select_dtypes('number')
        .sum()
        .sum()
    )

    sol2 = (
        df
        .assign(
            rucksack2 = df.rucksack.shift(-1),
            rucksack3 = df.rucksack.shift(-2)
        )
        .dropna()
        .iloc[range(0, df.shape[0], 3)]
        .assign(
            set1 = lambda x: x.rucksack.apply(lambda x: set(x)),
            set2 = lambda x: x.rucksack2.apply(lambda x: set(x)),
            set3 = lambda x: x.rucksack3.apply(lambda x: set(x)),
            common = lambda x: [(a & b & c).pop() for a, b, c in zip(x.set1, x.set2, x.set3)],
            is_upper = lambda x: x.common.str.isupper(),
            priority = lambda x: 26 * x.is_upper + x.common.apply(lambda x: 1 + string.ascii_lowercase.index(x.lower()))
        )
        .select_dtypes('number')
        .sum()
        .sum()
    )

    print('#################')
    print(f'Day 2.1 = {sol1}')
    print(f'Day 2.2 = {sol2}')



if __name__ == '__main__':

    day1()
    # day2()
    day3()