import string
import re
from itertools import product
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
    print(f'Day 3.1 = {sol1}')
    print(f'Day 3.2 = {sol2}')


def day4():

    df = pd.read_csv('input/day4.txt', header=None, names=['elf1', 'elf2'])

    sol1 = (
        df
        .assign(
            set1 = lambda x: x.elf1.apply(lambda x: eval('set(range(' + x.replace('-', ',') + '+1))')),
            set2 = lambda x: x.elf2.apply(lambda x: eval('set(range(' + x.replace('-', ',') + '+1))')),
            intersection = lambda x: [(a & b) for a, b in zip(x.set1, x.set2)],
            overlap = lambda x: (x.set1 == x.intersection) | (x.set2 == x.intersection)
        )
        ['overlap']
        .sum()
    )

    sol2 = (
        df
        .assign(
            set1 = lambda x: x.elf1.apply(lambda x: eval('set(range(' + x.replace('-', ',') + '+1))')),
            set2 = lambda x: x.elf2.apply(lambda x: eval('set(range(' + x.replace('-', ',') + '+1))')),
            intersection = lambda x: [(a & b) for a, b in zip(x.set1, x.set2)],
            overlap = lambda x: x.intersection.apply(lambda x: len(x) > 0)
        )
        ['overlap']
        .sum()
    )

    print('#################')
    print(f'Day 4.1 = {sol1}')
    print(f'Day 4.2 = {sol2}')


def day5():
    
    lines = [line[:-1] for line in open('input/day5.txt', 'r').readlines()]
    
    moves = [[int(x) for x in line.split(' ')[slice(1,6,2)]] for line in lines[10:]]

    pile = dict(zip(range(1, 10), ['']*10))
    for line, (i, j) in product(lines[:8][::-1], zip(range(1, 10), range(1, len(lines[0]), 4))):
        pile[i] += line[j].replace(' ', '')

    def rearrange(pile, invert=True):
        for count, _from, _to in moves:
            take = pile[_from][-count:]
            if invert: take = take[::-1]
            pile[_from] = pile[_from][:-count]
            pile[_to] += take
        return ''.join([pile[i][-1] for i in range(1, 10)])
    
    sol1 = rearrange(pile.copy(), invert=True)
    sol2 = rearrange(pile.copy(), invert=False)
    
    print('#################')
    print(f'Day 5.1 = {sol1}')
    print(f'Day 5.2 = {sol2}')


def day6():

    df = pd.DataFrame([char for char in open('input/day6.txt', 'r').read()[:-1]], columns=['pos0'])

    sol1 = (
        df
        .assign(
            **{f'pos{lag}' : df.pos0.shift(lag) for lag in range(1, 4)},
            length = lambda x: x.nunique(axis=1)
        )
        .query('length == 4')
        .index[0]
        + 1
    )

    sol2 = (
        df
        .assign(
            **{f'pos{lag}' : df.pos0.shift(lag) for lag in range(1, 14)},
            length = lambda x: x.nunique(axis=1)
        )
        .query('length == 14')
        .index[0]
        + 1
    )

    print('#################')
    print(f'Day 6.1 = {sol1}')
    print(f'Day 6.2 = {sol2}')
    

def day7():

    lines = [line[:-1] for line in open('input/day7.txt', 'r').readlines()]

    re_file = re.compile('\d+ \D')

    folders = {}
    path = '/'

    for line in lines:
        if re_file.findall(line):
            n = len(path.split('/'))
            subpaths = ['/'] + ['/'.join(path.split('/')[:i]) + '/' for i in range(2,n)]
            for subpath in subpaths:
                if subpath not in folders: folders[subpath] = 0
                folders[subpath] += int(line.split()[0])
        elif line == '$ cd ..':
            path = path.rsplit('/', maxsplit=2)[0] + '/'
        elif line == '$ cd /':
            path = '/'
        elif line.startswith('$ cd '):
            path += line.split()[2] + '/'

    sol1 = sum([size for size in folders.values() if size <= 100_000])
    sol2 = min([size for size in folders.values() if size >= folders['/'] - 70_000_000 + 30_000_000])

    print('#################')
    print(f'Day 7.1 = {sol1}')
    print(f'Day 7.2 = {sol2}')


if __name__ == '__main__':

    # day1()
    # day2()
    # day3()
    # day4()
    # day5()
    # day6()
    day7()