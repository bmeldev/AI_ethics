import json
import sys
import pandas as pd

from verifauth1 import similarity, readibility, avg_syl
from verifauth2 import compare


def load(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)


def process(config):
    infile = config['input']
    jobs = config['jobs']

    try:
        sheets_dict = pd.read_excel(infile, sheet_name=None)
    except Exception as e:
        print(f'Error loading file: {e}')
        sys.exit(1)

    output = {}
    index = 0

    for job in jobs:
        sheet = job['sheet']
        pre = job['pre']
        post = job['post']

        if sheet not in sheets_dict:
            print(f'Warning: Sheet {sheet} not found in {infile}. Skipping...')
            continue

        df = sheets_dict[sheet]

        if pre not in df.columns or post not in df.columns:
            print(f'Warning: One or both tabs ({pre}, {post}) not found in sheet {sheet}. Skipping...')
            continue

        print(f'\n=== Processing {sheet}: {pre} & {post} ===')

        mask = df[pre].notna() & (df[pre] != '') & df[post].notna() & (df[post] != '')
        df = df.copy()
        df.loc[mask, 'len_pre'] = df.loc[mask, pre].str.split().str.len()
        df.loc[mask, 'len_post'] = df.loc[mask, post].str.split().str.len()
        # df.loc[mask, 'zp_pre'] = [readibility(x) for x in df.loc[mask, pre]]
        # df.loc[mask, 'zp_post'] = [readibility(y) for y in df.loc[mask, post]]
        # df.loc[mask, 'avgsyl_pre'] = [avg_syl(x) for x in df.loc[mask, pre]]
        # df.loc[mask, 'avgsyl_post'] = [avg_syl(y) for y in df.loc[mask, post]]
        df.loc[mask, 'sim1'] = [get_sim1(x, y) for x, y in zip(df.loc[mask, pre], df.loc[mask, post])]
        df.loc[mask, 'sim2'] = [compare(x, y) for x, y in zip(df.loc[mask, pre], df.loc[mask, post])]


        index += 1
        output[sheet + str(index)] = df

    with pd.ExcelWriter('output.xlsx') as writer:
        for name, df in output.items():
            valid_name = str(name)[:31]  # truncate name to 31 chars (limit excel)
            df.to_excel(writer, sheet_name=valid_name, index=False)


def get_sim1(x, y):
    score = similarity(x, y)
    print(x)
    print(len(x.split()))
    print(y)
    print(len(y.split()))
    print(score)
    print('---')
    return score


def get_readibility(s):
    score = readibility(s)
    print(f'readibility = {score}')
    return score


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python script.py config.json')
        sys.exit(1)

    config = load(sys.argv[1])
    process(config)
