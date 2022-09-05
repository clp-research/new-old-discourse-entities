# date: 02.02.22

# Compare output of labelling experiment from different models

import argparse


def compare(file1, file2):
    with open(file1, encoding='utf-8') as f1:
        with open(file2, encoding='utf-8') as f2:
            counter = 0
            differences = []
            tokens = []
            orig = []
            index = 0
            for l1, l2 in zip(f1, f2):
                if index == 0:
                    if l1 == "\n" and l2 == "\n":
                        continue
                    assert l1.startswith("[(")
                    t1 = l1.strip("[(,\n)]").split(",), (")
                    t2 = l2.strip("[(,\n)]").split(",), (")
                    assert t1 == t2
                    tokens = t1
                    index = 1
                    counter += 1
                elif index == 1:
                    gold1 = l1.strip("['\n]").split("', '")
                    gold2 = l2.strip("['\n]").split("', '")
                    assert gold1 == gold2
                    orig = gold1
                    index = 2
                elif index == 2:
                    output1 = l1.strip("['\n]").split("', '")
                    output2 = l2.strip("['\n]").split("', '")
                    if output1 != output2:
                        differences.append((tokens, orig, output1, output2))
                    tokens = []
                    orig = []
                    index = 0

    for sample in differences:
        print(f"token\ttrue\tmodel1\tmodel2")
        for t, o, m1, m2 in zip(*sample):
            print(f"{t}\t{o}\t{m1}\t{m2}")

    print(f"Found differences in {len(differences)} of {counter} segments")


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Compare model outputs')
    parser.add_argument('file1', help='path to output file from model 1', type=str)
    parser.add_argument('file2', help='path to output file from model 2', type=str)
    args = parser.parse_args()
    compare(args.file1, args.file2)
