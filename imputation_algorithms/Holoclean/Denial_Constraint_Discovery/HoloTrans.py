import re

with open('hcvdcs.txt', 'r') as file:
    lines = file.readlines()
    result = []

with open('hcv_constraints.txt', 'w') as output_file:
    for line in lines:
        processed_line = line[4:-2].rstrip('\n').strip()
        word_list = processed_line.split()
        holo = "t1&t2"
        for word in word_list:
            if not word == 'and':
                tokens = re.split(r'\s*(==|<=|>=|<>|<|>)\s*', word)
                if tokens[1] == '==':
                    holo += "&EQ(" + tokens[0] + "," + tokens[2] + ")"
                if tokens[1] == '<=':
                    holo += "&LTE(" + tokens[0] + "," + tokens[2] + ")"
                if tokens[1] == '>=':
                    holo += "&GTE(" + tokens[0] + "," + tokens[2] + ")"
                if tokens[1] == '<':
                    holo += "&LT(" + tokens[0] + "," + tokens[2] + ")"
                if tokens[1] == '>':
                    holo += "&GT(" + tokens[0] + "," + tokens[2] + ")"
                if tokens[1] == '<>':
                    holo += "&IQ(" + tokens[0] + "," + tokens[2] + ")"
        output_file.write(holo + '\n')