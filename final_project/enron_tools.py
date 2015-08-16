import sys
sys.path.append("../tools/")

def calculate_fraction(data_dict, key_1, key_2):
    values = []

    for i in data_dict:
        value_1 = data_dict[i][key_1]
        value_2 = data_dict[i][key_2]

        if value_1 == "NaN" or value_2 == "NaN":
            values.append(0.)
        elif value_1 >= 0:
            values.append(float(value_1)/float(value_2))

    return values

def add_features(data_dict, feature_name, values):
    iterator = 0

    for i in data_dict:
        data_dict[i][feature_name] = values[iterator]
        iterator += 1

    return data_dict