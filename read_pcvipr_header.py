import csv


def read_pcvipr_header(in_path):
    try:
        with open(in_path + "pcvipr_header.txt") as hfile:
            reader = csv.reader(
                hfile,
                delimiter=" ",
            )
            out_dict = {}
            for row in reader:
                # print("{}:{}".format(row[0], row[1]))
                out_dict.update({row[0]: row[1]})
        return out_dict
    except FileNotFoundError:
        print("Header file does not exist")
