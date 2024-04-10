def save_to_file(path, contents):
    fh = open(path, 'a')
    fh.write(contents + "\n")
    fh.close()

def save_args2file(path, args):
    fh = open(path, 'a')
    fh.write("\n===args===\n")
    for arg, value in vars(args).items():
        arg_item = f"{arg}: {value}"
        print(arg_item)
        fh.write(arg_item + '\n')
    fh.write("\n==========\n")
    fh.close()