import glob
import os
from widowx_envs.utils.datautils.raw2lmdb import parse_args, run
import copy

def convert_single_room(args, inputfolder=None):
    if inputfolder is None:
        inputfolder = args.input_folder

    folders = glob.glob(inputfolder + '/*')
    for folder in folders:
        newargs = copy.deepcopy(args)
        newargs.input_folder = folder
        run(newargs)

def convert_all_rooms(args):
    folders = glob.glob(args.input_folder + '/*')
    for folder in folders:
        newargs = copy.deepcopy(args)
        print('procesing room', folder)
        convert_single_room(newargs, folder)

if __name__ == '__main__':
    args = parse_args()
    # convert_single_room(args)
    convert_all_rooms(args)

