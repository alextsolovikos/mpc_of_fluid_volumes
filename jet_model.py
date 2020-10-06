import numpy as np
import argparse





if __name__ == '__name__':

    # Input arguments
    parser = argparse.ArgumentParser(description = 'Compute a DMDcsp model of the vertical velocity component in the field inside the control grid.')
    parser.add_argument('--train', default=False, action='store_true', help='Flag to train a new DMDcsp model.')
    parser.add_argument('--training_data', dest='training_data', help='Training snapshot file location.')
    parser.add_argument('--test_data', dest='test_data', help='Test snapshot file location.')
    parser.add_argument('--grid', dest='grid_file', help='Grid file location.')
    args = parser.parse_args()
    























