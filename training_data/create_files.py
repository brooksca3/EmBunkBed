## file to create / organize some data for training
import random

MAX_LENGTH = 512
truncated = []
with open('/scratch/gpfs/cabrooks/deleteme_data/validation_sequences.txt') as f:
    lines = f.readlines()
    truncated = [line.strip()[:MAX_LENGTH] for line in lines]
random.shuffle(truncated)
with open('16K_truncated_512_validation.txt', 'w') as f:
    ## write it to a line separated file
    for line in truncated[:int(0.01 * len(truncated))]:
        f.write(line + '\n')