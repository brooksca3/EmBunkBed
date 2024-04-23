This directory contains the data (prepared in different ways for different trainings) that we use for training

These notes are for internal use -- the data itself is too large to git commit

full uniprot (/scratch/gpfs/cabrooks/deleteme_data/output_sequences.txt) has 62,254,414 sequences. We will take 1% of these sequences and truncate them to the first 512 amino acids. We will call this "623K_truncated_512_train.txt"

the uniprot validation set has 1594640 lines. We'll similarly take 1% of these and truncate them to the first 512 amino acids. We will call this "16K_truncated_512_validation.txt"