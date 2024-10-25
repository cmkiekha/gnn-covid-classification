"""

1.) Load all the data


Cross validation:
splits means take all the data, make many tr/te sets
e.g. k=5 fold cross validation means make 5 different tr/te sets
so e.g. if you have 100 datapoints in total, 
- 1st split use the first 20 points for testing, the rest for training
- 2nd split use points 20-40 for testing, rest for training
- and so on
For (tr, te) in splits:
    do some stuff
    evaluate it
average the evaluation metrics


for (tr, te) in splits:
    - Train WGAN on tr set
    - Generate samples with the WGAN
    - Measure KS(test set, generated samples)
Report the average KS statistic across the splits



for (tr, te) in splits:
    - Train WGAN on tr set
    - Generate samples with the WGAN, augment training set only
    - Train classifier 
    - Evaluate classifier on the testing set (without any synthetic data)
Report average metrics across splits

"""


n_datapoints = dataset.shape[0] # total number of datapoints
random_idxs = torch.randperm(n_datapoints) # this will create a random shuffling of (1, 2, ..., n_datapoints)

# n_datapoints = 5
# random_idxs = [4, 2, 3, 0, 1]

for test_index in random_idxs:
    train_idxs = [k for k in range(n_datapoints) if k!= test_index]

    train_data = dataset[train_idxs]
    test_data = dataset[test_index]

    # After this do WGAN, classifier, eval

