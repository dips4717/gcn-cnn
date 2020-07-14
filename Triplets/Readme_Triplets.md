# Generating triplets for training GCN-CNN-TRI model

*To construct positive and negative pairs, we compute a average IoU value between a pair of UIs. For every UI in training set, we need to compute IoU with remaining UIs. This is abit expensive computation. We share our generated triplets.

* Triplets used for all the experiments can be obtained from [here] ()

* If you want to obtain your own training triplets, then follow steps below.
	* Since it takes several hours to compute pairwise ious, we divide the training set into several segments, and later combine them.
	
	Run `python compute_pairwise_IoU` for several segment (batch of each 1000) of training UIs. 
	* Once you compute iou values for all the segments, combine the output pickle files into one.
	Run `python combine_segments.py` .
	* Next, run `get_APN_triplet_dict.py` to get the final dictionary of the triplets. 