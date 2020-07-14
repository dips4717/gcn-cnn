# Learning UI Similarity using Graph Networks

<div align="center">
  <img src="data/gcncnn_arch.png"/>
</div>


# Datasets
### RICO Datasets
* Download RICO dataset from [rico](https://interactionmining.org/rico) (Optional)
* We use semantic UI screenshots and annotations. Simplified annotation for semantic RICO UIs is given in *data/rico_box_info_list.pkl*
* Dataset partition sets (train/gallery/query) used for all experiments are in `data/UI_data.p` and 'UI_test_data.p'
* 

### GoogleUI Datasets
* We release GoogleUI dataset. GoogleUI is a new dataset of 18.5K UX designs collected from web.
* Dataset/annoations/descriptions can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1LdhtDfiv48jSAbaLmL3rbrLBi4ZByd6p?usp=sharing)



# Evaluation code and model
To evaluate the model:
* Prior to evaluation/training, prepare graph represenatations for UIs following steps below:
	* Run ` python graph_scripts/cal_geometry_feat.py`. This will compute the geometric features for all rico UIs
	* Run `python graph_scripts/build_geomerty_graph.py`. This will pre-construct the graph data for UIs; saved under *graph_data/*

* Run `evaluate.py` to get the performance metrics: top-k mIoU and mPixAcc


# Training
* To train GCN-CNN model
```
python train.py --batch_size 10 --decoder_model 'strided' --dim 1024 \
--use_directed_graph True 
```	
	* It is recommended to pre-compute the 25-Channel representations for all RICO UIsfor faster dataloading and training.
	* To do so: run `python compute_25Chan_Imgs.py` 
	* This will save all 25 Channel represenations for all UIs into data/25ChanImages

* To train GCN-CNN model using pre-computed 25-Channel representations
 ```
python train.py --batch_size 10 --decoder_model 'strided' --dim 1024 \
--use_directed_graph True \
--use_precomputed_25Chan_imgs True\
--Channel25_img_dir 'data\25ChanImages
```		
	
