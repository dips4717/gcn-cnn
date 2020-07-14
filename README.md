# Learning UI Similarity using Graph Networks

<div align="center">
  <img src="data/gcncnn_arch.png"/>
</div>


## Datasets
### RICO Datasets
* Download RICO dataset from [rico](https://interactionmining.org/rico) (Optional)
* We use semantic UI screenshots and annotations. Simplified annotation for semantic RICO UIs is given in *data/rico_box_info_list.pkl*
* Dataset partition sets (train/gallery/query) in data/
* 

### GoogleUI Datasets
* We release GoogleUI dataset. GoogleUI is a new dataset of 18.5K UX designs obtained from the web.
* Dataset and annoations can be downloaded from (Google Drive)[https://drive.google.com/drive/folders/1LdhtDfiv48jSAbaLmL3rbrLBi4ZByd6p?usp=sharing]



## Evaluation code and model
To evaluate the model:
* Prior to evaluation/training, prepare graph represenatation for UIs following:
	* Run graph_scripts/cal_geometry_feat.py. This will compute the geometric features for all rico UIs
	* Run graph_scripts/build_geomerty_graph.py. This will pre-construct the graph data for UIs; saved under *graph_data/*
