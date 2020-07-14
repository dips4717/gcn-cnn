# Learning UI Similarity using Graph Networks

<div align="center">
  <img src="data/gcncnn_arch.png"/>
</div>


## Evaluation code and model
To get the performance metrics:
Item 1 Run graph_scripts/ cal_geometry_feat.py. This will compute the geometric features for all rico UIs
Item 2 Rn graph_scripts/build_geomerty_graph.py. This will pre-construct the graph data for UIs; saved under *graph_data/*
