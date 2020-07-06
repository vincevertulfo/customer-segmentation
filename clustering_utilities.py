from yellowbrick.cluster import KElbowVisualizer

def find_optimal_k(model, k_fold, X):
  '''
  Returns optimal no. of k clusters

  Parameters:
      model (object): Algorithm object (e.g. KMeans)
      k_fold (int): Check up to how many k

  Returns:
     The optimal no. of k clusters
  '''
  visualizer = KElbowVisualizer(model, k=(2,k_fold))
  visualizer.fit(X)

  return visualizer.elbow_value_