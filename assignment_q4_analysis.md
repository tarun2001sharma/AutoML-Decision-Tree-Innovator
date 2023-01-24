
##### Learning Tree and Prediction by varying M

<p align="center">
  <img width="400" src="learn_M.png">
</p>

<p align="center">
  <img width="400" src="predict_M.png">
</p>

1. Theoretical train time complexity of O(M) when number of examples is kept constant.
2. In the learning graph shown above, the time taken increases with M but not linearly.
3. Test time complexity would be O(depth) since we have to move from root to a leaf node of the decision tree.

----------------------------------------------
 
##### Learning tree and Prediction by varying N

<p align="center">
  <img width="400" src="learn_N.png">
</p>

<p align="center">
  <img width="400" src="predict_N.png">
</p>

1. Theoretical train time complexity of O( N * logN ) when number of attributes is kept constant.
2. In the learning graph shown above, the time taken increases with N.
3. Test time complexity would be O(depth) since we have to move from root to a leaf node of the decision tree.