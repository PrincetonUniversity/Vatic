\ Model matrix1
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  x[0] + x[1] + 2 x[2]
Subject To
 c[0]: x[0] + 2 x[1] + 3 x[2] <= 4
 c[1]: - x[0] - x[1] <= -1
Bounds
End
