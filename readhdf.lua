require 'hdf5'  
local cachePath='data/BOX/emdb-48-56G.h5'
local h5 = hdf5.open(cachePath,'r')    
local x = h5:read('xdata'):all():float()
print(x:size())
local y = h5:read('ydata'):all():float()
print("yyy")   
local n = x:size(1)
local nTrain, _ = math.modf(n * 0.8)
print(("Total: %s, Train: %s, Val: %s"):format(n, nTrain, n - nTrain))
local tdata  = x:narrow(1,1,nTrain)
local vdata  = x:narrow(1,nTrain+1, n - nTrain)
local tlabel = y:narrow(1,1,nTrain)
local vlabel = y:narrow(1,nTrain+1, n - nTrain)
local trainData = { 
    data = tdata,
    labels = tlabel
}              
local valData = { 
    data = vdata,
    labels = vlabel
}              
imageInfo = {  
  train = trainData, 
  val = valData
}
torch.save("data/BOX/emdb-48.t7",imageInfo)
