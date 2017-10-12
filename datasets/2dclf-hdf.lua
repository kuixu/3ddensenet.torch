--
--
require 'hdf5'

local M = {}

function M.loadhdf(opt, split)
    local imageInfo
    cachePath = opt.gendata:gsub("train",split)
    -- cachePath = opt.gendata.."_"..split..".h5"
    -- print("loading hdf ".. cachePath)
    local h5 = hdf5.open(cachePath,'r')
    local x = h5:read('data'):all():float()
    local y = h5:read('label'):all():int()
    local n = x:size(1)
    --local nTrain, _ = math.modf(n * 0.8)
    print(("Total: %s"):format(n))
    
    -- local tdata  = x:narrow(1,1,nTrain)
    -- local tlabel = y:narrow(1,1,nTrain) 
    -- if split == 'val' then
    --     tdata = x:narrow(1,nTrain+1, n - nTrain) 
    --     tlabel = y:narrow(1,nTrain+1, n - nTrain)
    -- end
    -- local vdata  = x:narrow(1,nTrain+1, n - nTrain)
    -- local tlabel = y:narrow(1,1,nTrain)
    -- local vlabel = y:narrow(1,nTrain+1, n - nTrain)
    --print(x:size())
    --print(y:size())
    local dataset = {
        data = x,
        labels = y+1
    }
    -- local valData = {
    --     data = vdata,
    --     labels = vlabel
    -- }
    imageInfo = {
      train = {},
      val = {}
    }
    imageInfo[split] = dataset
    return imageInfo
end

return M
