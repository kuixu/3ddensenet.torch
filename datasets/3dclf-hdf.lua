--
--
require 'hdf5'

local M = {}

function M.loadhdf(opt, split)
    local imageInfo
    -- print(opt.gendata)
    cachePath = opt.gendata:gsub("train",split)
    -- cachePath = opt.gendata.."_"..split..".h5"
    -- print("loading file ".. cachePath .. " ...")
    local h5 = hdf5.open(cachePath,'r')
    local x = h5:read('data'):all():float()
    local y = h5:read('label'):all():int()
    if opt.labelstart0 then
	y = y + 1
    end
    local n = x:size(1)
    --local nTrain, _ = math.modf(n * 0.8)
    -- time = sys.execute('date +%Y%m%d%H%M%S')
    -- print(("[%s] Sample Num: %s"):format(time, n))
    
    local dataset = {
        data = x,
        labels = y
    }
    imageInfo = {
      train = {},
      val = {}
    }
    imageInfo[split] = dataset
    return imageInfo
end

return M
