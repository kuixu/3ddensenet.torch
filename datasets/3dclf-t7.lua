--
--

local M = {}

function M.loadt7(opt, split)
    local imageInfo
    -- print(opt.gendata)
    cachePath = opt.gendata:gsub("train",split)
    -- cachePath = opt.gendata.."_"..split..".h5"
    t7data = torch.load(cachePath)
    local data   = t7data['data']
    local labels = t7data['labels'] 
    if opt.labelstart0 then
        labels = labels + 1
    end
    
    local dataset = {
        data = data,
        labels = labels
    }
    imageInfo = {
      train = {},
      val = {}
    }
    imageInfo[split] = dataset
    return imageInfo
end

return M
