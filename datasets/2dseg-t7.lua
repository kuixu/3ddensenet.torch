--
--

local M = {}

function M.loadt7(opt, split)
    local imageInfo
    -- print(opt.gendata)
    cachePath = opt.gendata:gsub("train",split)
    -- cachePath = opt.gendata.."_"..split..".h5"
    -- print("loading file ".. cachePath .. " ...")
    t7data = torch.load(cachePath)
    local dataset = {
        data = t7data['data'],
        labels = t7data['labels']
    }
    imageInfo = {
      train = {},
      val = {}
    }
    imageInfo[split] = dataset
    return imageInfo
end

return M
