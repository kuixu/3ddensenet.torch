require 'hdf5'
function loadhdf(hdf5path)
    -- local imageInfo
    -- print(opt.gendata)
    -- cachePath = opt.gendata:gsub("train",split)
    -- cachePath = opt.gendata.."_"..split..".h5"
    -- print("loading file ".. cachePath .. " ...")
    local labelstart0 = false
    --local hdf5path = "/home/scs4850/kuixu/data/cryoem/insilico_res_map_2/x2_0.h5"
    
    print("reading ...")
    local h5 = hdf5.open(hdf5path, 'r')
    local x = h5:read('data'):all():float()
    local y = h5:read('label'):all():int()
    if labelstart0 then
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
    print("saving ")
    t7path = hdf5path:gsub(".h5",".t7") 
    torch.save(t7path, dataset)
    print("saved ")
    -- return imageInfo
end


cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Transfer hdf5 file into torch7 format file')
cmd:text()
cmd:text('Options')
cmd:option('-filepath','test.h5','h5 file path')
-- cmd:option('-booloption',false,'boolean option')
-- cmd:option('-stroption','mystring','string option')
cmd:text()

-- parse input params
opt = cmd:parse(arg)




loadhdf(opt.filepath)

