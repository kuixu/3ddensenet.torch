
local mytest = torch.TestSuite()

local tester = torch.Tester()

require('torch')
require('nn')
local opts = require 'opts' 
local models = require 'models/init_den' 
local checkpoints = require 'checkpoints' 

local opt = opts.parse(arg)

function mytest.net()
    local model, criterion = models.setup(opt, checkpoint)
    container = nn.Container()
    container:add(model)
    print(torch.max(container:getParameters():float()))
    model:zeroGradParameters()
    --print(torch.max(model:getParameters():float()))
    -- print(model)
end



-- opt.netType='3dunethkm-drop-deep'
opt.netType='3dnin_fc4'
opt.netType='voxnet3'
opt.netType='3dnin_fc4_vox'
opt.netType='3dnin_fc_vox_ft'
opt.netType='3dunethkm-drop-deep'
opt.netType='3dunethkm'
opt.netType='subvolumesup'
opt.netType='unet'
-- test 3ddensenet
opt.growthRate=12
opt.depth=40
opt.nClasses=21
opt.dataset='pdb'
opt.netType='3ddensenet_modelnet'
-- Run tests
tester:add(mytest)
tester:run()
